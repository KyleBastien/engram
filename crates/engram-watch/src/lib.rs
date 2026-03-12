use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use engram_core::{
    ChunkMetadata, EmbeddingProvider, EngramError, Result, SourceConfig, WatcherConfig,
};
use engram_ingest::{
    content_hash, detect_language, ChunkerKind, MarkdownChunker, RawChunk,
    SlidingWindowChunker, TreeSitterChunker,
};
use engram_store::{
    chunks_path, embeddings_path, write_chunks_jsonl, write_embeddings_bin, PRECISION_F32,
};
use globset::{Glob, GlobSet, GlobSetBuilder};
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use tokio::sync::mpsc;

/// A file change event detected by the watcher.
#[derive(Debug, Clone)]
pub struct WatchEvent {
    pub source_name: String,
    pub rel_path: PathBuf,
    pub abs_path: PathBuf,
}

/// Source directory mapping with compiled glob patterns.
struct SourceMapping {
    name: String,
    root: PathBuf,
    include: GlobSet,
    exclude: GlobSet,
}

/// File watcher that monitors source directories for changes and re-indexes files.
pub struct FileWatcher {
    sources: Vec<SourceMapping>,
    ignore: GlobSet,
    debounce_ms: u64,
    store_root: PathBuf,
}

impl FileWatcher {
    /// Create a new FileWatcher from configuration.
    pub fn new(
        sources: &[SourceConfig],
        watcher_config: &WatcherConfig,
        store_root: &Path,
    ) -> Result<Self> {
        let mut mappings = Vec::new();
        for src in sources {
            let root = PathBuf::from(&src.path);
            let canonical = root.canonicalize().unwrap_or(root);
            mappings.push(SourceMapping {
                name: src.name.clone(),
                root: canonical,
                include: build_globset(&src.include)?,
                exclude: build_globset(&src.exclude)?,
            });
        }
        let ignore = build_globset(&watcher_config.ignore)?;
        Ok(Self {
            sources: mappings,
            ignore,
            debounce_ms: watcher_config.debounce_ms as u64,
            store_root: store_root.to_path_buf(),
        })
    }

    /// Match an absolute file path to a source, returning source name and relative path.
    /// Returns None if the path doesn't belong to any source or is filtered out.
    pub fn match_path(&self, abs_path: &Path) -> Option<(String, PathBuf)> {
        let canonical = abs_path.canonicalize().unwrap_or_else(|_| abs_path.to_path_buf());
        for source in &self.sources {
            if let Ok(rel) = canonical.strip_prefix(&source.root) {
                // Check watcher-level ignore patterns
                if self.ignore.is_match(rel) {
                    return None;
                }
                // Check source-level exclude patterns
                if !source.exclude.is_empty() && source.exclude.is_match(rel) {
                    return None;
                }
                // Check source-level include patterns (empty = include all)
                if !source.include.is_empty() && !source.include.is_match(rel) {
                    return None;
                }
                return Some((source.name.clone(), rel.to_path_buf()));
            }
        }
        None
    }

    /// Get the debounce duration.
    pub fn debounce_duration(&self) -> Duration {
        Duration::from_millis(self.debounce_ms)
    }

    /// Get the source root directory for a given source name.
    pub fn source_root(&self, name: &str) -> Option<&Path> {
        self.sources
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.root.as_path())
    }

    /// Get all directories to watch.
    pub fn watch_dirs(&self) -> Vec<&Path> {
        self.sources.iter().map(|s| s.root.as_path()).collect()
    }

    /// Re-chunk and re-embed a single file, writing results to the store without committing.
    ///
    /// Returns the number of chunks written, or 0 if the file was deleted/skipped.
    pub async fn reindex_file(
        &self,
        source_name: &str,
        rel_path: &Path,
        provider: &dyn EmbeddingProvider,
    ) -> Result<usize> {
        let source_root = self
            .source_root(source_name)
            .ok_or_else(|| EngramError::Config(format!("unknown source: {source_name}")))?;
        let abs_path = source_root.join(rel_path);

        // Handle deleted files: remove chunks and embeddings from store
        if !abs_path.exists() {
            if let Ok(cp) = chunks_path(&self.store_root, source_name, rel_path) {
                let _ = fs::remove_file(&cp);
            }
            if let Ok(ep) = embeddings_path(&self.store_root, source_name, rel_path) {
                let _ = fs::remove_file(&ep);
            }
            return Ok(0);
        }

        let kind = detect_language(rel_path);
        if matches!(kind, ChunkerKind::Skip) {
            return Ok(0);
        }

        let source_text = fs::read_to_string(&abs_path)?;
        let raw_chunks = chunk_source(&source_text, rel_path, kind);
        if raw_chunks.is_empty() {
            return Ok(0);
        }

        let dimensions = provider.dimensions();
        let max_batch = provider.max_batch_size();

        // Batch embed all chunks
        let texts: Vec<&str> = raw_chunks.iter().map(|c| c.content.as_str()).collect();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for start in (0..texts.len()).step_by(max_batch) {
            let end = (start + max_batch).min(texts.len());
            let batch = &texts[start..end];
            if !batch.is_empty() {
                let vecs = provider
                    .embed(batch)
                    .await
                    .map_err(|e| EngramError::Embed(e.to_string()))?;
                vectors.extend(vecs);
            }
        }

        let now = timestamp();
        let metadata: Vec<ChunkMetadata> = raw_chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| ChunkMetadata {
                chunk_id: format!("{}#{}#{}", source_name, rel_path.display(), chunk.name),
                kind: chunk.kind.clone(),
                name: chunk.name.clone(),
                signature: chunk.signature.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                content_hash: content_hash(&chunk.content),
                tags: vec![],
                indexed_at: now.clone(),
                source_commit: String::new(),
                embedding_offset: i as u64,
            })
            .collect();

        let count = metadata.len();

        // Write to store WITHOUT committing
        let cp = chunks_path(&self.store_root, source_name, rel_path)?;
        write_chunks_jsonl(&cp, &metadata)?;

        let ep = embeddings_path(&self.store_root, source_name, rel_path)?;
        write_embeddings_bin(&ep, &vectors, dimensions, PRECISION_F32)?;

        Ok(count)
    }
}

/// Debouncer that collects file paths and flushes them after a quiet period.
pub struct Debouncer {
    pending: HashMap<PathBuf, Instant>,
    debounce: Duration,
}

impl Debouncer {
    /// Create a new Debouncer with the given debounce duration.
    pub fn new(debounce: Duration) -> Self {
        Self {
            pending: HashMap::new(),
            debounce,
        }
    }

    /// Record a file change event, resetting the debounce timer for this path.
    pub fn record(&mut self, path: PathBuf) {
        self.pending.insert(path, Instant::now());
    }

    /// Drain all paths that have been quiet for at least the debounce duration.
    pub fn drain_ready(&mut self) -> Vec<PathBuf> {
        let now = Instant::now();
        let ready: Vec<PathBuf> = self
            .pending
            .iter()
            .filter(|(_, last)| now.duration_since(**last) >= self.debounce)
            .map(|(path, _)| path.clone())
            .collect();
        for path in &ready {
            self.pending.remove(path);
        }
        ready
    }

    /// Check if there are any pending events.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Number of pending events.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Create a notify watcher that watches the given directories and sends events
/// to an unbounded channel.
///
/// The caller must keep the returned `RecommendedWatcher` alive for watching to continue.
pub fn create_watcher(
    watch_dirs: &[&Path],
) -> Result<(RecommendedWatcher, mpsc::UnboundedReceiver<Event>)> {
    let (tx, rx) = mpsc::unbounded_channel();

    let mut watcher = RecommendedWatcher::new(
        move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                let _ = tx.send(event);
            }
        },
        Config::default(),
    )
    .map_err(|e| {
        EngramError::Io(std::io::Error::other(e.to_string()))
    })?;

    for dir in watch_dirs {
        if dir.exists() {
            watcher.watch(dir, RecursiveMode::Recursive).map_err(|e| {
                EngramError::Io(std::io::Error::other(e.to_string()))
            })?;
        }
    }

    Ok((watcher, rx))
}

fn chunk_source(source: &str, path: &Path, kind: ChunkerKind) -> Vec<RawChunk> {
    match kind {
        ChunkerKind::TreeSitter(lang) => TreeSitterChunker::new().chunk_file(path, source, lang),
        ChunkerKind::Markdown => MarkdownChunker::new().chunk_file(source),
        ChunkerKind::SlidingWindow => SlidingWindowChunker::new().chunk_file(source, 500, 50),
        ChunkerKind::Skip => vec![],
    }
}

fn build_globset(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pat in patterns {
        builder.add(
            Glob::new(pat)
                .map_err(|e| EngramError::Config(format!("invalid glob pattern: {e}")))?,
        );
    }
    builder
        .build()
        .map_err(|e| EngramError::Config(format!("failed to build globset: {e}")))
}

fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use engram_core::{EmbedError, StoreConfig};
    use engram_store::{read_chunks_jsonl, read_embeddings_bin, Store};
    use std::thread;
    use tempfile::TempDir;

    struct MockProvider {
        dims: usize,
    }

    #[async_trait]
    impl EmbeddingProvider for MockProvider {
        fn name(&self) -> &str {
            "mock/test"
        }
        fn dimensions(&self) -> usize {
            self.dims
        }
        fn max_batch_size(&self) -> usize {
            512
        }
        async fn embed(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
            Ok(texts.iter().map(|_| vec![0.1; self.dims]).collect())
        }
    }

    fn init_store() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("store");
        Store::init_local(&store_path, &StoreConfig::default()).unwrap();
        (dir, store_path)
    }

    fn make_source_dir(files: &[(&str, &str)]) -> TempDir {
        let dir = TempDir::new().unwrap();
        for (name, content) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, content).unwrap();
        }
        dir
    }

    fn source_config(name: &str, path: &Path) -> SourceConfig {
        SourceConfig {
            name: name.to_string(),
            path: path.to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        }
    }

    // --- Debouncer tests ---

    #[test]
    fn debouncer_records_events() {
        let mut debouncer = Debouncer::new(Duration::from_millis(100));
        debouncer.record(PathBuf::from("/a/b.rs"));
        debouncer.record(PathBuf::from("/a/c.rs"));
        assert_eq!(debouncer.pending_count(), 2);
        assert!(debouncer.has_pending());
    }

    #[test]
    fn debouncer_deduplicates_paths() {
        let mut debouncer = Debouncer::new(Duration::from_millis(100));
        debouncer.record(PathBuf::from("/a/b.rs"));
        debouncer.record(PathBuf::from("/a/b.rs"));
        assert_eq!(debouncer.pending_count(), 1);
    }

    #[test]
    fn debouncer_does_not_drain_before_timeout() {
        let mut debouncer = Debouncer::new(Duration::from_millis(500));
        debouncer.record(PathBuf::from("/a/b.rs"));
        let ready = debouncer.drain_ready();
        assert!(ready.is_empty());
        assert!(debouncer.has_pending());
    }

    #[test]
    fn debouncer_drains_after_timeout() {
        let mut debouncer = Debouncer::new(Duration::from_millis(50));
        debouncer.record(PathBuf::from("/a/b.rs"));
        thread::sleep(Duration::from_millis(100));
        let ready = debouncer.drain_ready();
        assert_eq!(ready.len(), 1);
        assert!(!debouncer.has_pending());
    }

    #[test]
    fn debouncer_resets_timer_on_re_record() {
        let mut debouncer = Debouncer::new(Duration::from_millis(100));
        debouncer.record(PathBuf::from("/a/b.rs"));
        thread::sleep(Duration::from_millis(60));
        // Re-record resets the timer
        debouncer.record(PathBuf::from("/a/b.rs"));
        thread::sleep(Duration::from_millis(60));
        // 60ms since last record, not yet ready (need 100ms)
        let ready = debouncer.drain_ready();
        assert!(ready.is_empty());
    }

    // --- FileWatcher match_path tests ---

    #[test]
    fn match_path_returns_source_and_relative_path() {
        let src_dir = make_source_dir(&[("lib.rs", "fn hello() {}")]);
        let config = WatcherConfig::default();
        let src = source_config("my-repo", src_dir.path());
        let (_store_dir, store_path) = init_store();

        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();

        let abs = src_dir.path().join("lib.rs");
        let result = watcher.match_path(&abs);
        assert!(result.is_some());
        let (name, rel) = result.unwrap();
        assert_eq!(name, "my-repo");
        assert_eq!(rel, Path::new("lib.rs"));
    }

    #[test]
    fn match_path_returns_none_for_unknown_path() {
        let src_dir = make_source_dir(&[]);
        let config = WatcherConfig::default();
        let src = source_config("my-repo", src_dir.path());
        let (_store_dir, store_path) = init_store();

        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();

        let result = watcher.match_path(Path::new("/some/other/path.rs"));
        assert!(result.is_none());
    }

    #[test]
    fn match_path_filters_by_ignore_patterns() {
        let src_dir = make_source_dir(&[("target/debug/main", "binary")]);
        let config = WatcherConfig {
            ignore: vec!["target/**".to_string()],
            ..WatcherConfig::default()
        };
        let src = source_config("my-repo", src_dir.path());
        let (_store_dir, store_path) = init_store();

        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();

        let abs = src_dir.path().join("target/debug/main");
        assert!(watcher.match_path(&abs).is_none());
    }

    #[test]
    fn match_path_filters_by_source_exclude() {
        let src_dir = make_source_dir(&[("vendor/dep.rs", "fn dep() {}")]);
        let config = WatcherConfig::default();
        let src = SourceConfig {
            name: "my-repo".to_string(),
            path: src_dir.path().to_string_lossy().to_string(),
            include: vec![],
            exclude: vec!["vendor/**".to_string()],
        };
        let (_store_dir, store_path) = init_store();

        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();

        let abs = src_dir.path().join("vendor/dep.rs");
        assert!(watcher.match_path(&abs).is_none());
    }

    #[test]
    fn match_path_filters_by_source_include() {
        let src_dir = make_source_dir(&[
            ("src/lib.rs", "fn lib() {}"),
            ("docs/readme.md", "# Readme"),
        ]);
        let config = WatcherConfig::default();
        let src = SourceConfig {
            name: "my-repo".to_string(),
            path: src_dir.path().to_string_lossy().to_string(),
            include: vec!["**/*.rs".to_string()],
            exclude: vec![],
        };
        let (_store_dir, store_path) = init_store();

        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();

        // .rs file matches include pattern
        let abs_rs = src_dir.path().join("src/lib.rs");
        assert!(watcher.match_path(&abs_rs).is_some());

        // .md file does not match include pattern
        let abs_md = src_dir.path().join("docs/readme.md");
        assert!(watcher.match_path(&abs_md).is_none());
    }

    #[test]
    fn match_path_with_multiple_sources() {
        let src_a = make_source_dir(&[("a.rs", "fn a() {}")]);
        let src_b = make_source_dir(&[("b.rs", "fn b() {}")]);
        let config = WatcherConfig::default();
        let sources = vec![
            source_config("repo-a", src_a.path()),
            source_config("repo-b", src_b.path()),
        ];
        let (_store_dir, store_path) = init_store();

        let watcher = FileWatcher::new(&sources, &config, &store_path).unwrap();

        let (name_a, _) = watcher.match_path(&src_a.path().join("a.rs")).unwrap();
        assert_eq!(name_a, "repo-a");

        let (name_b, _) = watcher.match_path(&src_b.path().join("b.rs")).unwrap();
        assert_eq!(name_b, "repo-b");
    }

    // --- FileWatcher configuration tests ---

    #[test]
    fn default_debounce_is_2000ms() {
        let config = WatcherConfig::default();
        assert_eq!(config.debounce_ms, 2000);
    }

    #[test]
    fn debounce_duration_matches_config() {
        let src_dir = make_source_dir(&[]);
        let config = WatcherConfig {
            debounce_ms: 3000,
            ..WatcherConfig::default()
        };
        let src = source_config("repo", src_dir.path());
        let (_store_dir, store_path) = init_store();

        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();
        assert_eq!(watcher.debounce_duration(), Duration::from_millis(3000));
    }

    #[test]
    fn watch_dirs_returns_all_source_dirs() {
        let src_a = make_source_dir(&[]);
        let src_b = make_source_dir(&[]);
        let config = WatcherConfig::default();
        let sources = vec![
            source_config("repo-a", src_a.path()),
            source_config("repo-b", src_b.path()),
        ];
        let (_store_dir, store_path) = init_store();

        let watcher = FileWatcher::new(&sources, &config, &store_path).unwrap();
        assert_eq!(watcher.watch_dirs().len(), 2);
    }

    // --- FileWatcher reindex tests ---

    #[tokio::test]
    async fn reindex_file_chunks_and_embeds() {
        let src_dir = make_source_dir(&[("lib.rs", "fn hello() {\n    42\n}")]);
        let (_store_dir, store_path) = init_store();
        let config = WatcherConfig::default();
        let src = source_config("repo", src_dir.path());
        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();
        let provider = MockProvider { dims: 4 };

        let count = watcher
            .reindex_file("repo", Path::new("lib.rs"), &provider)
            .await
            .unwrap();

        assert!(count > 0);

        // Verify chunks written
        let cp = chunks_path(&store_path, "repo", Path::new("lib.rs")).unwrap();
        assert!(cp.exists());
        let chunks = read_chunks_jsonl(&cp).unwrap();
        assert_eq!(chunks.len(), count);

        // Verify embeddings written
        let ep = embeddings_path(&store_path, "repo", Path::new("lib.rs")).unwrap();
        assert!(ep.exists());
        let emb = read_embeddings_bin(&ep).unwrap();
        assert_eq!(emb.dimensions, 4);
        assert_eq!(emb.count, count);
    }

    #[tokio::test]
    async fn reindex_file_handles_deletion() {
        let src_dir = make_source_dir(&[("lib.rs", "fn hello() {}")]);
        let (_store_dir, store_path) = init_store();
        let config = WatcherConfig::default();
        let src = source_config("repo", src_dir.path());
        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();
        let provider = MockProvider { dims: 4 };

        // First, index the file
        watcher
            .reindex_file("repo", Path::new("lib.rs"), &provider)
            .await
            .unwrap();
        let cp = chunks_path(&store_path, "repo", Path::new("lib.rs")).unwrap();
        assert!(cp.exists());

        // Delete the source file
        fs::remove_file(src_dir.path().join("lib.rs")).unwrap();

        // Reindex should remove store files
        let count = watcher
            .reindex_file("repo", Path::new("lib.rs"), &provider)
            .await
            .unwrap();
        assert_eq!(count, 0);
        assert!(!cp.exists());
    }

    #[tokio::test]
    async fn reindex_file_skips_unknown_extensions() {
        let src_dir = make_source_dir(&[("image.png", "not a real image")]);
        let (_store_dir, store_path) = init_store();
        let config = WatcherConfig::default();
        let src = source_config("repo", src_dir.path());
        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();
        let provider = MockProvider { dims: 4 };

        let count = watcher
            .reindex_file("repo", Path::new("image.png"), &provider)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn reindex_file_does_not_commit() {
        let src_dir = make_source_dir(&[("lib.rs", "fn hello() {\n    42\n}")]);
        let (_store_dir, store_path) = init_store();
        let config = WatcherConfig::default();
        let src = source_config("repo", src_dir.path());
        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();
        let provider = MockProvider { dims: 4 };

        // Get initial commit count
        let repo = git2::Repository::open(&store_path).unwrap();
        let initial_head = repo.head().unwrap().peel_to_commit().unwrap().id();

        watcher
            .reindex_file("repo", Path::new("lib.rs"), &provider)
            .await
            .unwrap();

        // HEAD should not have changed — no commit was made
        let repo = git2::Repository::open(&store_path).unwrap();
        let current_head = repo.head().unwrap().peel_to_commit().unwrap().id();
        assert_eq!(initial_head, current_head, "reindex_file should not commit");
    }

    #[tokio::test]
    async fn reindex_file_errors_on_unknown_source() {
        let src_dir = make_source_dir(&[]);
        let (_store_dir, store_path) = init_store();
        let config = WatcherConfig::default();
        let src = source_config("repo", src_dir.path());
        let watcher = FileWatcher::new(&[src], &config, &store_path).unwrap();
        let provider = MockProvider { dims: 4 };

        let result = watcher
            .reindex_file("nonexistent", Path::new("lib.rs"), &provider)
            .await;
        assert!(result.is_err());
    }
}
