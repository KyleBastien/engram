use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use engram_core::{
    ChunkMetadata, EmbeddingProvider, EngramError, Manifest, Result, SourceConfig,
};
use engram_store::{
    chunks_path, commit_changes, embeddings_path, read_chunks_jsonl, read_embeddings_bin,
    read_manifest, write_chunks_jsonl, write_embeddings_bin, write_manifest,
};
use git2::Repository;

use crate::{
    content_hash, detect_changed_files, detect_language, has_chunk_changed, ChunkerKind,
    MarkdownChunker, RawChunk, SlidingWindowChunker, TreeSitterChunker,
};

/// Report summarizing what the ingest pipeline did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IngestReport {
    pub files_processed: usize,
    pub chunks_created: usize,
    pub chunks_skipped: usize,
    pub chunks_deleted: usize,
    pub embed_calls: usize,
}

/// The full ingest pipeline orchestrator.
///
/// Indexes a source repository into the git-backed semantic store:
/// detect changes, chunk files, embed new chunks, write to store, commit.
pub struct IngestPipeline;

impl IngestPipeline {
    /// Run the full ingest pipeline for a source repo into a store.
    pub async fn run(
        source: &SourceConfig,
        store_root: &Path,
        provider: &dyn EmbeddingProvider,
        full: bool,
    ) -> Result<IngestReport> {
        let mut report = IngestReport::default();
        let source_path = Path::new(&source.path);
        let dimensions = provider.dimensions();
        let max_batch = provider.max_batch_size();

        // Read existing manifest for incremental mode
        let existing_manifest = read_manifest(store_root)?;
        let since_commit = if full {
            None
        } else {
            existing_manifest
                .as_ref()
                .and_then(|m| m.last_indexed_commit.as_deref())
        };

        // Detect changed files in source repo
        let changed = detect_changed_files(source_path, since_commit, source)?;

        let source_commit = head_oid(source_path)?;
        let now = timestamp();

        // Process added + modified files
        for rel_path in changed.added.iter().chain(changed.modified.iter()) {
            let kind = detect_language(rel_path);
            if matches!(kind, ChunkerKind::Skip) {
                continue;
            }

            let abs_path = source_path.join(rel_path);
            let source_text = match fs::read_to_string(&abs_path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let raw_chunks = chunk_source(&source_text, rel_path, kind);
            if raw_chunks.is_empty() {
                continue;
            }

            report.files_processed += 1;

            // Load existing data for change detection
            let old_chunks = load_old_chunks(store_root, &source.name, rel_path);
            let old_vectors = load_old_vectors(store_root, &source.name, rel_path);

            let old_map: HashMap<&str, (&str, usize)> = old_chunks
                .iter()
                .map(|c| {
                    (
                        c.name.as_str(),
                        (c.content_hash.as_str(), c.embedding_offset as usize),
                    )
                })
                .collect();

            // Pre-compute content hashes
            let hashes: Vec<String> =
                raw_chunks.iter().map(|c| content_hash(&c.content)).collect();

            // Classify chunks: reuse old embedding or need new embedding
            let mut reuse: HashMap<usize, usize> = HashMap::new();
            let mut to_embed: Vec<usize> = Vec::new();

            for (i, chunk) in raw_chunks.iter().enumerate() {
                if let Some(&(old_hash, old_offset)) = old_map.get(chunk.name.as_str()) {
                    if !has_chunk_changed(old_hash, &hashes[i]) {
                        reuse.insert(i, old_offset);
                        report.chunks_skipped += 1;
                        continue;
                    }
                }
                to_embed.push(i);
                report.chunks_created += 1;
            }

            // Batch embed new/modified chunks
            let texts: Vec<&str> = to_embed
                .iter()
                .map(|&i| raw_chunks[i].content.as_str())
                .collect();
            let mut new_vectors: Vec<Vec<f32>> = Vec::new();
            for start in (0..texts.len()).step_by(max_batch) {
                let end = (start + max_batch).min(texts.len());
                let batch = &texts[start..end];
                if !batch.is_empty() {
                    let vecs = provider
                        .embed(batch)
                        .await
                        .map_err(|e| EngramError::Embed(e.to_string()))?;
                    new_vectors.extend(vecs);
                    report.embed_calls += 1;
                }
            }

            // Assemble final vectors and metadata
            let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(raw_chunks.len());
            let mut metadata: Vec<ChunkMetadata> = Vec::with_capacity(raw_chunks.len());
            let mut embed_idx = 0;

            for (i, chunk) in raw_chunks.iter().enumerate() {
                let vec = if let Some(&old_offset) = reuse.get(&i) {
                    extract_vector(&old_vectors, old_offset, dimensions)
                } else {
                    let v = new_vectors[embed_idx].clone();
                    embed_idx += 1;
                    v
                };
                vectors.push(vec);

                metadata.push(ChunkMetadata {
                    chunk_id: format!("{}#{}#{}", source.name, rel_path.display(), chunk.name),
                    kind: chunk.kind.clone(),
                    name: chunk.name.clone(),
                    signature: chunk.signature.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    content_hash: hashes[i].clone(),
                    tags: vec![],
                    indexed_at: now.clone(),
                    source_commit: source_commit.clone(),
                    embedding_offset: i as u64,
                });
            }

            // Write to store
            let cp = chunks_path(store_root, &source.name, rel_path)?;
            write_chunks_jsonl(&cp, &metadata)?;

            let ep = embeddings_path(store_root, &source.name, rel_path)?;
            write_embeddings_bin(&ep, &vectors, dimensions)?;
        }

        // Handle deleted files
        for rel_path in &changed.deleted {
            if let Ok(cp) = chunks_path(store_root, &source.name, rel_path) {
                if cp.exists() {
                    if let Ok(old) = read_chunks_jsonl(&cp) {
                        report.chunks_deleted += old.len();
                    }
                    let _ = fs::remove_file(&cp);
                }
            }
            if let Ok(ep) = embeddings_path(store_root, &source.name, rel_path) {
                if ep.exists() {
                    let _ = fs::remove_file(&ep);
                }
            }
        }

        // Update manifest
        let manifest = Manifest {
            chunk_count: report.chunks_created + report.chunks_skipped,
            last_indexed_commit: Some(source_commit),
            model_name: provider.name().to_string(),
            dimensions,
            source_repos: vec![source.path.clone()],
            created_at: existing_manifest
                .as_ref()
                .map(|m| m.created_at.clone())
                .unwrap_or_else(|| now.clone()),
            updated_at: now,
        };
        write_manifest(store_root, &manifest)?;

        // Invalidate cache fingerprint
        let fp = store_root.join(".engram-cache").join("fingerprint");
        if fp.exists() {
            let _ = fs::remove_file(&fp);
        }

        // Commit changes to store
        let repo = Repository::open(store_root)
            .map_err(|e| EngramError::Git(format!("failed to open store repo: {e}")))?;
        let msg = format!(
            "engram: reindex {} files, {} chunks updated",
            report.files_processed, report.chunks_created
        );
        commit_changes(&repo, &msg)?;

        Ok(report)
    }
}

fn chunk_source(source: &str, path: &Path, kind: ChunkerKind) -> Vec<RawChunk> {
    match kind {
        ChunkerKind::TreeSitter(lang) => TreeSitterChunker::new().chunk_file(path, source, lang),
        ChunkerKind::Markdown => MarkdownChunker::new().chunk_file(source),
        ChunkerKind::SlidingWindow => SlidingWindowChunker::new().chunk_file(source, 500, 50),
        ChunkerKind::Skip => vec![],
    }
}

fn head_oid(repo_path: &Path) -> Result<String> {
    let repo = Repository::open(repo_path)
        .map_err(|e| EngramError::Git(format!("failed to open source repo: {e}")))?;
    let head = repo
        .head()
        .map_err(|e| EngramError::Git(format!("failed to get HEAD: {e}")))?;
    let oid = head
        .target()
        .ok_or_else(|| EngramError::Git("HEAD has no target".into()))?;
    Ok(oid.to_string())
}

fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".into())
}

fn load_old_chunks(store_root: &Path, source_name: &str, rel_path: &Path) -> Vec<ChunkMetadata> {
    chunks_path(store_root, source_name, rel_path)
        .ok()
        .and_then(|p| {
            if p.exists() {
                read_chunks_jsonl(&p).ok()
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn load_old_vectors(store_root: &Path, source_name: &str, rel_path: &Path) -> Vec<f32> {
    embeddings_path(store_root, source_name, rel_path)
        .ok()
        .and_then(|p| {
            if p.exists() {
                read_embeddings_bin(&p).ok().map(|ef| ef.vectors)
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn extract_vector(flat: &[f32], offset: usize, dims: usize) -> Vec<f32> {
    let start = offset * dims;
    let end = start + dims;
    if end <= flat.len() {
        flat[start..end].to_vec()
    } else {
        vec![0.0; dims]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use engram_core::{EmbedError, StoreConfig};
    use engram_store::Store;
    use std::path::PathBuf;
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

    /// Create a git repo with an initial commit containing the given files.
    fn setup_source(files: &[(&str, &str)]) -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let repo = Repository::init(dir.path()).unwrap();

        for (name, content) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, content).unwrap();
        }

        {
            let mut index = repo.index().unwrap();
            index
                .add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
                .unwrap();
            index.write().unwrap();
            let tree_oid = index.write_tree().unwrap();
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = git2::Signature::now("test", "test@test.com").unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "initial", &tree, &[])
                .unwrap();
        }

        let path = dir.path().to_path_buf();
        (dir, path)
    }

    /// Initialize a fresh store and return its path.
    fn init_store() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("store");
        Store::init_local(&store_path, &StoreConfig::default()).unwrap();
        (dir, store_path)
    }

    /// Make a commit in the given repo with all current changes staged.
    fn commit_source(repo_path: &Path, message: &str) {
        let repo = Repository::open(repo_path).unwrap();
        let mut index = repo.index().unwrap();
        index
            .add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
            .unwrap();
        index.write().unwrap();
        let tree_oid = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_oid).unwrap();
        let sig = git2::Signature::now("test", "test@test.com").unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &[&head])
            .unwrap();
    }

    fn source_config(name: &str, path: &Path) -> SourceConfig {
        SourceConfig {
            name: name.to_string(),
            path: path.to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        }
    }

    #[tokio::test]
    async fn full_ingest_indexes_all_files() {
        let (_src, src_path) = setup_source(&[
            ("src/main.rs", "fn main() {\n    println!(\"hello\");\n}"),
            ("README.md", "# Hello\nWorld"),
        ]);
        let (_store, store_path) = init_store();
        let provider = MockProvider { dims: 4 };
        let cfg = source_config("test-repo", &src_path);

        let report = IngestPipeline::run(&cfg, &store_path, &provider, true)
            .await
            .unwrap();

        assert_eq!(report.files_processed, 2);
        assert!(report.chunks_created > 0);
        assert_eq!(report.chunks_skipped, 0);
        assert_eq!(report.chunks_deleted, 0);
        assert!(report.embed_calls > 0);

        // Verify manifest
        let manifest = read_manifest(&store_path).unwrap().expect("manifest");
        assert_eq!(manifest.chunk_count, report.chunks_created);
        assert!(manifest.last_indexed_commit.is_some());
        assert_eq!(manifest.model_name, "mock/test");
        assert_eq!(manifest.dimensions, 4);
    }

    #[tokio::test]
    async fn full_ingest_commits_to_store() {
        let (_src, src_path) = setup_source(&[("lib.rs", "fn add(a: i32, b: i32) -> i32 { a + b }")]);
        let (_store, store_path) = init_store();
        let provider = MockProvider { dims: 4 };
        let cfg = source_config("repo", &src_path);

        IngestPipeline::run(&cfg, &store_path, &provider, true)
            .await
            .unwrap();

        let repo = Repository::open(&store_path).unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        let msg = head.message().unwrap();
        assert!(
            msg.starts_with("engram: reindex"),
            "commit message was: {msg}"
        );
    }

    #[tokio::test]
    async fn incremental_skips_unchanged_chunks() {
        // Source file with two functions
        let (_src, src_path) = setup_source(&[(
            "lib.rs",
            "fn hello() {\n    1\n}\n\nfn world() {\n    2\n}",
        )]);
        let (_store, store_path) = init_store();
        let provider = MockProvider { dims: 4 };
        let cfg = source_config("repo", &src_path);

        // Full ingest
        let r1 = IngestPipeline::run(&cfg, &store_path, &provider, true)
            .await
            .unwrap();
        assert_eq!(r1.chunks_created, 2); // hello + world
        assert_eq!(r1.chunks_skipped, 0);

        // Modify file: keep hello, change world to world_v2
        fs::write(
            src_path.join("lib.rs"),
            "fn hello() {\n    1\n}\n\nfn world_v2() {\n    3\n}",
        )
        .unwrap();
        commit_source(&src_path, "modify world");

        // Incremental ingest
        let r2 = IngestPipeline::run(&cfg, &store_path, &provider, false)
            .await
            .unwrap();
        assert_eq!(r2.files_processed, 1);
        assert_eq!(r2.chunks_skipped, 1); // hello unchanged
        assert_eq!(r2.chunks_created, 1); // world_v2 is new
    }

    #[tokio::test]
    async fn handles_deleted_files() {
        let (_src, src_path) = setup_source(&[
            ("a.rs", "fn a() {}"),
            ("b.rs", "fn b() {}"),
        ]);
        let (_store, store_path) = init_store();
        let provider = MockProvider { dims: 4 };
        let cfg = source_config("repo", &src_path);

        // Full ingest both files
        let r1 = IngestPipeline::run(&cfg, &store_path, &provider, true)
            .await
            .unwrap();
        assert_eq!(r1.files_processed, 2);

        // Delete b.rs and commit
        fs::remove_file(src_path.join("b.rs")).unwrap();
        {
            let repo = Repository::open(&src_path).unwrap();
            let mut index = repo.index().unwrap();
            index
                .remove_path(Path::new("b.rs"))
                .unwrap();
            index.write().unwrap();
            let tree_oid = index.write_tree().unwrap();
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = git2::Signature::now("test", "test@test.com").unwrap();
            let head = repo.head().unwrap().peel_to_commit().unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "delete b", &tree, &[&head])
                .unwrap();
        }

        // Incremental ingest
        let r2 = IngestPipeline::run(&cfg, &store_path, &provider, false)
            .await
            .unwrap();
        assert!(r2.chunks_deleted > 0);
    }

    #[tokio::test]
    async fn skips_binary_files() {
        let (_src, src_path) = setup_source(&[
            ("lib.rs", "fn hello() {}"),
            ("image.png", "not really an image"),
        ]);
        let (_store, store_path) = init_store();
        let provider = MockProvider { dims: 4 };
        let cfg = source_config("repo", &src_path);

        let report = IngestPipeline::run(&cfg, &store_path, &provider, true)
            .await
            .unwrap();

        // Only the .rs file should be processed, .png is skipped
        assert_eq!(report.files_processed, 1);
    }

    #[tokio::test]
    async fn manifest_source_repos_populated() {
        let (_src, src_path) = setup_source(&[("main.rs", "fn main() {}")]);
        let (_store, store_path) = init_store();
        let provider = MockProvider { dims: 4 };
        let cfg = source_config("my-project", &src_path);

        IngestPipeline::run(&cfg, &store_path, &provider, true)
            .await
            .unwrap();

        let manifest = read_manifest(&store_path).unwrap().unwrap();
        assert_eq!(manifest.source_repos, vec![src_path.to_string_lossy().to_string()]);
    }

    #[tokio::test]
    async fn chunks_and_embeddings_files_written() {
        let (_src, src_path) = setup_source(&[("lib.rs", "fn hello() { 42 }")]);
        let (_store, store_path) = init_store();
        let provider = MockProvider { dims: 4 };
        let cfg = source_config("repo", &src_path);

        IngestPipeline::run(&cfg, &store_path, &provider, true)
            .await
            .unwrap();

        // Verify chunks.jsonl exists
        let cp = chunks_path(&store_path, "repo", Path::new("lib.rs")).unwrap();
        assert!(cp.exists(), "chunks file should exist");

        // Verify embeddings.bin exists
        let ep = embeddings_path(&store_path, "repo", Path::new("lib.rs")).unwrap();
        assert!(ep.exists(), "embeddings file should exist");

        // Verify contents
        let chunks = read_chunks_jsonl(&cp).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].name, "hello");

        let emb = read_embeddings_bin(&ep).unwrap();
        assert_eq!(emb.dimensions, 4);
        assert_eq!(emb.count, chunks.len());
    }
}
