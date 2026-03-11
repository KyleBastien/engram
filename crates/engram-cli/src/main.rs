use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use engram_core::{EmbeddingProvider, OnboardingDepth, SourceConfig, StoreConfig};
use engram_ingest::{IngestPipeline, IngestReport};
use engram_mcp::{serve_sse, Context, McpServer};
use engram_query::IndexManager;
use engram_store::{compact_snapshots, read_manifest, sync_pull, sync_push, Store};

mod provider;

#[derive(Parser)]
#[command(name = "engram", about = "Git-backed semantic context for AI coding agents")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, ValueEnum)]
enum Transport {
    Stdio,
    Sse,
}

#[derive(Clone, ValueEnum)]
enum DepthArg {
    Quick,
    Standard,
    Deep,
}

#[derive(Clone, ValueEnum)]
enum SyncMode {
    Pull,
    Push,
    Both,
}

impl From<DepthArg> for OnboardingDepth {
    fn from(d: DepthArg) -> Self {
        match d {
            DepthArg::Quick => OnboardingDepth::Quick,
            DepthArg::Standard => OnboardingDepth::Standard,
            DepthArg::Deep => OnboardingDepth::Deep,
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP server
    Serve {
        /// Transport protocol (stdio or sse)
        #[arg(long, default_value = "stdio")]
        transport: Transport,

        /// Port for SSE transport (only used with --transport sse)
        #[arg(long, default_value = "3000")]
        port: u16,

        /// Server context name
        #[arg(long, default_value = "default")]
        context: String,

        /// Path to the store directory (defaults to ./engram-store)
        #[arg(long, default_value = "engram-store")]
        path: PathBuf,
    },

    /// Initialize a new semantic store
    Init {
        /// Create a local store
        #[arg(long, conflicts_with = "remote")]
        local: bool,

        /// Initialize from a remote git URL
        #[arg(long, conflicts_with = "local")]
        remote: Option<String>,

        /// Path for the store directory (defaults to ./engram-store)
        #[arg(long, default_value = "engram-store")]
        path: PathBuf,
    },

    /// Sync the store with its remote
    Sync {
        /// Sync mode: pull, push, or both (default: both)
        #[arg(long, default_value = "both")]
        mode: SyncMode,

        /// Path to the store directory (defaults to ./engram-store)
        #[arg(long, default_value = "engram-store")]
        path: PathBuf,
    },

    /// Check store health and index status
    Status {
        /// Path to the store directory (defaults to ./engram-store)
        #[arg(long, default_value = "engram-store")]
        path: PathBuf,
    },

    /// Run onboarding analysis on a source repository
    Onboard {
        /// Analysis depth (quick, standard, or deep)
        #[arg(long, default_value = "standard")]
        depth: DepthArg,

        /// Only onboard this source repo (defaults to the first configured source)
        #[arg(long)]
        repo: Option<String>,

        /// Path to the store directory (defaults to ./engram-store)
        #[arg(long, default_value = "engram-store")]
        path: PathBuf,
    },

    /// Compact snapshots by promoting through storage tiers
    Compact {
        /// Path to the store directory (defaults to ./engram-store)
        #[arg(long, default_value = "engram-store")]
        path: PathBuf,
    },

    /// Reindex source repositories
    Reindex {
        /// Force full re-chunk and re-embed
        #[arg(long, conflicts_with = "incremental")]
        full: bool,

        /// Only process changed files (default)
        #[arg(long, conflicts_with = "full")]
        incremental: bool,

        /// Only process files matching this glob pattern
        #[arg(long)]
        paths: Option<String>,

        /// Only reindex this source repo
        #[arg(long)]
        repo: Option<String>,

        /// Path to the store directory (defaults to ./engram-store)
        #[arg(long, default_value = "engram-store")]
        path: PathBuf,
    },
}

fn load_config(store_path: &Path) -> Result<StoreConfig, String> {
    let config_path = store_path.join("engram.config.yaml");
    let yaml = fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read config at {}: {e}", config_path.display()))?;
    serde_yaml::from_str(&yaml).map_err(|e| format!("failed to parse config: {e}"))
}

async fn run_reindex(
    store_path: &Path,
    provider: &dyn EmbeddingProvider,
    mut sources: Vec<SourceConfig>,
    full: bool,
    paths_filter: Option<&str>,
) -> engram_core::Result<IngestReport> {
    if let Some(glob) = paths_filter {
        for source in &mut sources {
            source.include = vec![glob.to_string()];
        }
    }
    for source in &sources {
        println!("Reindexing source '{}'...", source.name);
    }
    let report = IngestPipeline::run(&sources, store_path, provider, full).await?;
    println!(
        "  {} files processed, {} chunks created, {} skipped, {} deleted",
        report.files_processed,
        report.chunks_created,
        report.chunks_skipped,
        report.chunks_deleted,
    );
    Ok(report)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            transport,
            port,
            context,
            path,
        } => {
            if !path.join(".engram").exists() {
                eprintln!("Error: no engram store found at {}", path.display());
                eprintln!("Hint: run `engram init --local` first");
                process::exit(1);
            }

            let config = match load_config(&path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            };

            eprintln!("engram: booting index from {}...", path.display());
            let boot_start = Instant::now();

            let index_manager = match IndexManager::boot(&path, &config).await {
                Ok(mgr) => mgr,
                Err(e) => {
                    eprintln!("Error: failed to boot index: {e}");
                    process::exit(1);
                }
            };

            let boot_ms = boot_start.elapsed().as_millis();
            let chunk_count = index_manager.chunk_count();
            let cache_status = index_manager.cache_status.clone();

            eprintln!(
                "engram: index loaded — {} chunks, {}ms (cache: {})",
                chunk_count, boot_ms, cache_status
            );

            let provider = provider::OllamaProvider::from_config(&config.embedding);

            // Build source_roots from config sources
            let source_roots: HashMap<String, PathBuf> = config
                .sources
                .iter()
                .map(|s| (s.name.clone(), PathBuf::from(&s.path)))
                .collect();

            let graph = index_manager.graph().clone();
            let search = index_manager.into_hybrid_search();
            let mut server =
                McpServer::with_engine_and_sources(search, Box::new(provider), source_roots);
            server.set_boot_info(boot_ms as u64, path.to_string_lossy().to_string(), cache_status);
            server.set_graph(graph);
            server.set_context(Context::from_name(&context));

            match transport {
                Transport::Stdio => {
                    eprintln!("engram: serving on stdio (context: {context})");
                    let stdin = tokio::io::stdin();
                    let stdout = tokio::io::stdout();
                    if let Err(e) = server.run(stdin, stdout).await {
                        eprintln!("Error: MCP server error: {e}");
                        process::exit(1);
                    }
                }
                Transport::Sse => {
                    eprintln!("engram: serving on SSE port {port} (context: {context})");
                    if let Err(e) = serve_sse(server, port).await {
                        eprintln!("Error: SSE server error: {e}");
                        process::exit(1);
                    }
                }
            }
        }
        Commands::Status { path } => {
            if !path.join(".engram").exists() {
                eprintln!("Error: no engram store found at {}", path.display());
                eprintln!("Hint: run `engram init --local` first");
                process::exit(1);
            }

            let config = match load_config(&path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            };

            let manifest = match read_manifest(&path) {
                Ok(Some(m)) => Some(m),
                Ok(None) => None,
                Err(e) => {
                    eprintln!("Warning: could not read manifest: {e}");
                    None
                }
            };

            // Chunk count
            let chunk_count = manifest.as_ref().map_or(0, |m| m.chunk_count);
            println!("Engram Store Status");
            println!("===================");
            println!("Store path:         {}", path.display());
            println!("Chunks indexed:     {chunk_count}");

            // Source repos
            println!("\nSource Repositories:");
            if config.sources.is_empty() {
                println!("  (none configured)");
            } else {
                for source in &config.sources {
                    println!("  {} ({})", source.name, source.path);
                }
            }

            // Last indexed commits (per source repo)
            println!("\nLast Indexed Commits:");
            match manifest.as_ref() {
                Some(m) if !m.last_indexed_commits.is_empty() => {
                    for (repo, commit) in &m.last_indexed_commits {
                        println!("  {repo}: {commit}");
                    }
                }
                _ => println!("  (not yet indexed)"),
            }

            // Staleness summary
            println!("\nStaleness:");
            if config.sources.is_empty() {
                println!("  (no sources to check)");
            } else {
                for source in &config.sources {
                    let indexed_commit = manifest
                        .as_ref()
                        .and_then(|m| m.last_indexed_commits.get(&source.name))
                        .map(|s| s.as_str());
                    let source_path = Path::new(&source.path);
                    let status = if indexed_commit.is_none() {
                        "not indexed".to_string()
                    } else if !source_path.exists() {
                        "source path not found".to_string()
                    } else {
                        match git2::Repository::open(source_path) {
                            Ok(repo) => match repo.head().and_then(|r| r.peel_to_commit()) {
                                Ok(head) => {
                                    let head_hex = head.id().to_string();
                                    if indexed_commit == Some(head_hex.as_str()) {
                                        "up to date".to_string()
                                    } else {
                                        format!("stale (HEAD: {})", &head_hex[..8.min(head_hex.len())])
                                    }
                                }
                                Err(_) => "could not read HEAD".to_string(),
                            },
                            Err(_) => "not a git repository".to_string(),
                        }
                    };
                    println!("  {}: {status}", source.name);
                }
            }

            // Cache status
            let cache_exists = path.join(".engram-cache").join("fingerprint").exists();
            println!(
                "\nCache:              {}",
                if cache_exists { "warm" } else { "cold" }
            );

            // Embedding provider
            println!(
                "Embedding provider: {}/{}",
                config.embedding.provider, config.embedding.model
            );
        }
        Commands::Init {
            local,
            remote,
            path,
        } => {
            if !local && remote.is_none() {
                eprintln!("Error: --local or --remote <url> flag is required for init");
                process::exit(1);
            }

            if path.join(".engram").exists() {
                eprintln!("Error: store already exists at {}", path.display());
                process::exit(1);
            }

            let config = StoreConfig::default();
            if let Some(url) = remote {
                match Store::init_remote(&url, &path, &config) {
                    Ok(_) => {
                        println!(
                            "Initialized engram store at {} (remote: {url})",
                            path.display()
                        );
                    }
                    Err(e) => {
                        eprintln!("Error: failed to initialize remote store: {e}");
                        process::exit(1);
                    }
                }
            } else {
                match Store::init_local(&path, &config) {
                    Ok(_) => {
                        println!("Initialized engram store at {}", path.display());
                    }
                    Err(e) => {
                        eprintln!("Error: failed to initialize store: {e}");
                        process::exit(1);
                    }
                }
            }
        }
        Commands::Onboard { depth, repo, path } => {
            if !path.join(".engram").exists() {
                eprintln!("Error: no engram store found at {}", path.display());
                eprintln!("Hint: run `engram init --local` first");
                process::exit(1);
            }

            let config = match load_config(&path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            };

            let source = if let Some(ref repo_name) = repo {
                match config.sources.iter().find(|s| s.name == *repo_name) {
                    Some(s) => s.clone(),
                    None => {
                        eprintln!("Error: source repo '{repo_name}' not found in config");
                        process::exit(1);
                    }
                }
            } else if let Some(s) = config.sources.first() {
                s.clone()
            } else {
                eprintln!("Error: no source repos configured in engram.config.yaml");
                process::exit(1);
            };

            let onboarding_depth: OnboardingDepth = depth.into();
            let repo_path = PathBuf::from(&source.path);

            println!("Onboarding '{}' (depth: {onboarding_depth})...", source.name);
            println!("  [1/4] Detecting project metadata...");

            let start = Instant::now();
            match engram_ingest::run_onboarding(&repo_path, &path, &source, onboarding_depth).await
            {
                Ok(report) => {
                    let elapsed = start.elapsed();
                    if report.metadata_detected {
                        println!("  [2/4] Extracting build/test commands...");
                    }
                    if report.architecture_analyzed {
                        println!("  [3/4] Analyzing directory structure...");
                    }
                    if report.abstractions_extracted {
                        println!("  [4/4] Extracting key abstractions...");
                    }

                    println!("\nOnboarding complete ({:.1}s):", elapsed.as_secs_f64());
                    println!("  Depth:          {}", report.depth);
                    println!("  Metadata:       {}", if report.metadata_detected { "detected" } else { "skipped" });
                    println!("  Commands:       {}", if report.commands_extracted { "extracted" } else { "skipped" });
                    println!("  Architecture:   {}", if report.architecture_analyzed { "analyzed" } else { "skipped" });
                    println!("  Abstractions:   {}", if report.abstractions_extracted { "extracted" } else { "skipped" });
                    println!("  Files written:  {}", report.files_written.len());
                    for f in &report.files_written {
                        println!("    - {f}");
                    }
                    if let Some(ref hash) = report.commit_hash {
                        println!("  Commit:         {}", &hash[..8.min(hash.len())]);
                    }
                }
                Err(e) => {
                    eprintln!("Error: onboarding failed: {e}");
                    process::exit(1);
                }
            }
        }
        Commands::Compact { path } => {
            if !path.join(".engram").exists() {
                eprintln!("Error: no engram store found at {}", path.display());
                eprintln!("Hint: run `engram init --local` first");
                process::exit(1);
            }

            match compact_snapshots(&path) {
                Ok(report) => {
                    println!("Compaction complete:");
                    println!("  Active → Compressed: {}", report.active_to_compressed);
                    println!("  Compressed → Archived: {}", report.compressed_to_archived);
                }
                Err(e) => {
                    eprintln!("Error: compaction failed: {e}");
                    process::exit(1);
                }
            }
        }
        Commands::Reindex {
            full,
            incremental: _,
            paths,
            repo,
            path,
        } => {
            if !path.join(".engram").exists() {
                eprintln!("Error: no engram store found at {}", path.display());
                eprintln!("Hint: run `engram init --local` first");
                process::exit(1);
            }

            let config = match load_config(&path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            };

            let sources: Vec<SourceConfig> = if let Some(ref repo_name) = repo {
                config
                    .sources
                    .iter()
                    .filter(|s| s.name == *repo_name)
                    .cloned()
                    .collect()
            } else {
                config.sources.clone()
            };

            if sources.is_empty() {
                if let Some(ref repo_name) = repo {
                    eprintln!("Error: source repo '{repo_name}' not found in config");
                } else {
                    eprintln!("Error: no source repos configured in engram.config.yaml");
                }
                process::exit(1);
            }

            let provider = provider::OllamaProvider::from_config(&config.embedding);

            match run_reindex(&path, &provider, sources, full, paths.as_deref()).await {
                Ok(report) => {
                    println!("\nReindex complete:");
                    println!("  Files processed: {}", report.files_processed);
                    println!("  Chunks created:  {}", report.chunks_created);
                    println!("  Chunks skipped:  {}", report.chunks_skipped);
                    println!("  Chunks deleted:  {}", report.chunks_deleted);
                    println!("  Embed calls:     {}", report.embed_calls);
                }
                Err(e) => {
                    eprintln!("Error: reindex failed: {e}");
                    process::exit(1);
                }
            }
        }
        Commands::Sync { mode, path } => {
            if !path.join(".engram").exists() {
                eprintln!("Error: no engram store found at {}", path.display());
                eprintln!("Hint: run `engram init --local` first");
                process::exit(1);
            }

            let config = match load_config(&path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            };

            if config.store.remote.is_none() {
                eprintln!("Error: no remote configured for this store");
                eprintln!("Hint: run `engram init --remote <url>` to set up a remote store");
                process::exit(1);
            }

            fn print_report(report: &engram_store::SyncReport) {
                let dir = match report.direction {
                    engram_store::SyncDirection::Pull => "Pull",
                    engram_store::SyncDirection::Push => "Push",
                };
                println!("  {dir}: {} commits transferred", report.commits_transferred);
                if !report.conflicts.is_empty() {
                    println!("  Conflicts resolved: {}", report.conflicts.len());
                    for c in &report.conflicts {
                        println!("    - {c}");
                    }
                }
            }

            println!("Syncing store at {}...", path.display());

            match mode {
                SyncMode::Pull => match sync_pull(&path) {
                    Ok(report) => {
                        print_report(&report);
                    }
                    Err(e) => {
                        eprintln!("Error: sync pull failed: {e}");
                        process::exit(1);
                    }
                },
                SyncMode::Push => match sync_push(&path) {
                    Ok(report) => {
                        print_report(&report);
                    }
                    Err(e) => {
                        eprintln!("Error: sync push failed: {e}");
                        process::exit(1);
                    }
                },
                SyncMode::Both => {
                    match sync_pull(&path) {
                        Ok(report) => print_report(&report),
                        Err(e) => {
                            eprintln!("Error: sync pull failed: {e}");
                            process::exit(1);
                        }
                    }
                    match sync_push(&path) {
                        Ok(report) => print_report(&report),
                        Err(e) => {
                            eprintln!("Error: sync push failed: {e}");
                            process::exit(1);
                        }
                    }
                }
            }

            println!("Sync complete.");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use async_trait::async_trait;
    use engram_core::{EmbedError, EmbeddingProvider, SourceConfig, StoreConfig};
    use engram_store::Store;
    use tempfile::TempDir;

    use engram_core::Manifest;
    use engram_store::{read_manifest, write_manifest};

    use super::{load_config, run_reindex, Cli, Commands, DepthArg, SyncMode, Transport};

    // --- Mock embedding provider for tests ---

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

    // --- Helpers ---

    fn init_store_with_sources(sources: Vec<SourceConfig>) -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("store");
        let mut config = StoreConfig::default();
        config.sources = sources;
        Store::init_local(&store_path, &config).unwrap();
        (dir, store_path)
    }

    fn setup_source_repo(files: &[(&str, &str)]) -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let repo = git2::Repository::init(dir.path()).unwrap();

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

    // --- CLI arg parsing tests ---

    #[test]
    fn test_reindex_defaults_to_incremental() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "reindex"]);
        match cli.command {
            Commands::Reindex {
                full, incremental, ..
            } => {
                assert!(!full);
                assert!(!incremental);
            }
            _ => panic!("expected Reindex command"),
        }
    }

    #[test]
    fn test_reindex_full_flag() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "reindex", "--full"]);
        match cli.command {
            Commands::Reindex { full, .. } => {
                assert!(full);
            }
            _ => panic!("expected Reindex command"),
        }
    }

    #[test]
    fn test_reindex_incremental_flag() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "reindex", "--incremental"]);
        match cli.command {
            Commands::Reindex {
                full, incremental, ..
            } => {
                assert!(!full);
                assert!(incremental);
            }
            _ => panic!("expected Reindex command"),
        }
    }

    #[test]
    fn test_reindex_full_and_incremental_conflict() {
        use clap::Parser;
        let result = Cli::try_parse_from(["engram", "reindex", "--full", "--incremental"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reindex_paths_flag() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "reindex", "--paths", "**/*.rs"]);
        match cli.command {
            Commands::Reindex { paths, .. } => {
                assert_eq!(paths, Some("**/*.rs".to_string()));
            }
            _ => panic!("expected Reindex command"),
        }
    }

    #[test]
    fn test_reindex_repo_flag() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "reindex", "--repo", "my-project"]);
        match cli.command {
            Commands::Reindex { repo, .. } => {
                assert_eq!(repo, Some("my-project".to_string()));
            }
            _ => panic!("expected Reindex command"),
        }
    }

    #[test]
    fn test_reindex_custom_path() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "reindex", "--path", "/tmp/my-store"]);
        match cli.command {
            Commands::Reindex { path, .. } => {
                assert_eq!(path, PathBuf::from("/tmp/my-store"));
            }
            _ => panic!("expected Reindex command"),
        }
    }

    #[test]
    fn test_reindex_default_path() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "reindex"]);
        match cli.command {
            Commands::Reindex { path, .. } => {
                assert_eq!(path, PathBuf::from("engram-store"));
            }
            _ => panic!("expected Reindex command"),
        }
    }

    #[test]
    fn test_reindex_all_flags() {
        use clap::Parser;
        let cli = Cli::parse_from([
            "engram",
            "reindex",
            "--full",
            "--paths",
            "**/*.rs",
            "--repo",
            "my-repo",
            "--path",
            "/tmp/store",
        ]);
        match cli.command {
            Commands::Reindex {
                full,
                paths,
                repo,
                path,
                ..
            } => {
                assert!(full);
                assert_eq!(paths, Some("**/*.rs".to_string()));
                assert_eq!(repo, Some("my-repo".to_string()));
                assert_eq!(path, PathBuf::from("/tmp/store"));
            }
            _ => panic!("expected Reindex command"),
        }
    }

    // --- Config loading tests ---

    #[test]
    fn test_load_config_from_store() {
        let (_dir, store_path) = init_store_with_sources(vec![]);
        let config = load_config(&store_path).unwrap();
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.embedding.provider, "ollama");
    }

    #[test]
    fn test_load_config_with_sources() {
        let sources = vec![SourceConfig {
            name: "my-project".to_string(),
            path: "/tmp/project".to_string(),
            include: vec!["**/*.rs".to_string()],
            exclude: vec![],
        }];
        let (_dir, store_path) = init_store_with_sources(sources);
        let config = load_config(&store_path).unwrap();
        assert_eq!(config.sources.len(), 1);
        assert_eq!(config.sources[0].name, "my-project");
    }

    #[test]
    fn test_load_config_missing_store() {
        let result = load_config(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    // --- Pipeline integration tests ---

    #[tokio::test]
    async fn test_run_reindex_full() {
        let (_src, src_path) = setup_source_repo(&[
            ("src/main.rs", "fn main() {\n    println!(\"hello\");\n}"),
            ("README.md", "# Hello\nWorld"),
        ]);
        let sources = vec![SourceConfig {
            name: "test-repo".to_string(),
            path: src_path.to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        }];
        let (_store, store_path) = init_store_with_sources(sources.clone());
        let provider = MockProvider { dims: 4 };

        let report = run_reindex(&store_path, &provider, sources, true, None)
            .await
            .unwrap();

        assert_eq!(report.files_processed, 2);
        assert!(report.chunks_created > 0);
        assert_eq!(report.chunks_skipped, 0);
        assert!(report.embed_calls > 0);
    }

    #[tokio::test]
    async fn test_run_reindex_incremental() {
        let (_src, src_path) = setup_source_repo(&[(
            "lib.rs",
            "fn hello() {\n    1\n}\n\nfn world() {\n    2\n}",
        )]);
        let sources = vec![SourceConfig {
            name: "repo".to_string(),
            path: src_path.to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        }];
        let (_store, store_path) = init_store_with_sources(sources.clone());
        let provider = MockProvider { dims: 4 };

        // Full ingest first
        let r1 = run_reindex(&store_path, &provider, sources.clone(), true, None)
            .await
            .unwrap();
        assert!(r1.chunks_created > 0);

        // Modify source and commit
        fs::write(
            src_path.join("lib.rs"),
            "fn hello() {\n    1\n}\n\nfn world_v2() {\n    3\n}",
        )
        .unwrap();
        {
            let repo = git2::Repository::open(&src_path).unwrap();
            let mut index = repo.index().unwrap();
            index
                .add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
                .unwrap();
            index.write().unwrap();
            let tree_oid = index.write_tree().unwrap();
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = git2::Signature::now("test", "test@test.com").unwrap();
            let head = repo.head().unwrap().peel_to_commit().unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "modify", &tree, &[&head])
                .unwrap();
        }

        // Incremental ingest
        let r2 = run_reindex(&store_path, &provider, sources, false, None)
            .await
            .unwrap();
        assert_eq!(r2.files_processed, 1);
        assert!(r2.chunks_skipped > 0);
    }

    #[tokio::test]
    async fn test_run_reindex_with_paths_filter() {
        let (_src, src_path) = setup_source_repo(&[
            ("src/main.rs", "fn main() {\n    println!(\"hello\");\n}"),
            ("README.md", "# Hello\nWorld"),
            ("docs/guide.md", "# Guide\nSome content"),
        ]);
        let sources = vec![SourceConfig {
            name: "repo".to_string(),
            path: src_path.to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        }];
        let (_store, store_path) = init_store_with_sources(sources.clone());
        let provider = MockProvider { dims: 4 };

        // Only reindex .rs files
        let report = run_reindex(&store_path, &provider, sources, true, Some("**/*.rs"))
            .await
            .unwrap();

        assert_eq!(report.files_processed, 1);
    }

    #[tokio::test]
    async fn test_run_reindex_prints_report() {
        let (_src, src_path) =
            setup_source_repo(&[("lib.rs", "fn hello() {\n    42\n}")]);
        let sources = vec![SourceConfig {
            name: "repo".to_string(),
            path: src_path.to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        }];
        let (_store, store_path) = init_store_with_sources(sources.clone());
        let provider = MockProvider { dims: 4 };

        let report = run_reindex(&store_path, &provider, sources, true, None)
            .await
            .unwrap();

        // Verify report has expected fields populated
        assert!(report.files_processed > 0);
        assert!(report.chunks_created > 0);
        assert!(report.embed_calls > 0);
    }

    // --- Existing init tests ---

    #[test]
    fn test_init_creates_store_at_path() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("my-store");
        let config = StoreConfig::default();

        Store::init_local(&store_path, &config).unwrap();

        assert!(store_path.join(".engram").exists());
        assert!(store_path.join("engram.config.yaml").exists());
    }

    #[test]
    fn test_store_already_exists_detection() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("my-store");
        let config = StoreConfig::default();

        Store::init_local(&store_path, &config).unwrap();

        // The .engram directory should exist now
        assert!(store_path.join(".engram").exists());
    }

    #[test]
    fn test_default_path_value() {
        use clap::Parser;

        let cli = Cli::parse_from(["engram", "init", "--local"]);
        match cli.command {
            Commands::Init { local, remote, path } => {
                assert!(local);
                assert!(remote.is_none());
                assert_eq!(path, PathBuf::from("engram-store"));
            }
            _ => panic!("expected Init command"),
        }
    }

    #[test]
    fn test_custom_path_value() {
        use clap::Parser;

        let cli = Cli::parse_from(["engram", "init", "--local", "--path", "/tmp/my-store"]);
        match cli.command {
            Commands::Init { local, remote, path } => {
                assert!(local);
                assert!(remote.is_none());
                assert_eq!(path, PathBuf::from("/tmp/my-store"));
            }
            _ => panic!("expected Init command"),
        }
    }

    #[test]
    fn test_init_prints_success_path() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("test-store");
        let config = StoreConfig::default();

        Store::init_local(&store_path, &config).unwrap();

        // Verify the store was created with expected structure
        let version = fs::read_to_string(store_path.join(".engram/version")).unwrap();
        assert_eq!(version, "1.0.0");
    }

    // --- Serve command arg parsing tests ---

    #[test]
    fn test_serve_defaults() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "serve"]);
        match cli.command {
            Commands::Serve {
                transport,
                port,
                context,
                path,
            } => {
                assert!(matches!(transport, Transport::Stdio));
                assert_eq!(port, 3000);
                assert_eq!(context, "default");
                assert_eq!(path, PathBuf::from("engram-store"));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_stdio_transport() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "serve", "--transport", "stdio"]);
        match cli.command {
            Commands::Serve { transport, .. } => {
                assert!(matches!(transport, Transport::Stdio));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_sse_transport_with_port() {
        use clap::Parser;
        let cli = Cli::parse_from([
            "engram",
            "serve",
            "--transport",
            "sse",
            "--port",
            "8080",
        ]);
        match cli.command {
            Commands::Serve {
                transport, port, ..
            } => {
                assert!(matches!(transport, Transport::Sse));
                assert_eq!(port, 8080);
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_custom_context() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "serve", "--context", "my-project"]);
        match cli.command {
            Commands::Serve { context, .. } => {
                assert_eq!(context, "my-project");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_custom_path() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "serve", "--path", "/tmp/my-store"]);
        match cli.command {
            Commands::Serve { path, .. } => {
                assert_eq!(path, PathBuf::from("/tmp/my-store"));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_all_flags() {
        use clap::Parser;
        let cli = Cli::parse_from([
            "engram",
            "serve",
            "--transport",
            "sse",
            "--port",
            "9090",
            "--context",
            "workspace",
            "--path",
            "/opt/store",
        ]);
        match cli.command {
            Commands::Serve {
                transport,
                port,
                context,
                path,
            } => {
                assert!(matches!(transport, Transport::Sse));
                assert_eq!(port, 9090);
                assert_eq!(context, "workspace");
                assert_eq!(path, PathBuf::from("/opt/store"));
            }
            _ => panic!("expected Serve command"),
        }
    }

    // --- Status command tests ---

    #[test]
    fn test_status_default_path() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "status"]);
        match cli.command {
            Commands::Status { path } => {
                assert_eq!(path, PathBuf::from("engram-store"));
            }
            _ => panic!("expected Status command"),
        }
    }

    #[test]
    fn test_status_custom_path() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "status", "--path", "/tmp/my-store"]);
        match cli.command {
            Commands::Status { path } => {
                assert_eq!(path, PathBuf::from("/tmp/my-store"));
            }
            _ => panic!("expected Status command"),
        }
    }

    #[test]
    fn test_status_reads_config() {
        let sources = vec![SourceConfig {
            name: "my-project".to_string(),
            path: "/tmp/project".to_string(),
            include: vec![],
            exclude: vec![],
        }];
        let (_dir, store_path) = init_store_with_sources(sources);
        let config = load_config(&store_path).unwrap();
        assert_eq!(config.sources.len(), 1);
        assert_eq!(config.sources[0].name, "my-project");
    }

    #[test]
    fn test_status_reads_manifest() {
        let (_dir, store_path) = init_store_with_sources(vec![]);
        let manifest = Manifest {
            chunk_count: 42,
            last_indexed_commits: [("my-repo".to_string(), "abc123".to_string())].into_iter().collect(),
            model_name: "nomic-embed-text".to_string(),
            dimensions: 768,
            source_repos: vec!["my-repo".to_string()],
            created_at: "2026-03-09T00:00:00Z".to_string(),
            updated_at: "2026-03-09T12:00:00Z".to_string(),
        };
        write_manifest(&store_path, &manifest).unwrap();

        let read = read_manifest(&store_path).unwrap().unwrap();
        assert_eq!(read.chunk_count, 42);
        assert_eq!(read.last_indexed_commits.get("my-repo").map(|s| s.as_str()), Some("abc123"));
    }

    #[test]
    fn test_status_no_manifest_returns_none() {
        let (_dir, store_path) = init_store_with_sources(vec![]);
        // No manifest written — should return None
        let result = read_manifest(&store_path).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_status_cache_detection() {
        let (_dir, store_path) = init_store_with_sources(vec![]);
        // No cache yet
        assert!(!store_path.join(".engram-cache").join("fingerprint").exists());

        // Create cache fingerprint
        let cache_dir = store_path.join(".engram-cache");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::write(cache_dir.join("fingerprint"), "test-hash").unwrap();
        assert!(store_path.join(".engram-cache").join("fingerprint").exists());
    }

    #[test]
    fn test_status_no_store_detection() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("nonexistent");
        assert!(!path.join(".engram").exists());
    }

    #[test]
    fn test_status_staleness_check_with_source_repo() {
        let (_src, src_path) = setup_source_repo(&[("lib.rs", "fn hello() {}")]);

        // Get the HEAD commit OID
        let repo = git2::Repository::open(&src_path).unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        let head_oid = head.id().to_string();

        // Create store with manifest matching the HEAD
        let sources = vec![SourceConfig {
            name: "test-repo".to_string(),
            path: src_path.to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        }];
        let (_dir, store_path) = init_store_with_sources(sources);
        let manifest = Manifest {
            chunk_count: 1,
            last_indexed_commits: [("test-repo".to_string(), head_oid.clone())].into_iter().collect(),
            model_name: "test".to_string(),
            dimensions: 768,
            source_repos: vec!["test-repo".to_string()],
            created_at: "2026-03-09T00:00:00Z".to_string(),
            updated_at: "2026-03-09T12:00:00Z".to_string(),
        };
        write_manifest(&store_path, &manifest).unwrap();

        // Verify: indexed commit matches HEAD — should be "up to date"
        let read = read_manifest(&store_path).unwrap().unwrap();
        assert_eq!(read.last_indexed_commits.get("test-repo").map(|s| s.as_str()), Some(head_oid.as_str()));
    }

    // --- Onboard command tests ---

    #[test]
    fn test_onboard_default_depth() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "onboard"]);
        match cli.command {
            Commands::Onboard { depth, repo, path } => {
                assert!(matches!(depth, DepthArg::Standard));
                assert!(repo.is_none());
                assert_eq!(path, PathBuf::from("engram-store"));
            }
            _ => panic!("expected Onboard command"),
        }
    }

    #[test]
    fn test_onboard_quick_depth() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "onboard", "--depth", "quick"]);
        match cli.command {
            Commands::Onboard { depth, .. } => {
                assert!(matches!(depth, DepthArg::Quick));
            }
            _ => panic!("expected Onboard command"),
        }
    }

    #[test]
    fn test_onboard_deep_depth() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "onboard", "--depth", "deep"]);
        match cli.command {
            Commands::Onboard { depth, .. } => {
                assert!(matches!(depth, DepthArg::Deep));
            }
            _ => panic!("expected Onboard command"),
        }
    }

    #[test]
    fn test_onboard_repo_flag() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "onboard", "--repo", "my-project"]);
        match cli.command {
            Commands::Onboard { repo, .. } => {
                assert_eq!(repo, Some("my-project".to_string()));
            }
            _ => panic!("expected Onboard command"),
        }
    }

    #[test]
    fn test_onboard_custom_path() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "onboard", "--path", "/tmp/my-store"]);
        match cli.command {
            Commands::Onboard { path, .. } => {
                assert_eq!(path, PathBuf::from("/tmp/my-store"));
            }
            _ => panic!("expected Onboard command"),
        }
    }

    #[test]
    fn test_onboard_all_flags() {
        use clap::Parser;
        let cli = Cli::parse_from([
            "engram",
            "onboard",
            "--depth",
            "quick",
            "--repo",
            "my-repo",
            "--path",
            "/tmp/store",
        ]);
        match cli.command {
            Commands::Onboard { depth, repo, path } => {
                assert!(matches!(depth, DepthArg::Quick));
                assert_eq!(repo, Some("my-repo".to_string()));
                assert_eq!(path, PathBuf::from("/tmp/store"));
            }
            _ => panic!("expected Onboard command"),
        }
    }

    #[test]
    fn test_onboard_depth_arg_converts_to_onboarding_depth() {
        use engram_core::OnboardingDepth;
        assert_eq!(OnboardingDepth::from(DepthArg::Quick), OnboardingDepth::Quick);
        assert_eq!(OnboardingDepth::from(DepthArg::Standard), OnboardingDepth::Standard);
        assert_eq!(OnboardingDepth::from(DepthArg::Deep), OnboardingDepth::Deep);
    }

    #[tokio::test]
    async fn test_onboard_runs_pipeline() {
        let (_src, src_path) = setup_source_repo(&[
            ("Cargo.toml", "[package]\nname = \"test\"\nversion = \"0.1.0\"\nedition = \"2021\""),
            ("src/main.rs", "fn main() {\n    println!(\"hello\");\n}"),
        ]);
        let source = SourceConfig {
            name: "test-repo".to_string(),
            path: src_path.to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        };
        let (_dir, store_path) = init_store_with_sources(vec![source.clone()]);

        let report = engram_ingest::run_onboarding(
            &src_path,
            &store_path,
            &source,
            engram_core::OnboardingDepth::Quick,
        )
        .await
        .unwrap();

        assert!(report.metadata_detected);
        assert!(report.commands_extracted);
        assert!(!report.architecture_analyzed);
        assert!(!report.abstractions_extracted);
        assert_eq!(report.files_written.len(), 2);
    }

    #[tokio::test]
    async fn test_onboard_standard_depth_full_analysis() {
        let (_src, src_path) = setup_source_repo(&[
            ("Cargo.toml", "[package]\nname = \"test\"\nversion = \"0.1.0\"\nedition = \"2021\""),
            ("src/main.rs", "fn main() {\n    println!(\"hello\");\n}"),
        ]);
        let source = SourceConfig {
            name: "test-repo".to_string(),
            path: src_path.to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        };
        let (_dir, store_path) = init_store_with_sources(vec![source.clone()]);

        let report = engram_ingest::run_onboarding(
            &src_path,
            &store_path,
            &source,
            engram_core::OnboardingDepth::Standard,
        )
        .await
        .unwrap();

        assert!(report.metadata_detected);
        assert!(report.commands_extracted);
        assert!(report.architecture_analyzed);
        assert!(report.abstractions_extracted);
        assert_eq!(report.files_written.len(), 4);
    }

    // --- Compact command tests ---

    #[test]
    fn test_compact_default_path() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "compact"]);
        match cli.command {
            Commands::Compact { path } => {
                assert_eq!(path, PathBuf::from("engram-store"));
            }
            _ => panic!("expected Compact command"),
        }
    }

    #[test]
    fn test_compact_custom_path() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "compact", "--path", "/tmp/my-store"]);
        match cli.command {
            Commands::Compact { path } => {
                assert_eq!(path, PathBuf::from("/tmp/my-store"));
            }
            _ => panic!("expected Compact command"),
        }
    }

    #[test]
    fn test_compact_runs_on_store() {
        use engram_store::compact_snapshots;

        let (_dir, store_path) = init_store_with_sources(vec![]);

        // No snapshots yet — should return zero counts
        let report = compact_snapshots(&store_path).unwrap();
        assert_eq!(report.active_to_compressed, 0);
        assert_eq!(report.compressed_to_archived, 0);
    }

    // --- Sync command arg parsing tests ---

    #[test]
    fn test_sync_defaults_to_both() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "sync"]);
        match cli.command {
            Commands::Sync { mode, path } => {
                assert!(matches!(mode, SyncMode::Both));
                assert_eq!(path, PathBuf::from("engram-store"));
            }
            _ => panic!("expected Sync command"),
        }
    }

    #[test]
    fn test_sync_pull_mode() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "sync", "--mode", "pull"]);
        match cli.command {
            Commands::Sync { mode, .. } => {
                assert!(matches!(mode, SyncMode::Pull));
            }
            _ => panic!("expected Sync command"),
        }
    }

    #[test]
    fn test_sync_push_mode() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "sync", "--mode", "push"]);
        match cli.command {
            Commands::Sync { mode, .. } => {
                assert!(matches!(mode, SyncMode::Push));
            }
            _ => panic!("expected Sync command"),
        }
    }

    #[test]
    fn test_sync_custom_path() {
        use clap::Parser;
        let cli = Cli::parse_from(["engram", "sync", "--path", "/tmp/my-store"]);
        match cli.command {
            Commands::Sync { path, .. } => {
                assert_eq!(path, PathBuf::from("/tmp/my-store"));
            }
            _ => panic!("expected Sync command"),
        }
    }

    // --- Init --remote arg parsing tests ---

    #[test]
    fn test_init_remote_flag() {
        use clap::Parser;
        let cli = Cli::parse_from([
            "engram",
            "init",
            "--remote",
            "https://github.com/org/repo.git",
        ]);
        match cli.command {
            Commands::Init {
                local,
                remote,
                path,
            } => {
                assert!(!local);
                assert_eq!(remote, Some("https://github.com/org/repo.git".to_string()));
                assert_eq!(path, PathBuf::from("engram-store"));
            }
            _ => panic!("expected Init command"),
        }
    }

    #[test]
    fn test_init_remote_with_path() {
        use clap::Parser;
        let cli = Cli::parse_from([
            "engram",
            "init",
            "--remote",
            "https://github.com/org/repo.git",
            "--path",
            "/tmp/store",
        ]);
        match cli.command {
            Commands::Init {
                local,
                remote,
                path,
            } => {
                assert!(!local);
                assert_eq!(remote, Some("https://github.com/org/repo.git".to_string()));
                assert_eq!(path, PathBuf::from("/tmp/store"));
            }
            _ => panic!("expected Init command"),
        }
    }

    #[test]
    fn test_init_local_and_remote_conflict() {
        use clap::Parser;
        let result = Cli::try_parse_from([
            "engram",
            "init",
            "--local",
            "--remote",
            "https://github.com/org/repo.git",
        ]);
        assert!(result.is_err());
    }
}
