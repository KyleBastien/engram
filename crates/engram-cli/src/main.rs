use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use engram_core::{EmbeddingProvider, SourceConfig, StoreConfig};
use engram_ingest::{IngestPipeline, IngestReport};
use engram_mcp::McpServer;
use engram_query::IndexManager;
use engram_store::Store;

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
        #[arg(long)]
        local: bool,

        /// Path for the store directory (defaults to ./engram-store)
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
    sources: Vec<SourceConfig>,
    full: bool,
    paths_filter: Option<&str>,
) -> engram_core::Result<IngestReport> {
    let mut total = IngestReport::default();
    for mut source in sources {
        if let Some(glob) = paths_filter {
            source.include = vec![glob.to_string()];
        }
        println!("Reindexing source '{}'...", source.name);
        let report = IngestPipeline::run(&source, store_path, provider, full).await?;
        println!(
            "  {} files processed, {} chunks created, {} skipped, {} deleted",
            report.files_processed,
            report.chunks_created,
            report.chunks_skipped,
            report.chunks_deleted,
        );
        total.files_processed += report.files_processed;
        total.chunks_created += report.chunks_created;
        total.chunks_skipped += report.chunks_skipped;
        total.chunks_deleted += report.chunks_deleted;
        total.embed_calls += report.embed_calls;
    }
    Ok(total)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            transport,
            port: _port,
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

            let search = index_manager.into_hybrid_search();
            let mut server =
                McpServer::with_engine_and_sources(search, Box::new(provider), source_roots);
            server.set_boot_info(boot_ms as u64, path.to_string_lossy().to_string(), cache_status);

            eprintln!("engram: serving on stdio (context: {context})");

            match transport {
                Transport::Stdio => {
                    let stdin = tokio::io::stdin();
                    let stdout = tokio::io::stdout();
                    if let Err(e) = server.run(stdin, stdout).await {
                        eprintln!("Error: MCP server error: {e}");
                        process::exit(1);
                    }
                }
                Transport::Sse => {
                    eprintln!("Error: SSE transport is not yet implemented (Phase 1 only supports stdio)");
                    process::exit(1);
                }
            }
        }
        Commands::Init { local, path } => {
            if !local {
                eprintln!("Error: --local flag is required for init");
                process::exit(1);
            }

            if path.join(".engram").exists() {
                eprintln!("Error: store already exists at {}", path.display());
                process::exit(1);
            }

            let config = StoreConfig::default();
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

    use super::{load_config, run_reindex, Cli, Commands, Transport};

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
            Commands::Init { local, path } => {
                assert!(local);
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
            Commands::Init { local, path } => {
                assert!(local);
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
}
