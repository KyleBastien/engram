use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};
use engram_core::StoreConfig;
use engram_store::Store;

#[derive(Parser)]
#[command(name = "engram", about = "Git-backed semantic context for AI coding agents")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new semantic store
    Init {
        /// Create a local store
        #[arg(long)]
        local: bool,

        /// Path for the store directory (defaults to ./engram-store)
        #[arg(long, default_value = "engram-store")]
        path: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
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
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use engram_core::StoreConfig;
    use engram_store::Store;
    use tempfile::TempDir;

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
        use std::path::PathBuf;

        use super::Cli;

        let cli = Cli::parse_from(["engram", "init", "--local"]);
        match cli.command {
            super::Commands::Init { local, path } => {
                assert!(local);
                assert_eq!(path, PathBuf::from("engram-store"));
            }
        }
    }

    #[test]
    fn test_custom_path_value() {
        use clap::Parser;
        use std::path::PathBuf;

        use super::Cli;

        let cli = Cli::parse_from(["engram", "init", "--local", "--path", "/tmp/my-store"]);
        match cli.command {
            super::Commands::Init { local, path } => {
                assert!(local);
                assert_eq!(path, PathBuf::from("/tmp/my-store"));
            }
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
}
