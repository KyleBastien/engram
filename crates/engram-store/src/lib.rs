mod jsonl;

use std::fs;
use std::path::{Path, PathBuf};

use engram_core::{EngramError, Result, StoreConfig};
use git2::{Repository, Signature};
use uuid::Uuid;

pub use jsonl::{read_chunks_jsonl, write_chunks_jsonl};

/// A git-backed semantic store.
pub struct Store {
    /// Path to the store root directory.
    pub path: PathBuf,
    /// The git repository backing this store.
    #[allow(dead_code)]
    repo: Repository,
}

impl Store {
    /// Initialize a new local store at the given path.
    ///
    /// Creates the directory structure, writes config and metadata files,
    /// initialises a git repository, and makes the initial commit.
    pub fn init_local(path: &Path, config: &StoreConfig) -> Result<Store> {
        // Create root directory
        fs::create_dir_all(path)?;

        // Create .engram metadata directory
        let engram_dir = path.join(".engram");
        fs::create_dir_all(&engram_dir)?;
        fs::write(engram_dir.join("version"), "1.0.0")?;
        fs::write(engram_dir.join("store-id"), Uuid::new_v4().to_string())?;

        // Create directory structure
        let dirs = [
            "index",
            "knowledge/decisions",
            "knowledge/lessons",
            "knowledge/patterns",
            "knowledge/glossary",
            "knowledge/onboarding",
            "snapshots/active",
            "snapshots/compressed",
            "snapshots/archived",
            "metrics/baseline",
            "metrics/runs",
        ];
        for dir in &dirs {
            fs::create_dir_all(path.join(dir))?;
        }

        // Write config file
        let yaml = serde_yaml::to_string(config)
            .map_err(|e| EngramError::Config(e.to_string()))?;
        fs::write(path.join("engram.config.yaml"), yaml)?;

        // Write .gitignore
        fs::write(path.join(".gitignore"), ".engram-cache/\n")?;

        // Initialize git repo and make initial commit
        let repo = Repository::init(path)
            .map_err(|e| EngramError::Git(e.to_string()))?;

        {
            // Stage all files
            let mut index = repo.index()
                .map_err(|e| EngramError::Git(e.to_string()))?;
            index
                .add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
                .map_err(|e| EngramError::Git(e.to_string()))?;
            index.write().map_err(|e| EngramError::Git(e.to_string()))?;
            let tree_oid = index
                .write_tree()
                .map_err(|e| EngramError::Git(e.to_string()))?;
            let tree = repo
                .find_tree(tree_oid)
                .map_err(|e| EngramError::Git(e.to_string()))?;

            let sig = Signature::now("engram", "engram@localhost")
                .map_err(|e| EngramError::Git(e.to_string()))?;
            repo.commit(Some("HEAD"), &sig, &sig, "engram: initialize store", &tree, &[])
                .map_err(|e| EngramError::Git(e.to_string()))?;
        }

        Ok(Store {
            path: path.to_path_buf(),
            repo,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::StoreConfig;
    use tempfile::TempDir;

    #[test]
    fn test_init_local_creates_directory_structure() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("test-store");
        let config = StoreConfig::default();

        let store = Store::init_local(&store_path, &config).unwrap();
        assert_eq!(store.path, store_path);

        // Verify .engram metadata
        assert!(store_path.join(".engram/version").exists());
        assert!(store_path.join(".engram/store-id").exists());
        let version = fs::read_to_string(store_path.join(".engram/version")).unwrap();
        assert_eq!(version, "1.0.0");
        let store_id = fs::read_to_string(store_path.join(".engram/store-id")).unwrap();
        // Verify it's a valid UUID
        Uuid::parse_str(&store_id).unwrap();

        // Verify directory structure
        assert!(store_path.join("index").is_dir());
        assert!(store_path.join("knowledge/decisions").is_dir());
        assert!(store_path.join("knowledge/lessons").is_dir());
        assert!(store_path.join("knowledge/patterns").is_dir());
        assert!(store_path.join("knowledge/glossary").is_dir());
        assert!(store_path.join("knowledge/onboarding").is_dir());
        assert!(store_path.join("snapshots/active").is_dir());
        assert!(store_path.join("snapshots/compressed").is_dir());
        assert!(store_path.join("snapshots/archived").is_dir());
        assert!(store_path.join("metrics/baseline").is_dir());
        assert!(store_path.join("metrics/runs").is_dir());
    }

    #[test]
    fn test_init_local_writes_config() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("test-store");
        let config = StoreConfig::default();

        Store::init_local(&store_path, &config).unwrap();

        let yaml = fs::read_to_string(store_path.join("engram.config.yaml")).unwrap();
        let deserialized: StoreConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(deserialized, config);
    }

    #[test]
    fn test_init_local_writes_gitignore() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("test-store");
        let config = StoreConfig::default();

        Store::init_local(&store_path, &config).unwrap();

        let gitignore = fs::read_to_string(store_path.join(".gitignore")).unwrap();
        assert!(gitignore.contains(".engram-cache/"));
    }

    #[test]
    fn test_init_local_creates_git_repo_with_commit() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("test-store");
        let config = StoreConfig::default();

        Store::init_local(&store_path, &config).unwrap();

        // Verify it's a git repository
        let repo = Repository::open(&store_path).unwrap();
        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        assert_eq!(commit.message().unwrap(), "engram: initialize store");
    }

    #[test]
    fn test_init_local_all_files_committed() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("test-store");
        let config = StoreConfig::default();

        Store::init_local(&store_path, &config).unwrap();

        // Verify working directory is clean (no untracked or modified files)
        let repo = Repository::open(&store_path).unwrap();
        let statuses = repo.statuses(None).unwrap();
        assert!(
            statuses.is_empty(),
            "Expected clean working directory, found {} dirty entries",
            statuses.len()
        );
    }

    #[test]
    fn test_init_local_with_custom_config() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("test-store");
        let mut config = StoreConfig::default();
        config.embedding.model = "custom-model".to_string();
        config.embedding.dimensions = 1024;
        config.search.default_limit = 50;

        Store::init_local(&store_path, &config).unwrap();

        let yaml = fs::read_to_string(store_path.join("engram.config.yaml")).unwrap();
        let deserialized: StoreConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(deserialized.embedding.model, "custom-model");
        assert_eq!(deserialized.embedding.dimensions, 1024);
        assert_eq!(deserialized.search.default_limit, 50);
    }
}
