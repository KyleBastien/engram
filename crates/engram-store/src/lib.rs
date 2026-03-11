mod commit;
mod embeddings;
mod jsonl;
mod knowledge;
mod manifest;
mod paths;
mod snapshot;

use std::fs;
use std::path::{Path, PathBuf};

use engram_core::{EngramError, Result, StoreConfig};
use git2::{Repository, Signature};
use uuid::Uuid;

pub use commit::commit_changes;
pub use embeddings::{read_embeddings_bin, write_embeddings_bin, EmbeddingFile};
pub use jsonl::{read_chunks_jsonl, write_chunks_jsonl};
pub use manifest::{read_manifest, write_manifest};
pub use knowledge::{
    decision_embed_text, knowledge_embedding_path, lesson_embed_text, pattern_embed_text,
    scan_knowledge_embeddings, write_decision, write_decision_with_embedding,
    write_glossary_entry, write_knowledge_embedding, write_lesson, write_lesson_with_embedding,
    write_pattern, write_pattern_with_embedding, KnowledgeEmbeddingInfo,
};
pub use paths::{chunks_path, embeddings_path};
pub use snapshot::{
    compact_snapshots, read_snapshot, scan_snapshot_embeddings, snapshot_embed_text,
    snapshot_embedding_path, write_snapshot, write_snapshot_embedding,
    write_snapshot_with_embedding, CompactReport, SnapshotEmbeddingInfo,
};

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

    /// Initialize a remote-backed store.
    ///
    /// If the remote repository has commits, clones it to `local_path`.
    /// If the remote repository is empty, initializes a local store and pushes the initial commit.
    /// The remote URL is stored in `engram.config.yaml` under `store.remote`.
    pub fn init_remote(url: &str, local_path: &Path, config: &StoreConfig) -> Result<Store> {
        // Try to clone — check if the result has actual content (HEAD exists)
        let clone_result = Repository::clone(url, local_path);
        let remote_has_content = clone_result
            .as_ref()
            .ok()
            .is_some_and(|repo| repo.head().is_ok());

        if remote_has_content {
            let repo = clone_result.unwrap();
            // Clone succeeded with content — update config to include remote URL
            let mut updated_config = if local_path.join("engram.config.yaml").exists() {
                let yaml = fs::read_to_string(local_path.join("engram.config.yaml"))
                    .map_err(|e| EngramError::Config(e.to_string()))?;
                serde_yaml::from_str::<StoreConfig>(&yaml)
                    .map_err(|e| EngramError::Config(e.to_string()))?
            } else {
                config.clone()
            };
            updated_config.store.remote = Some(url.to_string());
            let yaml = serde_yaml::to_string(&updated_config)
                .map_err(|e| EngramError::Config(e.to_string()))?;
            fs::write(local_path.join("engram.config.yaml"), yaml)?;

            // Commit config update if it changed
            let _ = commit_changes(&repo, "engram: set remote URL in config");

            Ok(Store {
                path: local_path.to_path_buf(),
                repo,
            })
        } else {
            // Remote is empty or clone failed — initialize locally and push
            // Drop the clone result before cleaning up
            drop(clone_result);
            let _ = fs::remove_dir_all(local_path);

            let mut remote_config = config.clone();
            remote_config.store.remote = Some(url.to_string());

            Self::init_local(local_path, &remote_config)?;

            // Add the remote and push
            let repo = Repository::open(local_path)
                .map_err(|e| EngramError::Git(e.to_string()))?;
            repo.remote("origin", url)
                .map_err(|e| EngramError::Git(e.to_string()))?;

            {
                // Scope borrows so they're dropped before we move repo
                let head = repo.head().map_err(|e| EngramError::Git(e.to_string()))?;
                let branch_ref = head
                    .name()
                    .ok_or_else(|| {
                        EngramError::Git("HEAD is not a valid UTF-8 ref".to_string())
                    })?
                    .to_string();
                let refspec = format!("{branch_ref}:{branch_ref}");

                let mut remote = repo
                    .find_remote("origin")
                    .map_err(|e| EngramError::Git(e.to_string()))?;
                remote
                    .push(&[&refspec], None)
                    .map_err(|e| EngramError::Git(e.to_string()))?;
            }

            Ok(Store {
                path: local_path.to_path_buf(),
                repo,
            })
        }
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

    /// Helper: create a bare repo to act as a remote.
    fn create_bare_remote(tmp: &TempDir) -> PathBuf {
        let bare_path = tmp.path().join("remote.git");
        Repository::init_bare(&bare_path).unwrap();
        bare_path
    }

    #[test]
    fn test_init_remote_empty_remote_creates_store_and_pushes() {
        let tmp = TempDir::new().unwrap();
        let bare_path = create_bare_remote(&tmp);
        let local_path = tmp.path().join("local-store");
        let config = StoreConfig::default();

        let url = bare_path.to_str().unwrap();
        let store = Store::init_remote(url, &local_path, &config).unwrap();
        assert_eq!(store.path, local_path);

        // Verify local store has the directory structure
        assert!(local_path.join(".engram/version").exists());
        assert!(local_path.join("index").is_dir());
        assert!(local_path.join("engram.config.yaml").exists());

        // Verify remote URL stored in config
        let yaml = fs::read_to_string(local_path.join("engram.config.yaml")).unwrap();
        let deserialized: StoreConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(deserialized.store.remote.as_deref(), Some(url));

        // Verify the bare remote now has commits
        let remote_repo = Repository::open_bare(&bare_path).unwrap();
        let refs: Vec<_> = remote_repo.references().unwrap().collect();
        assert!(!refs.is_empty(), "Remote should have refs after push");
    }

    #[test]
    fn test_init_remote_existing_remote_clones() {
        let tmp = TempDir::new().unwrap();
        let bare_path = create_bare_remote(&tmp);
        let url = bare_path.to_str().unwrap();

        // First: populate the remote by initializing a store and pushing
        let first_local = tmp.path().join("first-local");
        Store::init_remote(url, &first_local, &StoreConfig::default()).unwrap();

        // Second: clone from the now-populated remote
        let second_local = tmp.path().join("second-local");
        let store = Store::init_remote(url, &second_local, &StoreConfig::default()).unwrap();
        assert_eq!(store.path, second_local);

        // Verify cloned store has committed files (git doesn't track empty dirs)
        assert!(second_local.join(".engram/version").exists());
        assert!(second_local.join("engram.config.yaml").exists());
        assert!(second_local.join(".gitignore").exists());

        // Verify remote URL stored in config
        let yaml = fs::read_to_string(second_local.join("engram.config.yaml")).unwrap();
        let deserialized: StoreConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(deserialized.store.remote.as_deref(), Some(url));
    }

    #[test]
    fn test_init_remote_stores_remote_in_config() {
        let tmp = TempDir::new().unwrap();
        let bare_path = create_bare_remote(&tmp);
        let local_path = tmp.path().join("local-store");
        let config = StoreConfig::default();

        let url = bare_path.to_str().unwrap();
        Store::init_remote(url, &local_path, &config).unwrap();

        // Read the config and check the remote field
        let yaml = fs::read_to_string(local_path.join("engram.config.yaml")).unwrap();
        let deserialized: StoreConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(deserialized.store.remote, Some(url.to_string()));
    }

    #[test]
    fn test_init_remote_local_has_origin() {
        let tmp = TempDir::new().unwrap();
        let bare_path = create_bare_remote(&tmp);
        let local_path = tmp.path().join("local-store");
        let config = StoreConfig::default();

        let url = bare_path.to_str().unwrap();
        Store::init_remote(url, &local_path, &config).unwrap();

        // Verify the local repo has an "origin" remote
        let repo = Repository::open(&local_path).unwrap();
        let remote = repo.find_remote("origin").unwrap();
        assert_eq!(remote.url().unwrap(), url);
    }
}
