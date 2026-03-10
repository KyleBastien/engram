use engram_core::{EngramError, Result};
use git2::{Oid, Repository, Signature, StatusOptions};

/// Stage all changes (new, modified, deleted) and commit with the given message.
///
/// Returns the commit OID on success. If there are no changes to commit,
/// returns `Ok(None)` without creating an empty commit.
pub fn commit_changes(repo: &Repository, message: &str) -> Result<Option<Oid>> {
    // Check if there are any changes to commit
    let mut opts = StatusOptions::new();
    opts.include_untracked(true);
    let statuses = repo
        .statuses(Some(&mut opts))
        .map_err(|e| EngramError::Git(e.to_string()))?;

    if statuses.is_empty() {
        return Ok(None);
    }

    // Stage all changes
    let mut index = repo
        .index()
        .map_err(|e| EngramError::Git(e.to_string()))?;
    index
        .add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
        .map_err(|e| EngramError::Git(e.to_string()))?;

    // Remove deleted files from the index
    for entry in statuses.iter() {
        let status = entry.status();
        if status.contains(git2::Status::WT_DELETED) || status.contains(git2::Status::INDEX_DELETED)
        {
            if let Some(path) = entry.path() {
                index
                    .remove_path(std::path::Path::new(path))
                    .map_err(|e| EngramError::Git(e.to_string()))?;
            }
        }
    }

    index
        .write()
        .map_err(|e| EngramError::Git(e.to_string()))?;
    let tree_oid = index
        .write_tree()
        .map_err(|e| EngramError::Git(e.to_string()))?;
    let tree = repo
        .find_tree(tree_oid)
        .map_err(|e| EngramError::Git(e.to_string()))?;

    let sig = Signature::now("engram", "engram@local")
        .map_err(|e| EngramError::Git(e.to_string()))?;

    // Get parent commit if HEAD exists
    let parent_commit = repo.head().ok().and_then(|head| head.peel_to_commit().ok());
    let parents: Vec<&git2::Commit> = parent_commit.iter().collect();

    let oid = repo
        .commit(Some("HEAD"), &sig, &sig, message, &tree, &parents)
        .map_err(|e| EngramError::Git(e.to_string()))?;

    Ok(Some(oid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn init_repo(tmp: &TempDir) -> Repository {
        let repo = Repository::init(tmp.path()).unwrap();
        // Make an initial commit so HEAD exists
        {
            let mut index = repo.index().unwrap();
            let tree_oid = index.write_tree().unwrap();
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = Signature::now("test", "test@test").unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "initial", &tree, &[])
                .unwrap();
        }
        repo
    }

    #[test]
    fn test_commit_new_file() {
        let tmp = TempDir::new().unwrap();
        let repo = init_repo(&tmp);

        fs::write(tmp.path().join("hello.txt"), "hello world").unwrap();

        let result = commit_changes(&repo, "add hello").unwrap();
        assert!(result.is_some());

        let head = repo.head().unwrap().peel_to_commit().unwrap();
        assert_eq!(head.message().unwrap(), "add hello");
        assert_eq!(head.author().name().unwrap(), "engram");
        assert_eq!(head.author().email().unwrap(), "engram@local");
    }

    #[test]
    fn test_commit_modified_file() {
        let tmp = TempDir::new().unwrap();
        let repo = init_repo(&tmp);

        // Create and commit a file first
        fs::write(tmp.path().join("file.txt"), "v1").unwrap();
        commit_changes(&repo, "add file").unwrap();

        // Modify the file
        fs::write(tmp.path().join("file.txt"), "v2").unwrap();
        let result = commit_changes(&repo, "update file").unwrap();
        assert!(result.is_some());

        let head = repo.head().unwrap().peel_to_commit().unwrap();
        assert_eq!(head.message().unwrap(), "update file");
    }

    #[test]
    fn test_commit_deleted_file() {
        let tmp = TempDir::new().unwrap();
        let repo = init_repo(&tmp);

        // Create and commit a file first
        fs::write(tmp.path().join("file.txt"), "content").unwrap();
        commit_changes(&repo, "add file").unwrap();

        // Delete the file
        fs::remove_file(tmp.path().join("file.txt")).unwrap();
        let result = commit_changes(&repo, "delete file").unwrap();
        assert!(result.is_some());

        let head = repo.head().unwrap().peel_to_commit().unwrap();
        assert_eq!(head.message().unwrap(), "delete file");

        // Verify the file is gone from the tree
        let tree = head.tree().unwrap();
        assert!(tree.get_name("file.txt").is_none());
    }

    #[test]
    fn test_no_changes_returns_none() {
        let tmp = TempDir::new().unwrap();
        let repo = init_repo(&tmp);

        let result = commit_changes(&repo, "empty").unwrap();
        assert!(result.is_none());

        // HEAD should still be the initial commit
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        assert_eq!(head.message().unwrap(), "initial");
    }

    #[test]
    fn test_returns_commit_oid() {
        let tmp = TempDir::new().unwrap();
        let repo = init_repo(&tmp);

        fs::write(tmp.path().join("test.txt"), "data").unwrap();

        let oid = commit_changes(&repo, "test oid").unwrap().unwrap();

        // Verify the OID matches HEAD
        let head_oid = repo.head().unwrap().target().unwrap();
        assert_eq!(oid, head_oid);
    }

    #[test]
    fn test_stages_all_changes() {
        let tmp = TempDir::new().unwrap();
        let repo = init_repo(&tmp);

        // Create multiple files
        fs::write(tmp.path().join("a.txt"), "aaa").unwrap();
        fs::write(tmp.path().join("b.txt"), "bbb").unwrap();
        fs::create_dir_all(tmp.path().join("sub")).unwrap();
        fs::write(tmp.path().join("sub/c.txt"), "ccc").unwrap();

        commit_changes(&repo, "add multiple").unwrap();

        // Working directory should be clean
        let statuses = repo.statuses(None).unwrap();
        assert!(
            statuses.is_empty(),
            "Expected clean working directory, found {} dirty entries",
            statuses.len()
        );
    }
}
