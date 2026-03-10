use std::collections::HashMap;
use std::path::Path;

use engram_core::{ChunkMetadata, Result};
use git2::Repository;

/// Checks which chunks are stale relative to the HEAD of the source repository.
///
/// A chunk is stale if its `source_commit` is not an ancestor of (or equal to) HEAD.
/// Returns a map from chunk_id to a boolean indicating staleness (true = stale).
pub fn check_staleness(
    chunks: &[ChunkMetadata],
    source_repo_path: &Path,
) -> Result<HashMap<String, bool>> {
    let repo = Repository::open(source_repo_path)
        .map_err(|e| engram_core::EngramError::Git(e.to_string()))?;

    let head_commit = repo
        .head()
        .and_then(|r| r.peel_to_commit())
        .map_err(|e| engram_core::EngramError::Git(e.to_string()))?;

    let head_oid = head_commit.id();

    let mut result = HashMap::new();

    for chunk in chunks {
        let stale = if chunk.source_commit.is_empty() {
            true
        } else {
            match git2::Oid::from_str(&chunk.source_commit) {
                Ok(commit_oid) => {
                    if commit_oid == head_oid {
                        false
                    } else {
                        // Check if source_commit is an ancestor of HEAD
                        match repo.graph_descendant_of(head_oid, commit_oid) {
                            Ok(is_ancestor) => !is_ancestor,
                            Err(_) => true,
                        }
                    }
                }
                Err(_) => true,
            }
        };

        result.insert(chunk.chunk_id.clone(), stale);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::{ChunkKind, ChunkMetadata};
    use git2::Signature;
    use std::fs;
    use tempfile::TempDir;

    fn make_chunk(chunk_id: &str, source_commit: &str) -> ChunkMetadata {
        ChunkMetadata {
            chunk_id: chunk_id.to_string(),
            kind: ChunkKind::Function,
            name: "test_fn".to_string(),
            signature: None,
            start_line: 1,
            end_line: 10,
            content_hash: "abc123".to_string(),
            tags: vec![],
            indexed_at: "2026-03-09T00:00:00Z".to_string(),
            source_commit: source_commit.to_string(),
            embedding_offset: 0,
        }
    }

    fn init_repo_with_commits(dir: &Path) -> (String, String) {
        let repo = Repository::init(dir).unwrap();
        let sig = Signature::now("test", "test@test.com").unwrap();

        // First commit
        fs::write(dir.join("file.txt"), "hello").unwrap();
        let mut index = repo.index().unwrap();
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None).unwrap();
        index.write().unwrap();
        let tree_oid = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_oid).unwrap();
        let first_oid = repo
            .commit(Some("HEAD"), &sig, &sig, "first commit", &tree, &[])
            .unwrap();

        // Second commit (HEAD)
        fs::write(dir.join("file2.txt"), "world").unwrap();
        let mut index = repo.index().unwrap();
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None).unwrap();
        index.write().unwrap();
        let tree_oid = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_oid).unwrap();
        let first_commit = repo.find_commit(first_oid).unwrap();
        let second_oid = repo
            .commit(Some("HEAD"), &sig, &sig, "second commit", &tree, &[&first_commit])
            .unwrap();

        (first_oid.to_string(), second_oid.to_string())
    }

    #[test]
    fn head_commit_is_not_stale() {
        let dir = TempDir::new().unwrap();
        let (_first, head) = init_repo_with_commits(dir.path());

        let chunks = vec![make_chunk("chunk1", &head)];
        let result = check_staleness(&chunks, dir.path()).unwrap();

        assert_eq!(result.get("chunk1"), Some(&false));
    }

    #[test]
    fn ancestor_commit_is_not_stale() {
        let dir = TempDir::new().unwrap();
        let (first, _head) = init_repo_with_commits(dir.path());

        let chunks = vec![make_chunk("chunk1", &first)];
        let result = check_staleness(&chunks, dir.path()).unwrap();

        assert_eq!(result.get("chunk1"), Some(&false));
    }

    #[test]
    fn unknown_commit_is_stale() {
        let dir = TempDir::new().unwrap();
        let (_first, _head) = init_repo_with_commits(dir.path());

        let fake_oid = "0000000000000000000000000000000000000000";
        let chunks = vec![make_chunk("chunk1", fake_oid)];
        let result = check_staleness(&chunks, dir.path()).unwrap();

        assert_eq!(result.get("chunk1"), Some(&true));
    }

    #[test]
    fn empty_source_commit_is_stale() {
        let dir = TempDir::new().unwrap();
        let (_first, _head) = init_repo_with_commits(dir.path());

        let chunks = vec![make_chunk("chunk1", "")];
        let result = check_staleness(&chunks, dir.path()).unwrap();

        assert_eq!(result.get("chunk1"), Some(&true));
    }

    #[test]
    fn invalid_oid_string_is_stale() {
        let dir = TempDir::new().unwrap();
        let (_first, _head) = init_repo_with_commits(dir.path());

        let chunks = vec![make_chunk("chunk1", "not-a-valid-oid")];
        let result = check_staleness(&chunks, dir.path()).unwrap();

        assert_eq!(result.get("chunk1"), Some(&true));
    }

    #[test]
    fn mixed_staleness() {
        let dir = TempDir::new().unwrap();
        let (first, head) = init_repo_with_commits(dir.path());

        let chunks = vec![
            make_chunk("current", &head),
            make_chunk("ancestor", &first),
            make_chunk("unknown", "0000000000000000000000000000000000000000"),
            make_chunk("empty", ""),
        ];

        let result = check_staleness(&chunks, dir.path()).unwrap();

        assert_eq!(result.get("current"), Some(&false));
        assert_eq!(result.get("ancestor"), Some(&false));
        assert_eq!(result.get("unknown"), Some(&true));
        assert_eq!(result.get("empty"), Some(&true));
    }

    #[test]
    fn empty_chunks_returns_empty_map() {
        let dir = TempDir::new().unwrap();
        let (_first, _head) = init_repo_with_commits(dir.path());

        let result = check_staleness(&[], dir.path()).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn returns_entry_per_chunk() {
        let dir = TempDir::new().unwrap();
        let (_first, head) = init_repo_with_commits(dir.path());

        let chunks = vec![
            make_chunk("a", &head),
            make_chunk("b", &head),
            make_chunk("c", &head),
        ];

        let result = check_staleness(&chunks, dir.path()).unwrap();

        assert_eq!(result.len(), 3);
    }
}
