use std::path::Path;

use engram_core::{EngramError, Manifest, Result};
use git2::{
    AnnotatedCommit, AutotagOption, FetchOptions, Index, MergeOptions, Oid, Repository, Signature,
};
use serde::Serialize;

use crate::manifest::write_manifest;

/// Direction of a sync operation.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncDirection {
    Pull,
    Push,
}

/// Report returned by sync_pull / sync_push.
#[derive(Debug, Clone, Serialize)]
pub struct SyncReport {
    pub direction: SyncDirection,
    pub commits_transferred: usize,
    pub conflicts: Vec<String>,
}

/// Pull changes from the remote "origin" into the local store.
///
/// Uses fetch + merge. Conflicts on chunk files are resolved via last-write-wins
/// (accept theirs). Conflicts on manifest.json are resolved by merging fields
/// (higher chunk_count, more recent updated_at).
pub fn sync_pull(store_root: &Path) -> Result<SyncReport> {
    let repo =
        Repository::open(store_root).map_err(|e| EngramError::Git(e.to_string()))?;

    // Count commits before fetch so we can compute commits_transferred
    let local_head_before = head_oid(&repo);

    // Fetch from origin
    let branch = current_branch_name(&repo)?;
    fetch_origin(&repo, &branch)?;

    // Find the remote tracking ref
    let fetch_head = repo
        .find_reference("FETCH_HEAD")
        .map_err(|e| EngramError::Git(e.to_string()))?;
    let remote_commit = repo
        .reference_to_annotated_commit(&fetch_head)
        .map_err(|e| EngramError::Git(e.to_string()))?;

    // Analyze what kind of merge we need
    let (analysis, _) = repo
        .merge_analysis(&[&remote_commit])
        .map_err(|e| EngramError::Git(e.to_string()))?;

    let mut conflicts = Vec::new();

    if analysis.is_up_to_date() {
        // Nothing to do
        return Ok(SyncReport {
            direction: SyncDirection::Pull,
            commits_transferred: 0,
            conflicts,
        });
    }

    if analysis.is_fast_forward() {
        // Fast-forward: just move the branch ref
        fast_forward(&repo, &remote_commit, &branch)?;
    } else if analysis.is_normal() {
        // Normal merge with potential conflicts
        conflicts = do_merge(&repo, &remote_commit, store_root)?;
    } else {
        return Err(EngramError::Git(
            "Merge analysis returned unhandled state".to_string(),
        ));
    }

    // Count commits transferred
    let local_head_after = head_oid(&repo);
    let commits_transferred = count_commits_between(&repo, local_head_before, local_head_after);

    Ok(SyncReport {
        direction: SyncDirection::Pull,
        commits_transferred,
        conflicts,
    })
}

/// Push local commits to the remote "origin".
pub fn sync_push(store_root: &Path) -> Result<SyncReport> {
    let repo =
        Repository::open(store_root).map_err(|e| EngramError::Git(e.to_string()))?;

    let branch = current_branch_name(&repo)?;
    let refspec = format!("refs/heads/{branch}:refs/heads/{branch}");

    // Count local commits ahead of remote
    let commits_ahead = count_commits_ahead(&repo, &branch);

    let mut remote = repo
        .find_remote("origin")
        .map_err(|e| EngramError::Git(e.to_string()))?;
    remote
        .push(&[&refspec], None)
        .map_err(|e| EngramError::Git(e.to_string()))?;

    Ok(SyncReport {
        direction: SyncDirection::Push,
        commits_transferred: commits_ahead,
        conflicts: vec![],
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn head_oid(repo: &Repository) -> Option<Oid> {
    repo.head().ok().and_then(|h| h.target())
}

fn current_branch_name(repo: &Repository) -> Result<String> {
    let head = repo
        .head()
        .map_err(|e| EngramError::Git(e.to_string()))?;
    let name = head
        .shorthand()
        .ok_or_else(|| EngramError::Git("HEAD is not a valid UTF-8 ref".to_string()))?;
    Ok(name.to_string())
}

fn fetch_origin(repo: &Repository, branch: &str) -> Result<()> {
    let mut remote = repo
        .find_remote("origin")
        .map_err(|e| EngramError::Git(e.to_string()))?;
    let mut fo = FetchOptions::new();
    fo.download_tags(AutotagOption::None);
    remote
        .fetch(&[branch], Some(&mut fo), None)
        .map_err(|e| EngramError::Git(e.to_string()))?;
    Ok(())
}

fn fast_forward(repo: &Repository, remote_commit: &AnnotatedCommit, branch: &str) -> Result<()> {
    let refname = format!("refs/heads/{branch}");
    let mut reference = repo
        .find_reference(&refname)
        .map_err(|e| EngramError::Git(e.to_string()))?;
    reference
        .set_target(remote_commit.id(), "engram: fast-forward pull")
        .map_err(|e| EngramError::Git(e.to_string()))?;
    repo.set_head(&refname)
        .map_err(|e| EngramError::Git(e.to_string()))?;
    repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))
        .map_err(|e| EngramError::Git(e.to_string()))?;
    Ok(())
}

/// Perform a normal merge, resolving conflicts according to our strategy.
/// Returns the list of conflicted file paths that were resolved.
fn do_merge(
    repo: &Repository,
    remote_commit: &AnnotatedCommit,
    store_root: &Path,
) -> Result<Vec<String>> {
    let mut merge_opts = MergeOptions::new();
    let mut checkout = git2::build::CheckoutBuilder::default();
    // Allow conflicts to be written to working dir for us to resolve
    checkout.allow_conflicts(true);

    repo.merge(&[remote_commit], Some(&mut merge_opts), Some(&mut checkout))
        .map_err(|e| EngramError::Git(e.to_string()))?;

    let mut conflicts = Vec::new();

    // Check for conflicts and resolve them
    let index = repo
        .index()
        .map_err(|e| EngramError::Git(e.to_string()))?;

    if index.has_conflicts() {
        let conflict_paths = collect_conflict_paths(&index)?;
        for path in &conflict_paths {
            resolve_conflict(repo, store_root, path)?;
        }
        conflicts = conflict_paths;
    }

    // Stage everything and commit the merge
    let mut index = repo
        .index()
        .map_err(|e| EngramError::Git(e.to_string()))?;
    index
        .add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
        .map_err(|e| EngramError::Git(e.to_string()))?;
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

    let local_commit = repo
        .head()
        .map_err(|e| EngramError::Git(e.to_string()))?
        .peel_to_commit()
        .map_err(|e| EngramError::Git(e.to_string()))?;
    let remote_commit_obj = repo
        .find_commit(remote_commit.id())
        .map_err(|e| EngramError::Git(e.to_string()))?;

    repo.commit(
        Some("HEAD"),
        &sig,
        &sig,
        "engram: merge remote changes",
        &tree,
        &[&local_commit, &remote_commit_obj],
    )
    .map_err(|e| EngramError::Git(e.to_string()))?;

    // Clean up merge state
    repo.cleanup_state()
        .map_err(|e| EngramError::Git(e.to_string()))?;

    Ok(conflicts)
}

/// Collect paths that have conflicts in the index.
fn collect_conflict_paths(index: &Index) -> Result<Vec<String>> {
    let mut paths = Vec::new();
    let conflicts = index
        .conflicts()
        .map_err(|e| EngramError::Git(e.to_string()))?;
    for entry in conflicts {
        let entry = entry.map_err(|e| EngramError::Git(e.to_string()))?;
        // Use whichever entry is available to get the path
        let path = entry
            .their
            .as_ref()
            .or(entry.our.as_ref())
            .or(entry.ancestor.as_ref())
            .and_then(|e| String::from_utf8(e.path.clone()).ok());
        if let Some(p) = path {
            paths.push(p);
        }
    }
    Ok(paths)
}

/// Resolve a single conflicted file according to our strategy:
/// - manifest.json: merge fields (higher chunk_count, more recent updated_at)
/// - everything else (chunk files, etc.): last-write-wins (accept theirs)
fn resolve_conflict(repo: &Repository, store_root: &Path, path: &str) -> Result<()> {
    if path == "index/manifest.json" {
        resolve_manifest_conflict(repo, store_root)?;
    } else {
        // Last-write-wins: accept "theirs" (remote version)
        resolve_last_write_wins(repo, path)?;
    }
    Ok(())
}

/// Resolve manifest.json conflict by merging fields:
/// - chunk_count: take the higher value
/// - updated_at: take the more recent timestamp
/// - source_repos: union of both
/// - last_indexed_commits: merge maps (theirs wins per key)
fn resolve_manifest_conflict(repo: &Repository, store_root: &Path) -> Result<()> {
    let index = repo
        .index()
        .map_err(|e| EngramError::Git(e.to_string()))?;

    let mut our_manifest: Option<Manifest> = None;
    let mut their_manifest: Option<Manifest> = None;

    // Read both versions from the index conflict entries
    let conflicts = index
        .conflicts()
        .map_err(|e| EngramError::Git(e.to_string()))?;
    for entry in conflicts {
        let entry = entry.map_err(|e| EngramError::Git(e.to_string()))?;
        let entry_path = entry
            .our
            .as_ref()
            .or(entry.their.as_ref())
            .and_then(|e| String::from_utf8(e.path.clone()).ok());
        if entry_path.as_deref() != Some("index/manifest.json") {
            continue;
        }
        if let Some(ref our) = entry.our {
            let blob = repo
                .find_blob(our.id)
                .map_err(|e| EngramError::Git(e.to_string()))?;
            let content = std::str::from_utf8(blob.content())
                .map_err(|e| EngramError::Serialize(e.to_string()))?;
            our_manifest = serde_json::from_str(content).ok();
        }
        if let Some(ref their) = entry.their {
            let blob = repo
                .find_blob(their.id)
                .map_err(|e| EngramError::Git(e.to_string()))?;
            let content = std::str::from_utf8(blob.content())
                .map_err(|e| EngramError::Serialize(e.to_string()))?;
            their_manifest = serde_json::from_str(content).ok();
        }
    }

    let merged = match (our_manifest, their_manifest) {
        (Some(ours), Some(theirs)) => merge_manifests(&ours, &theirs),
        (Some(ours), None) => ours,
        (None, Some(theirs)) => theirs,
        (None, None) => {
            return Err(EngramError::Git(
                "Could not read either manifest version during conflict resolution".to_string(),
            ))
        }
    };

    // Write merged manifest to disk
    write_manifest(store_root, &merged)?;

    Ok(())
}

/// Merge two manifest versions: higher chunk_count, more recent updated_at,
/// union of source_repos, merge of last_indexed_commits.
fn merge_manifests(ours: &Manifest, theirs: &Manifest) -> Manifest {
    let chunk_count = ours.chunk_count.max(theirs.chunk_count);
    let updated_at = if theirs.updated_at > ours.updated_at {
        theirs.updated_at.clone()
    } else {
        ours.updated_at.clone()
    };

    // Merge last_indexed_commits: theirs wins per key
    let mut last_indexed_commits = ours.last_indexed_commits.clone();
    for (k, v) in &theirs.last_indexed_commits {
        last_indexed_commits.insert(k.clone(), v.clone());
    }

    // Union of source_repos (preserving order, dedup)
    let mut source_repos = ours.source_repos.clone();
    for repo in &theirs.source_repos {
        if !source_repos.contains(repo) {
            source_repos.push(repo.clone());
        }
    }

    Manifest {
        chunk_count,
        last_indexed_commits,
        model_name: theirs.model_name.clone(),
        dimensions: theirs.dimensions.max(ours.dimensions),
        source_repos,
        created_at: if ours.created_at < theirs.created_at {
            ours.created_at.clone()
        } else {
            theirs.created_at.clone()
        },
        updated_at,
    }
}

/// Resolve a conflict by accepting the remote ("theirs") version (last-write-wins).
fn resolve_last_write_wins(repo: &Repository, path: &str) -> Result<()> {
    let index = repo
        .index()
        .map_err(|e| EngramError::Git(e.to_string()))?;

    let conflicts = index
        .conflicts()
        .map_err(|e| EngramError::Git(e.to_string()))?;
    for entry in conflicts {
        let entry = entry.map_err(|e| EngramError::Git(e.to_string()))?;
        let entry_path = entry
            .their
            .as_ref()
            .or(entry.our.as_ref())
            .and_then(|e| String::from_utf8(e.path.clone()).ok());
        if entry_path.as_deref() != Some(path) {
            continue;
        }
        if let Some(ref their) = entry.their {
            // Write theirs to disk
            let blob = repo
                .find_blob(their.id)
                .map_err(|e| EngramError::Git(e.to_string()))?;
            let file_path = repo
                .workdir()
                .ok_or_else(|| EngramError::Git("No workdir".to_string()))?
                .join(path);
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file_path, blob.content())?;
        }
        break;
    }
    Ok(())
}

/// Count commits between two OIDs by walking from `after` back.
fn count_commits_between(repo: &Repository, before: Option<Oid>, after: Option<Oid>) -> usize {
    let Some(after_oid) = after else { return 0 };
    let Some(before_oid) = before else {
        // No previous HEAD; count all commits up to after
        let Ok(mut walk) = repo.revwalk() else {
            return 0;
        };
        let _ = walk.push(after_oid);
        return walk.count();
    };
    if before_oid == after_oid {
        return 0;
    }
    let Ok(mut walk) = repo.revwalk() else {
        return 0;
    };
    let _ = walk.push(after_oid);
    let _ = walk.hide(before_oid);
    walk.count()
}

/// Count commits on the local branch that are ahead of the remote tracking branch.
fn count_commits_ahead(repo: &Repository, branch: &str) -> usize {
    let remote_ref = format!("refs/remotes/origin/{branch}");
    let local_oid = match repo.head().ok().and_then(|h| h.target()) {
        Some(oid) => oid,
        None => return 0,
    };
    let remote_oid = match repo
        .find_reference(&remote_ref)
        .ok()
        .and_then(|r| r.target())
    {
        Some(oid) => oid,
        None => {
            // No remote tracking yet; count all local commits
            let Ok(mut walk) = repo.revwalk() else {
                return 0;
            };
            let _ = walk.push(local_oid);
            return walk.count();
        }
    };
    if local_oid == remote_oid {
        return 0;
    }
    let Ok(mut walk) = repo.revwalk() else {
        return 0;
    };
    let _ = walk.push(local_oid);
    let _ = walk.hide(remote_oid);
    walk.count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::read_manifest;
    use engram_core::StoreConfig;
    use std::fs;
    use tempfile::TempDir;

    /// Create a bare repo acting as the remote.
    fn create_bare_remote(tmp: &TempDir) -> PathBuf {
        let bare_path = tmp.path().join("remote.git");
        Repository::init_bare(&bare_path).unwrap();
        bare_path
    }

    use std::path::PathBuf;

    use crate::Store;

    /// Create a local store connected to a bare remote.
    fn create_synced_store(tmp: &TempDir, name: &str) -> (PathBuf, PathBuf) {
        let bare_path = create_bare_remote(tmp);
        let local_path = tmp.path().join(name);
        let url = bare_path.to_str().unwrap();
        Store::init_remote(url, &local_path, &StoreConfig::default()).unwrap();
        (bare_path, local_path)
    }

    /// Clone from the bare remote into a second local path.
    fn clone_store(bare_path: &Path, tmp: &TempDir, name: &str) -> PathBuf {
        let local2 = tmp.path().join(name);
        let url = bare_path.to_str().unwrap();
        Store::init_remote(url, &local2, &StoreConfig::default()).unwrap();
        local2
    }

    #[test]
    fn test_sync_pull_no_remote_changes() {
        let tmp = TempDir::new().unwrap();
        let (bare_path, local1) = create_synced_store(&tmp, "local1");
        let local2 = clone_store(&bare_path, &tmp, "local2");

        // Pull with nothing new
        let report = sync_pull(&local2).unwrap();
        assert!(matches!(report.direction, SyncDirection::Pull));
        assert_eq!(report.commits_transferred, 0);
        assert!(report.conflicts.is_empty());

        // Also make sure local1 is clean
        let report = sync_pull(&local1).unwrap();
        assert_eq!(report.commits_transferred, 0);
    }

    #[test]
    fn test_sync_push_transfers_commits() {
        let tmp = TempDir::new().unwrap();
        let (_bare_path, local1) = create_synced_store(&tmp, "local1");

        // Make a local change and commit
        fs::write(local1.join("test.txt"), "hello").unwrap();
        let repo = Repository::open(&local1).unwrap();
        crate::commit_changes(&repo, "add test file").unwrap();

        let report = sync_push(&local1).unwrap();
        assert!(matches!(report.direction, SyncDirection::Push));
        assert_eq!(report.commits_transferred, 1);
        assert!(report.conflicts.is_empty());
    }

    #[test]
    fn test_sync_pull_fast_forward() {
        let tmp = TempDir::new().unwrap();
        let (bare_path, local1) = create_synced_store(&tmp, "local1");
        let local2 = clone_store(&bare_path, &tmp, "local2");

        // Make a change in local1 and push
        fs::write(local1.join("new-file.txt"), "content").unwrap();
        let repo1 = Repository::open(&local1).unwrap();
        crate::commit_changes(&repo1, "add new file").unwrap();
        sync_push(&local1).unwrap();

        // Pull into local2
        let report = sync_pull(&local2).unwrap();
        assert!(matches!(report.direction, SyncDirection::Pull));
        assert_eq!(report.commits_transferred, 1);
        assert!(report.conflicts.is_empty());

        // Verify the file appeared
        assert!(local2.join("new-file.txt").exists());
        let content = fs::read_to_string(local2.join("new-file.txt")).unwrap();
        assert_eq!(content, "content");
    }

    #[test]
    fn test_sync_pull_merge_no_conflict() {
        let tmp = TempDir::new().unwrap();
        let (bare_path, local1) = create_synced_store(&tmp, "local1");
        let local2 = clone_store(&bare_path, &tmp, "local2");

        // Change different files in each clone
        fs::write(local1.join("file-a.txt"), "from local1").unwrap();
        let repo1 = Repository::open(&local1).unwrap();
        crate::commit_changes(&repo1, "add file-a").unwrap();
        sync_push(&local1).unwrap();

        fs::write(local2.join("file-b.txt"), "from local2").unwrap();
        let repo2 = Repository::open(&local2).unwrap();
        crate::commit_changes(&repo2, "add file-b").unwrap();

        // Pull should merge cleanly
        let report = sync_pull(&local2).unwrap();
        assert!(matches!(report.direction, SyncDirection::Pull));
        assert!(report.conflicts.is_empty());

        // Both files should exist
        assert!(local2.join("file-a.txt").exists());
        assert!(local2.join("file-b.txt").exists());
    }

    #[test]
    fn test_sync_pull_conflict_last_write_wins() {
        let tmp = TempDir::new().unwrap();
        let (bare_path, local1) = create_synced_store(&tmp, "local1");
        let local2 = clone_store(&bare_path, &tmp, "local2");

        // Both modify the same chunk file
        let chunk_path = "index/chunks.jsonl";
        fs::create_dir_all(local1.join("index")).unwrap();
        fs::write(local1.join(chunk_path), "local1-version\n").unwrap();
        let repo1 = Repository::open(&local1).unwrap();
        crate::commit_changes(&repo1, "local1 chunk").unwrap();
        sync_push(&local1).unwrap();

        fs::create_dir_all(local2.join("index")).unwrap();
        fs::write(local2.join(chunk_path), "local2-version\n").unwrap();
        let repo2 = Repository::open(&local2).unwrap();
        crate::commit_changes(&repo2, "local2 chunk").unwrap();

        // Pull — should resolve conflict via last-write-wins (theirs = remote = local1)
        let report = sync_pull(&local2).unwrap();
        assert!(matches!(report.direction, SyncDirection::Pull));
        assert!(!report.conflicts.is_empty());
        assert!(report.conflicts.contains(&chunk_path.to_string()));

        // The remote version wins
        let content = fs::read_to_string(local2.join(chunk_path)).unwrap();
        assert_eq!(content, "local1-version\n");
    }

    #[test]
    fn test_sync_pull_manifest_conflict_merges() {
        let tmp = TempDir::new().unwrap();
        let (bare_path, local1) = create_synced_store(&tmp, "local1");
        let local2 = clone_store(&bare_path, &tmp, "local2");

        // local1: write manifest with high chunk count
        let manifest1 = Manifest {
            chunk_count: 100,
            last_indexed_commits: [("repo-a".to_string(), "aaa111".to_string())]
                .into_iter()
                .collect(),
            model_name: "nomic-embed-text".to_string(),
            dimensions: 768,
            source_repos: vec!["repo-a".to_string()],
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-03-10T12:00:00Z".to_string(),
        };
        write_manifest(&local1, &manifest1).unwrap();
        let repo1 = Repository::open(&local1).unwrap();
        crate::commit_changes(&repo1, "local1 manifest").unwrap();
        sync_push(&local1).unwrap();

        // local2: write manifest with different data
        let manifest2 = Manifest {
            chunk_count: 50,
            last_indexed_commits: [("repo-b".to_string(), "bbb222".to_string())]
                .into_iter()
                .collect(),
            model_name: "nomic-embed-text".to_string(),
            dimensions: 768,
            source_repos: vec!["repo-b".to_string()],
            created_at: "2026-02-01T00:00:00Z".to_string(),
            updated_at: "2026-03-10T14:00:00Z".to_string(),
        };
        write_manifest(&local2, &manifest2).unwrap();
        let repo2 = Repository::open(&local2).unwrap();
        crate::commit_changes(&repo2, "local2 manifest").unwrap();

        // Pull — should merge manifests
        let report = sync_pull(&local2).unwrap();
        assert!(report
            .conflicts
            .contains(&"index/manifest.json".to_string()));

        // Read merged manifest
        let merged = read_manifest(&local2).unwrap().unwrap();
        // Higher chunk_count wins
        assert_eq!(merged.chunk_count, 100);
        // More recent updated_at wins
        assert_eq!(merged.updated_at, "2026-03-10T14:00:00Z");
        // Both repos in source_repos
        assert!(merged.source_repos.contains(&"repo-a".to_string()));
        assert!(merged.source_repos.contains(&"repo-b".to_string()));
        // Both commits in last_indexed_commits
        assert_eq!(
            merged.last_indexed_commits.get("repo-a").unwrap(),
            "aaa111"
        );
        assert_eq!(
            merged.last_indexed_commits.get("repo-b").unwrap(),
            "bbb222"
        );
    }

    #[test]
    fn test_sync_push_no_changes() {
        let tmp = TempDir::new().unwrap();
        let (_bare_path, local1) = create_synced_store(&tmp, "local1");

        // Push with nothing new (initial commit already pushed by init_remote)
        let report = sync_push(&local1).unwrap();
        assert!(matches!(report.direction, SyncDirection::Push));
        assert_eq!(report.commits_transferred, 0);
    }

    #[test]
    fn test_sync_push_multiple_commits() {
        let tmp = TempDir::new().unwrap();
        let (_bare_path, local1) = create_synced_store(&tmp, "local1");

        // Make two local commits
        fs::write(local1.join("a.txt"), "aaa").unwrap();
        let repo = Repository::open(&local1).unwrap();
        crate::commit_changes(&repo, "commit 1").unwrap();

        fs::write(local1.join("b.txt"), "bbb").unwrap();
        crate::commit_changes(&repo, "commit 2").unwrap();

        let report = sync_push(&local1).unwrap();
        assert_eq!(report.commits_transferred, 2);
    }

    #[test]
    fn test_sync_pull_no_remote_errors() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("no-remote");
        Store::init_local(&store_path, &StoreConfig::default()).unwrap();

        let result = sync_pull(&store_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_sync_push_no_remote_errors() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("no-remote");
        Store::init_local(&store_path, &StoreConfig::default()).unwrap();

        let result = sync_push(&store_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_manifests_higher_chunk_count_wins() {
        let m1 = Manifest {
            chunk_count: 200,
            last_indexed_commits: Default::default(),
            model_name: "m1".to_string(),
            dimensions: 768,
            source_repos: vec![],
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let m2 = Manifest {
            chunk_count: 100,
            last_indexed_commits: Default::default(),
            model_name: "m2".to_string(),
            dimensions: 768,
            source_repos: vec![],
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let merged = merge_manifests(&m1, &m2);
        assert_eq!(merged.chunk_count, 200);
    }

    #[test]
    fn test_merge_manifests_recent_updated_at_wins() {
        let m1 = Manifest {
            chunk_count: 10,
            last_indexed_commits: Default::default(),
            model_name: "m".to_string(),
            dimensions: 768,
            source_repos: vec![],
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-03-01T00:00:00Z".to_string(),
        };
        let m2 = Manifest {
            chunk_count: 10,
            last_indexed_commits: Default::default(),
            model_name: "m".to_string(),
            dimensions: 768,
            source_repos: vec![],
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-03-10T00:00:00Z".to_string(),
        };
        let merged = merge_manifests(&m1, &m2);
        assert_eq!(merged.updated_at, "2026-03-10T00:00:00Z");
    }

    #[test]
    fn test_merge_manifests_unions_source_repos() {
        let m1 = Manifest {
            chunk_count: 10,
            last_indexed_commits: Default::default(),
            model_name: "m".to_string(),
            dimensions: 768,
            source_repos: vec!["a".to_string(), "b".to_string()],
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let m2 = Manifest {
            chunk_count: 10,
            last_indexed_commits: Default::default(),
            model_name: "m".to_string(),
            dimensions: 768,
            source_repos: vec!["b".to_string(), "c".to_string()],
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let merged = merge_manifests(&m1, &m2);
        assert_eq!(merged.source_repos, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_merge_manifests_merges_indexed_commits() {
        let m1 = Manifest {
            chunk_count: 10,
            last_indexed_commits: [
                ("repo-a".to_string(), "aaa".to_string()),
                ("repo-shared".to_string(), "old".to_string()),
            ]
            .into_iter()
            .collect(),
            model_name: "m".to_string(),
            dimensions: 768,
            source_repos: vec![],
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let m2 = Manifest {
            chunk_count: 10,
            last_indexed_commits: [
                ("repo-b".to_string(), "bbb".to_string()),
                ("repo-shared".to_string(), "new".to_string()),
            ]
            .into_iter()
            .collect(),
            model_name: "m".to_string(),
            dimensions: 768,
            source_repos: vec![],
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let merged = merge_manifests(&m1, &m2);
        assert_eq!(merged.last_indexed_commits.get("repo-a").unwrap(), "aaa");
        assert_eq!(merged.last_indexed_commits.get("repo-b").unwrap(), "bbb");
        // Theirs wins on shared key
        assert_eq!(
            merged.last_indexed_commits.get("repo-shared").unwrap(),
            "new"
        );
    }

    #[test]
    fn test_sync_report_serializes() {
        let report = SyncReport {
            direction: SyncDirection::Pull,
            commits_transferred: 3,
            conflicts: vec!["index/chunks.jsonl".to_string()],
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"pull\""));
        assert!(json.contains("\"commits_transferred\":3"));
        assert!(json.contains("index/chunks.jsonl"));
    }
}
