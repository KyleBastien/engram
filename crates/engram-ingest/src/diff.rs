use std::path::{Path, PathBuf};

use engram_core::{EngramError, SourceConfig};
use git2::{Oid, Repository};
use globset::{Glob, GlobSet, GlobSetBuilder};

/// Result of detecting changed files in a source repository.
#[derive(Debug, Clone, Default)]
pub struct ChangedFiles {
    pub added: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
}

/// Detect files that changed in a git repository since a given commit.
///
/// If `since_commit` is `None`, all tracked files are returned as `added` (full index).
/// Results are filtered by `include` and `exclude` glob patterns from the source config.
pub fn detect_changed_files(
    repo_path: &Path,
    since_commit: Option<&str>,
    source_config: &SourceConfig,
) -> engram_core::Result<ChangedFiles> {
    let repo = Repository::open(repo_path)
        .map_err(|e| EngramError::Git(format!("failed to open repo: {e}")))?;

    let include_set = build_glob_set(&source_config.include)?;
    let exclude_set = build_glob_set(&source_config.exclude)?;

    let has_include = !source_config.include.is_empty();
    let has_exclude = !source_config.exclude.is_empty();

    let filter = |path: &Path| -> bool {
        if has_include && !include_set.is_match(path) {
            return false;
        }
        if has_exclude && exclude_set.is_match(path) {
            return false;
        }
        true
    };

    match since_commit {
        None => full_index(&repo, filter),
        Some(commit_str) => diff_since(&repo, commit_str, filter),
    }
}

fn build_glob_set(patterns: &[String]) -> engram_core::Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = Glob::new(pattern)
            .map_err(|e| EngramError::Config(format!("invalid glob pattern '{pattern}': {e}")))?;
        builder.add(glob);
    }
    builder
        .build()
        .map_err(|e| EngramError::Config(format!("failed to build glob set: {e}")))
}

/// Full index: return all tracked files as "added".
fn full_index(
    repo: &Repository,
    filter: impl Fn(&Path) -> bool,
) -> engram_core::Result<ChangedFiles> {
    let head = repo
        .head()
        .map_err(|e| EngramError::Git(format!("failed to get HEAD: {e}")))?;
    let tree = head
        .peel_to_tree()
        .map_err(|e| EngramError::Git(format!("failed to peel HEAD to tree: {e}")))?;

    let mut added = Vec::new();
    tree.walk(git2::TreeWalkMode::PreOrder, |dir, entry| {
        if entry.kind() == Some(git2::ObjectType::Blob) {
            let path = PathBuf::from(format!(
                "{}{}",
                dir,
                entry.name().unwrap_or("")
            ));
            if filter(&path) {
                added.push(path);
            }
        }
        git2::TreeWalkResult::Ok
    })
    .map_err(|e| EngramError::Git(format!("failed to walk tree: {e}")))?;

    added.sort();

    Ok(ChangedFiles {
        added,
        ..Default::default()
    })
}

/// Diff since a specific commit: compare that commit's tree to HEAD's tree.
fn diff_since(
    repo: &Repository,
    since_commit: &str,
    filter: impl Fn(&Path) -> bool,
) -> engram_core::Result<ChangedFiles> {
    let old_oid = Oid::from_str(since_commit)
        .map_err(|e| EngramError::Git(format!("invalid commit hash '{since_commit}': {e}")))?;
    let old_commit = repo
        .find_commit(old_oid)
        .map_err(|e| EngramError::Git(format!("commit not found '{since_commit}': {e}")))?;
    let old_tree = old_commit
        .tree()
        .map_err(|e| EngramError::Git(format!("failed to get tree for old commit: {e}")))?;

    let head = repo
        .head()
        .map_err(|e| EngramError::Git(format!("failed to get HEAD: {e}")))?;
    let new_tree = head
        .peel_to_tree()
        .map_err(|e| EngramError::Git(format!("failed to peel HEAD to tree: {e}")))?;

    let diff = repo
        .diff_tree_to_tree(Some(&old_tree), Some(&new_tree), None)
        .map_err(|e| EngramError::Git(format!("failed to compute diff: {e}")))?;

    let mut added = Vec::new();
    let mut modified = Vec::new();
    let mut deleted = Vec::new();

    diff.foreach(
        &mut |delta, _progress| {
            let status = delta.status();
            // Use the new file path for added/modified, old file path for deleted
            let path = match status {
                git2::Delta::Deleted => delta
                    .old_file()
                    .path()
                    .map(PathBuf::from),
                _ => delta
                    .new_file()
                    .path()
                    .map(PathBuf::from),
            };

            if let Some(p) = path {
                if filter(&p) {
                    match status {
                        git2::Delta::Added => added.push(p),
                        git2::Delta::Modified => modified.push(p),
                        git2::Delta::Deleted => deleted.push(p),
                        git2::Delta::Renamed | git2::Delta::Copied => {
                            // Treat renamed/copied as added (new location)
                            added.push(p);
                            // Also mark old path as deleted if it passes filter
                            if let Some(old_path) = delta.old_file().path().map(PathBuf::from) {
                                if filter(&old_path) {
                                    deleted.push(old_path);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            true
        },
        None,
        None,
        None,
    )
    .map_err(|e| EngramError::Git(format!("failed to iterate diff: {e}")))?;

    added.sort();
    modified.sort();
    deleted.sort();

    Ok(ChangedFiles {
        added,
        modified,
        deleted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Helper: create a git repo with an initial commit containing the given files.
    fn setup_repo(files: &[(&str, &str)]) -> (TempDir, Repository) {
        let dir = TempDir::new().unwrap();
        let repo = Repository::init(dir.path()).unwrap();

        for (name, content) in files {
            let file_path = dir.path().join(name);
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&file_path, content).unwrap();
        }

        // Stage and commit (scope borrows before moving repo)
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

        (dir, repo)
    }

    /// Helper: make a commit on the repo with current index state.
    fn make_commit(repo: &Repository, message: &str) -> Oid {
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
            .unwrap()
    }

    fn empty_source() -> SourceConfig {
        SourceConfig {
            name: "test".to_string(),
            path: "/tmp".to_string(),
            include: vec![],
            exclude: vec![],
        }
    }

    #[test]
    fn test_full_index_returns_all_tracked_files_as_added() {
        let (dir, _repo) = setup_repo(&[
            ("src/main.rs", "fn main() {}"),
            ("src/lib.rs", "pub mod foo;"),
            ("README.md", "# Hello"),
        ]);

        let result = detect_changed_files(dir.path(), None, &empty_source()).unwrap();

        assert_eq!(result.added.len(), 3);
        assert!(result.added.contains(&PathBuf::from("README.md")));
        assert!(result.added.contains(&PathBuf::from("src/main.rs")));
        assert!(result.added.contains(&PathBuf::from("src/lib.rs")));
        assert!(result.modified.is_empty());
        assert!(result.deleted.is_empty());
    }

    #[test]
    fn test_diff_detects_added_files() {
        let (dir, repo) = setup_repo(&[("a.rs", "fn a() {}")]);
        let old_oid = repo.head().unwrap().peel_to_commit().unwrap().id();

        fs::write(dir.path().join("b.rs"), "fn b() {}").unwrap();
        make_commit(&repo, "add b.rs");

        let result = detect_changed_files(
            dir.path(),
            Some(&old_oid.to_string()),
            &empty_source(),
        )
        .unwrap();

        assert_eq!(result.added, vec![PathBuf::from("b.rs")]);
        assert!(result.modified.is_empty());
        assert!(result.deleted.is_empty());
    }

    #[test]
    fn test_diff_detects_modified_files() {
        let (dir, repo) = setup_repo(&[("a.rs", "fn a() {}")]);
        let old_oid = repo.head().unwrap().peel_to_commit().unwrap().id();

        fs::write(dir.path().join("a.rs"), "fn a_modified() {}").unwrap();
        make_commit(&repo, "modify a.rs");

        let result = detect_changed_files(
            dir.path(),
            Some(&old_oid.to_string()),
            &empty_source(),
        )
        .unwrap();

        assert!(result.added.is_empty());
        assert_eq!(result.modified, vec![PathBuf::from("a.rs")]);
        assert!(result.deleted.is_empty());
    }

    #[test]
    fn test_diff_detects_deleted_files() {
        let (dir, repo) = setup_repo(&[("a.rs", "fn a() {}"), ("b.rs", "fn b() {}")]);
        let old_oid = repo.head().unwrap().peel_to_commit().unwrap().id();

        fs::remove_file(dir.path().join("b.rs")).unwrap();
        // Stage the deletion
        let mut index = repo.index().unwrap();
        index.remove_path(Path::new("b.rs")).unwrap();
        index.write().unwrap();
        let tree_oid = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_oid).unwrap();
        let sig = git2::Signature::now("test", "test@test.com").unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "delete b.rs", &tree, &[&head])
            .unwrap();

        let result = detect_changed_files(
            dir.path(),
            Some(&old_oid.to_string()),
            &empty_source(),
        )
        .unwrap();

        assert!(result.added.is_empty());
        assert!(result.modified.is_empty());
        assert_eq!(result.deleted, vec![PathBuf::from("b.rs")]);
    }

    #[test]
    fn test_include_filter() {
        let (dir, _repo) = setup_repo(&[
            ("src/main.rs", "fn main() {}"),
            ("src/lib.rs", "pub mod foo;"),
            ("README.md", "# Hello"),
            ("docs/guide.md", "# Guide"),
        ]);

        let config = SourceConfig {
            name: "test".to_string(),
            path: "/tmp".to_string(),
            include: vec!["**/*.rs".to_string()],
            exclude: vec![],
        };

        let result = detect_changed_files(dir.path(), None, &config).unwrap();

        assert_eq!(result.added.len(), 2);
        assert!(result.added.contains(&PathBuf::from("src/main.rs")));
        assert!(result.added.contains(&PathBuf::from("src/lib.rs")));
    }

    #[test]
    fn test_exclude_filter() {
        let (dir, _repo) = setup_repo(&[
            ("src/main.rs", "fn main() {}"),
            ("target/debug/out.rs", "compiled"),
            ("node_modules/pkg/index.js", "module"),
        ]);

        let config = SourceConfig {
            name: "test".to_string(),
            path: "/tmp".to_string(),
            include: vec![],
            exclude: vec!["**/target/**".to_string(), "**/node_modules/**".to_string()],
        };

        let result = detect_changed_files(dir.path(), None, &config).unwrap();

        assert_eq!(result.added.len(), 1);
        assert_eq!(result.added[0], PathBuf::from("src/main.rs"));
    }

    #[test]
    fn test_include_and_exclude_combined() {
        let (dir, _repo) = setup_repo(&[
            ("src/main.rs", "fn main() {}"),
            ("src/generated/types.rs", "// auto"),
            ("README.md", "# Hello"),
        ]);

        let config = SourceConfig {
            name: "test".to_string(),
            path: "/tmp".to_string(),
            include: vec!["**/*.rs".to_string()],
            exclude: vec!["**/generated/**".to_string()],
        };

        let result = detect_changed_files(dir.path(), None, &config).unwrap();

        assert_eq!(result.added.len(), 1);
        assert_eq!(result.added[0], PathBuf::from("src/main.rs"));
    }

    #[test]
    fn test_diff_with_include_filter() {
        let (dir, repo) = setup_repo(&[("a.rs", "fn a() {}")]);
        let old_oid = repo.head().unwrap().peel_to_commit().unwrap().id();

        fs::write(dir.path().join("b.rs"), "fn b() {}").unwrap();
        fs::write(dir.path().join("c.txt"), "text file").unwrap();
        make_commit(&repo, "add b.rs and c.txt");

        let config = SourceConfig {
            name: "test".to_string(),
            path: "/tmp".to_string(),
            include: vec!["**/*.rs".to_string()],
            exclude: vec![],
        };

        let result = detect_changed_files(
            dir.path(),
            Some(&old_oid.to_string()),
            &config,
        )
        .unwrap();

        assert_eq!(result.added, vec![PathBuf::from("b.rs")]);
    }

    #[test]
    fn test_mixed_changes() {
        let (dir, repo) = setup_repo(&[
            ("a.rs", "fn a() {}"),
            ("b.rs", "fn b() {}"),
            ("c.rs", "fn c() {}"),
        ]);
        let old_oid = repo.head().unwrap().peel_to_commit().unwrap().id();

        // Modify a.rs
        fs::write(dir.path().join("a.rs"), "fn a_v2() {}").unwrap();
        // Add d.rs
        fs::write(dir.path().join("d.rs"), "fn d() {}").unwrap();
        // Delete b.rs
        fs::remove_file(dir.path().join("b.rs")).unwrap();

        let mut index = repo.index().unwrap();
        index.remove_path(Path::new("b.rs")).unwrap();
        index
            .add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
            .unwrap();
        index.write().unwrap();
        let tree_oid = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_oid).unwrap();
        let sig = git2::Signature::now("test", "test@test.com").unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "mixed changes", &tree, &[&head])
            .unwrap();

        let result = detect_changed_files(
            dir.path(),
            Some(&old_oid.to_string()),
            &empty_source(),
        )
        .unwrap();

        assert_eq!(result.added, vec![PathBuf::from("d.rs")]);
        assert_eq!(result.modified, vec![PathBuf::from("a.rs")]);
        assert_eq!(result.deleted, vec![PathBuf::from("b.rs")]);
    }

    #[test]
    fn test_no_changes_since_commit() {
        let (dir, repo) = setup_repo(&[("a.rs", "fn a() {}")]);
        let oid = repo.head().unwrap().peel_to_commit().unwrap().id();

        let result = detect_changed_files(
            dir.path(),
            Some(&oid.to_string()),
            &empty_source(),
        )
        .unwrap();

        assert!(result.added.is_empty());
        assert!(result.modified.is_empty());
        assert!(result.deleted.is_empty());
    }

    #[test]
    fn test_full_index_sorted() {
        let (dir, _repo) = setup_repo(&[
            ("z.rs", "z"),
            ("a.rs", "a"),
            ("m.rs", "m"),
        ]);

        let result = detect_changed_files(dir.path(), None, &empty_source()).unwrap();

        assert_eq!(
            result.added,
            vec![
                PathBuf::from("a.rs"),
                PathBuf::from("m.rs"),
                PathBuf::from("z.rs"),
            ]
        );
    }

    #[test]
    fn test_nested_directory_files() {
        let (dir, _repo) = setup_repo(&[
            ("src/a/b/c.rs", "deep"),
            ("src/main.rs", "main"),
        ]);

        let result = detect_changed_files(dir.path(), None, &empty_source()).unwrap();

        assert_eq!(result.added.len(), 2);
        assert!(result.added.contains(&PathBuf::from("src/a/b/c.rs")));
        assert!(result.added.contains(&PathBuf::from("src/main.rs")));
    }
}
