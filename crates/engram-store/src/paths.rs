use std::fs;
use std::path::{Component, Path, PathBuf};

use engram_core::Result;

/// Sanitize a path by removing any traversal components (`.`, `..`, prefix/root).
/// Only normal path components are kept.
fn sanitize(path: &Path) -> PathBuf {
    path.components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s),
            _ => None,
        })
        .collect()
}

/// Resolve the path for a `.chunks.jsonl` file within the store.
///
/// Returns `{store_root}/index/{source_name}/{relative_file_path}.chunks.jsonl`
/// and creates parent directories if they don't exist.
pub fn chunks_path(
    store_root: &Path,
    source_name: &str,
    relative_file_path: &Path,
) -> Result<PathBuf> {
    let safe_source = sanitize(Path::new(source_name));
    let safe_rel = sanitize(relative_file_path);

    let mut file_name = safe_rel
        .file_name()
        .unwrap_or_default()
        .to_os_string();
    file_name.push(".chunks.jsonl");

    let dir = store_root
        .join("index")
        .join(&safe_source)
        .join(safe_rel.parent().unwrap_or(Path::new("")));

    fs::create_dir_all(&dir)?;

    Ok(dir.join(file_name))
}

/// Resolve the path for a `.embeddings.bin` file within the store.
///
/// Returns `{store_root}/index/{source_name}/{relative_file_path}.embeddings.bin`
/// and creates parent directories if they don't exist.
pub fn embeddings_path(
    store_root: &Path,
    source_name: &str,
    relative_file_path: &Path,
) -> Result<PathBuf> {
    let safe_source = sanitize(Path::new(source_name));
    let safe_rel = sanitize(relative_file_path);

    let mut file_name = safe_rel
        .file_name()
        .unwrap_or_default()
        .to_os_string();
    file_name.push(".embeddings.bin");

    let dir = store_root
        .join("index")
        .join(&safe_source)
        .join(safe_rel.parent().unwrap_or(Path::new("")));

    fs::create_dir_all(&dir)?;

    Ok(dir.join(file_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_chunks_path_simple() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        let path = chunks_path(root, "my-repo", Path::new("src/main.rs")).unwrap();

        assert_eq!(
            path,
            root.join("index/my-repo/src/main.rs.chunks.jsonl")
        );
        assert!(path.parent().unwrap().is_dir());
    }

    #[test]
    fn test_embeddings_path_simple() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        let path = embeddings_path(root, "my-repo", Path::new("src/main.rs")).unwrap();

        assert_eq!(
            path,
            root.join("index/my-repo/src/main.rs.embeddings.bin")
        );
        assert!(path.parent().unwrap().is_dir());
    }

    #[test]
    fn test_nested_paths() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        let path =
            chunks_path(root, "my-repo", Path::new("src/utils/helpers/math.ts")).unwrap();

        assert_eq!(
            path,
            root.join("index/my-repo/src/utils/helpers/math.ts.chunks.jsonl")
        );
        assert!(path.parent().unwrap().is_dir());
    }

    #[test]
    fn test_parent_directories_created() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Directories don't exist yet
        assert!(!root.join("index").exists());

        chunks_path(root, "repo", Path::new("deep/nested/file.rs")).unwrap();

        assert!(root.join("index/repo/deep/nested").is_dir());
    }

    #[test]
    fn test_path_traversal_prevented() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Relative file path with traversal
        let path =
            chunks_path(root, "repo", Path::new("../../etc/passwd")).unwrap();

        // The .. components should be stripped
        assert_eq!(
            path,
            root.join("index/repo/etc/passwd.chunks.jsonl")
        );
        // Must stay inside store_root/index/
        assert!(path.starts_with(root.join("index")));
    }

    #[test]
    fn test_source_name_traversal_prevented() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        let path =
            chunks_path(root, "../evil", Path::new("file.rs")).unwrap();

        // The .. in source name should be stripped
        assert_eq!(
            path,
            root.join("index/evil/file.rs.chunks.jsonl")
        );
        assert!(path.starts_with(root.join("index")));
    }

    #[test]
    fn test_top_level_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        let path = chunks_path(root, "repo", Path::new("README.md")).unwrap();

        assert_eq!(
            path,
            root.join("index/repo/README.md.chunks.jsonl")
        );
    }

    #[test]
    fn test_embeddings_nested() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        let path = embeddings_path(
            root,
            "my-project",
            Path::new("crates/core/src/lib.rs"),
        )
        .unwrap();

        assert_eq!(
            path,
            root.join("index/my-project/crates/core/src/lib.rs.embeddings.bin")
        );
        assert!(path.parent().unwrap().is_dir());
    }
}
