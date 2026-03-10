use std::fs;
use std::path::Path;

use engram_core::{Manifest, Result};

/// Write a manifest to `{store_root}/index/manifest.json`.
///
/// The JSON is pretty-printed for readability in git diffs.
pub fn write_manifest(store_root: &Path, manifest: &Manifest) -> Result<()> {
    let manifest_path = store_root.join("index").join("manifest.json");
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|e| engram_core::EngramError::Serialize(e.to_string()))?;
    fs::write(manifest_path, json)?;
    Ok(())
}

/// Read a manifest from `{store_root}/index/manifest.json`.
///
/// Returns `None` if the file does not exist.
pub fn read_manifest(store_root: &Path) -> Result<Option<Manifest>> {
    let manifest_path = store_root.join("index").join("manifest.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let json = fs::read_to_string(&manifest_path)?;
    let manifest: Manifest = serde_json::from_str(&json)
        .map_err(|e| engram_core::EngramError::Serialize(e.to_string()))?;
    Ok(Some(manifest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_manifest() -> Manifest {
        Manifest {
            chunk_count: 42,
            last_indexed_commit: Some("abc123def456".to_string()),
            model_name: "nomic-embed-text".to_string(),
            dimensions: 768,
            source_repos: vec![
                "/home/user/project-a".to_string(),
                "/home/user/project-b".to_string(),
            ],
            created_at: "2026-03-09T00:00:00Z".to_string(),
            updated_at: "2026-03-09T12:00:00Z".to_string(),
        }
    }

    #[test]
    fn round_trip() {
        let tmp = TempDir::new().unwrap();
        let store_root = tmp.path();
        fs::create_dir_all(store_root.join("index")).unwrap();

        let manifest = sample_manifest();
        write_manifest(store_root, &manifest).unwrap();
        let read_back = read_manifest(store_root).unwrap().expect("should exist");

        assert_eq!(manifest.chunk_count, read_back.chunk_count);
        assert_eq!(manifest.last_indexed_commit, read_back.last_indexed_commit);
        assert_eq!(manifest.model_name, read_back.model_name);
        assert_eq!(manifest.dimensions, read_back.dimensions);
        assert_eq!(manifest.source_repos, read_back.source_repos);
        assert_eq!(manifest.created_at, read_back.created_at);
        assert_eq!(manifest.updated_at, read_back.updated_at);
    }

    #[test]
    fn read_returns_none_when_missing() {
        let tmp = TempDir::new().unwrap();
        let result = read_manifest(tmp.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn json_is_pretty_printed() {
        let tmp = TempDir::new().unwrap();
        let store_root = tmp.path();
        fs::create_dir_all(store_root.join("index")).unwrap();

        write_manifest(store_root, &sample_manifest()).unwrap();

        let raw = fs::read_to_string(store_root.join("index/manifest.json")).unwrap();
        // Pretty-printed JSON contains newlines and indentation
        assert!(raw.contains('\n'));
        assert!(raw.contains("  "));
    }

    #[test]
    fn creates_index_directory_if_missing() {
        let tmp = TempDir::new().unwrap();
        let store_root = tmp.path().join("new-store");
        // Don't create the index dir — write_manifest should handle it
        write_manifest(&store_root, &sample_manifest()).unwrap();

        let read_back = read_manifest(&store_root).unwrap();
        assert!(read_back.is_some());
    }
}
