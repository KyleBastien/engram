use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use engram_core::{EngramError, Result, Snapshot, SnapshotTier};

use crate::embeddings::write_embeddings_bin;

/// Map a [`SnapshotTier`] to its directory name under `snapshots/`.
fn tier_dir_name(tier: &SnapshotTier) -> &'static str {
    match tier {
        SnapshotTier::Active => "active",
        SnapshotTier::Compressed => "compressed",
        SnapshotTier::Archived => "archived",
    }
}

/// Make a timestamp string safe for use in filenames.
///
/// Replaces colons with hyphens (e.g. `2026-03-09T12:30:00Z` → `2026-03-09T12-30-00Z`).
fn sanitize_timestamp(ts: &str) -> String {
    ts.replace(':', "-")
}

/// Write a snapshot to `snapshots/active/{timestamp}_{session_id}.yaml`.
///
/// The snapshot is always written to the `active` tier regardless of the
/// `tier` field value (newly written snapshots start as active).
pub fn write_snapshot(store_root: &Path, snapshot: &Snapshot) -> Result<PathBuf> {
    let ts = sanitize_timestamp(&snapshot.created_at);
    let filename = format!("{}_{}.yaml", ts, snapshot.session_id);
    let path = store_root.join("snapshots/active").join(&filename);

    fs::create_dir_all(path.parent().unwrap())?;
    let yaml =
        serde_yaml::to_string(snapshot).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(&path, yaml)?;

    Ok(path)
}

/// Read a snapshot by session_id, searching all tiers.
///
/// Searches `active`, `compressed`, and `archived` tiers in order.
/// Compressed snapshots (`.yaml.zst`) are transparently decompressed.
pub fn read_snapshot(store_root: &Path, session_id: &str) -> Result<Snapshot> {
    let tiers = [
        SnapshotTier::Active,
        SnapshotTier::Compressed,
        SnapshotTier::Archived,
    ];

    for tier in &tiers {
        let dir = store_root.join("snapshots").join(tier_dir_name(tier));
        if !dir.is_dir() {
            continue;
        }

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let fname = path.file_name().unwrap_or_default().to_string_lossy();

            // Match by session_id in the filename
            if !fname.contains(session_id) {
                continue;
            }

            // Handle .yaml.zst (compressed)
            if fname.ends_with(".yaml.zst") {
                let compressed = fs::read(&path)?;
                let mut decoder = zstd::Decoder::new(&compressed[..])
                    .map_err(|e| EngramError::Store(format!("zstd decode error: {e}")))?;
                let mut yaml = String::new();
                decoder
                    .read_to_string(&mut yaml)
                    .map_err(|e| EngramError::Store(format!("zstd read error: {e}")))?;
                let snapshot: Snapshot = serde_yaml::from_str(&yaml)
                    .map_err(|e| EngramError::Serialize(e.to_string()))?;
                return Ok(snapshot);
            }

            // Handle plain .yaml
            if fname.ends_with(".yaml") {
                let yaml = fs::read_to_string(&path)?;
                let snapshot: Snapshot = serde_yaml::from_str(&yaml)
                    .map_err(|e| EngramError::Serialize(e.to_string()))?;
                return Ok(snapshot);
            }
        }
    }

    Err(EngramError::Store(format!(
        "snapshot not found for session: {session_id}"
    )))
}

/// Compute the `.embedding.bin` path for a snapshot YAML file.
pub fn snapshot_embedding_path(yaml_path: &Path) -> PathBuf {
    let stem = yaml_path.file_stem().unwrap().to_string_lossy();
    yaml_path.with_file_name(format!("{stem}.embedding.bin"))
}

/// Write an embedding binary alongside a snapshot YAML file.
pub fn write_snapshot_embedding(
    yaml_path: &Path,
    embedding: &[f32],
    dimensions: usize,
) -> Result<PathBuf> {
    let emb_path = snapshot_embedding_path(yaml_path);
    write_embeddings_bin(&emb_path, &[embedding.to_vec()], dimensions)?;
    Ok(emb_path)
}

/// Extract embeddable text from a Snapshot (summary + key_context).
pub fn snapshot_embed_text(snapshot: &Snapshot) -> String {
    let mut text = snapshot.summary.clone();
    for ctx in &snapshot.key_context {
        text.push('\n');
        text.push_str(ctx);
    }
    text
}

/// Write a snapshot with its embedding. Sets `embedding_ref` in the YAML.
pub fn write_snapshot_with_embedding(
    store_root: &Path,
    snapshot: &Snapshot,
    embedding: &[f32],
    dimensions: usize,
) -> Result<PathBuf> {
    let yaml_path = write_snapshot(store_root, snapshot)?;
    let emb_path = write_snapshot_embedding(&yaml_path, embedding, dimensions)?;

    let rel = emb_path
        .strip_prefix(store_root)
        .unwrap_or(&emb_path)
        .to_string_lossy()
        .to_string();

    let mut updated = snapshot.clone();
    updated.embedding_ref = Some(rel);
    let yaml =
        serde_yaml::to_string(&updated).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(&yaml_path, yaml)?;

    Ok(yaml_path)
}

/// Metadata about a snapshot that has an embedding, used during boot indexing
/// into HNSW partition 3.
pub struct SnapshotEmbeddingInfo {
    /// Session identifier.
    pub session_id: String,
    /// Snapshot summary.
    pub summary: String,
    /// Path to the embedding binary file.
    pub embedding_path: PathBuf,
    /// Timestamp from the snapshot.
    pub created_at: String,
}

/// Scan all snapshot tiers for items that have embeddings.
///
/// Returns metadata for each snapshot that has a corresponding
/// `.embedding.bin` file alongside its YAML.
pub fn scan_snapshot_embeddings(store_root: &Path) -> Result<Vec<SnapshotEmbeddingInfo>> {
    let mut results = Vec::new();

    let tiers = [
        SnapshotTier::Active,
        SnapshotTier::Compressed,
        SnapshotTier::Archived,
    ];

    for tier in &tiers {
        let dir = store_root.join("snapshots").join(tier_dir_name(tier));
        if !dir.is_dir() {
            continue;
        }

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml") {
                let emb_path = snapshot_embedding_path(&path);
                if emb_path.exists() {
                    let yaml = fs::read_to_string(&path)?;
                    let snapshot: Snapshot = serde_yaml::from_str(&yaml)
                        .map_err(|e| EngramError::Serialize(e.to_string()))?;
                    results.push(SnapshotEmbeddingInfo {
                        session_id: snapshot.session_id,
                        summary: snapshot.summary,
                        embedding_path: emb_path,
                        created_at: snapshot.created_at,
                    });
                }
            }
        }
    }

    // Sort by session_id for deterministic ordering
    results.sort_by(|a, b| a.session_id.cmp(&b.session_id));
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_snapshot_dirs(root: &Path) {
        fs::create_dir_all(root.join("snapshots/active")).unwrap();
        fs::create_dir_all(root.join("snapshots/compressed")).unwrap();
        fs::create_dir_all(root.join("snapshots/archived")).unwrap();
    }

    fn sample_snapshot() -> Snapshot {
        Snapshot {
            session_id: "session-001".to_string(),
            summary: "Implemented snapshot storage for engram".to_string(),
            key_context: vec![
                "Added write_snapshot function".to_string(),
                "Added read_snapshot with tier search".to_string(),
            ],
            full_transcript: "User: Add snapshots\nAssistant: Done.".to_string(),
            created_at: "2026-03-09T14:30:00Z".to_string(),
            tier: SnapshotTier::Active,
            embedding_ref: None,
        }
    }

    #[test]
    fn test_write_snapshot_creates_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        let snapshot = sample_snapshot();
        let path = write_snapshot(root, &snapshot).unwrap();

        assert_eq!(
            path,
            root.join("snapshots/active/2026-03-09T14-30-00Z_session-001.yaml")
        );
        assert!(path.exists());

        let content = fs::read_to_string(&path).unwrap();
        let deserialized: Snapshot = serde_yaml::from_str(&content).unwrap();
        assert_eq!(deserialized, snapshot);
    }

    #[test]
    fn test_write_snapshot_creates_parent_dirs() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("deep/nested/store");
        // Don't call make_snapshot_dirs — let write_snapshot create parents

        let snapshot = sample_snapshot();
        let path = write_snapshot(&root, &snapshot).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_read_snapshot_from_active() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        let snapshot = sample_snapshot();
        write_snapshot(root, &snapshot).unwrap();

        let loaded = read_snapshot(root, "session-001").unwrap();
        assert_eq!(loaded, snapshot);
    }

    #[test]
    fn test_read_snapshot_from_compressed_tier() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        // Write a snapshot YAML, then compress it into the compressed tier
        let snapshot = sample_snapshot();
        let yaml =
            serde_yaml::to_string(&snapshot).unwrap();
        let compressed_path = root.join(
            "snapshots/compressed/2026-03-09T14-30-00Z_session-001.yaml.zst",
        );
        let compressed = zstd::encode_all(yaml.as_bytes(), 3).unwrap();
        fs::write(&compressed_path, compressed).unwrap();

        let loaded = read_snapshot(root, "session-001").unwrap();
        assert_eq!(loaded, snapshot);
    }

    #[test]
    fn test_read_snapshot_from_archived_tier() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        // Write a plain YAML in archived tier
        let snapshot = sample_snapshot();
        let yaml = serde_yaml::to_string(&snapshot).unwrap();
        let archived_path =
            root.join("snapshots/archived/2026-03-09T14-30-00Z_session-001.yaml");
        fs::write(&archived_path, yaml).unwrap();

        let loaded = read_snapshot(root, "session-001").unwrap();
        assert_eq!(loaded, snapshot);
    }

    #[test]
    fn test_read_snapshot_not_found() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        let err = read_snapshot(root, "nonexistent").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not found"), "error was: {msg}");
    }

    #[test]
    fn test_read_snapshot_zstd_transparent_decompression() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        let snapshot = Snapshot {
            session_id: "session-zst".to_string(),
            summary: "Compressed snapshot test".to_string(),
            key_context: vec!["zstd decompression works".to_string()],
            full_transcript: "Full transcript content here".to_string(),
            created_at: "2026-03-09T10:00:00Z".to_string(),
            tier: SnapshotTier::Compressed,
            embedding_ref: None,
        };

        let yaml = serde_yaml::to_string(&snapshot).unwrap();
        let compressed = zstd::encode_all(yaml.as_bytes(), 3).unwrap();
        let path = root.join("snapshots/compressed/2026-03-09T10-00-00Z_session-zst.yaml.zst");
        fs::write(&path, compressed).unwrap();

        let loaded = read_snapshot(root, "session-zst").unwrap();
        assert_eq!(loaded.session_id, "session-zst");
        assert_eq!(loaded.summary, "Compressed snapshot test");
        assert_eq!(loaded.key_context, vec!["zstd decompression works"]);
    }

    #[test]
    fn test_snapshot_embed_text() {
        let snapshot = sample_snapshot();
        let text = snapshot_embed_text(&snapshot);
        assert!(text.contains(&snapshot.summary));
        for ctx in &snapshot.key_context {
            assert!(text.contains(ctx));
        }
    }

    #[test]
    fn test_snapshot_embedding_path() {
        let yaml = Path::new("/store/snapshots/active/2026-03-09T14-30-00Z_session-001.yaml");
        let emb = snapshot_embedding_path(yaml);
        assert_eq!(
            emb,
            PathBuf::from(
                "/store/snapshots/active/2026-03-09T14-30-00Z_session-001.embedding.bin"
            )
        );
    }

    #[test]
    fn test_write_snapshot_with_embedding() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        let snapshot = sample_snapshot();
        let embedding = vec![0.1f32, 0.2, 0.3];
        let yaml_path =
            write_snapshot_with_embedding(root, &snapshot, &embedding, 3).unwrap();

        // YAML should have embedding_ref set
        let content = fs::read_to_string(&yaml_path).unwrap();
        let loaded: Snapshot = serde_yaml::from_str(&content).unwrap();
        assert!(loaded.embedding_ref.is_some());
        let emb_ref = loaded.embedding_ref.unwrap();
        assert!(emb_ref.ends_with(".embedding.bin"));
        assert!(emb_ref.starts_with("snapshots/active/"));

        // Embedding binary should exist alongside YAML
        let emb_path = snapshot_embedding_path(&yaml_path);
        assert!(emb_path.exists());
    }

    #[test]
    fn test_write_snapshot_embedding_creates_binary() {
        let tmp = TempDir::new().unwrap();
        let yaml_path = tmp.path().join("test.yaml");
        fs::write(&yaml_path, "dummy").unwrap();

        let embedding = vec![1.0f32, 2.0, 3.0, 4.0];
        let emb_path = write_snapshot_embedding(&yaml_path, &embedding, 4).unwrap();

        assert_eq!(emb_path, tmp.path().join("test.embedding.bin"));
        assert!(emb_path.exists());

        let file = crate::read_embeddings_bin(&emb_path).unwrap();
        assert_eq!(file.count, 1);
        assert_eq!(file.dimensions, 4);
        assert_eq!(file.vectors, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scan_snapshot_embeddings_empty() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        let results = scan_snapshot_embeddings(root).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_scan_snapshot_embeddings_finds_items() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        // Write a snapshot with embedding
        let snapshot = sample_snapshot();
        write_snapshot_with_embedding(root, &snapshot, &[0.1, 0.2, 0.3], 3).unwrap();

        // Write a snapshot WITHOUT embedding (should not appear)
        let snapshot2 = Snapshot {
            session_id: "session-002".to_string(),
            summary: "No embedding".to_string(),
            key_context: vec![],
            full_transcript: String::new(),
            created_at: "2026-03-09T15:00:00Z".to_string(),
            tier: SnapshotTier::Active,
            embedding_ref: None,
        };
        write_snapshot(root, &snapshot2).unwrap();

        let results = scan_snapshot_embeddings(root).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, "session-001");
    }

    #[test]
    fn test_scan_snapshot_embeddings_no_dirs() {
        let tmp = TempDir::new().unwrap();
        let results = scan_snapshot_embeddings(tmp.path()).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_sanitize_timestamp() {
        assert_eq!(
            sanitize_timestamp("2026-03-09T14:30:00Z"),
            "2026-03-09T14-30-00Z"
        );
        assert_eq!(sanitize_timestamp("2026-03-09"), "2026-03-09");
    }

    #[test]
    fn test_tier_dir_name() {
        assert_eq!(tier_dir_name(&SnapshotTier::Active), "active");
        assert_eq!(tier_dir_name(&SnapshotTier::Compressed), "compressed");
        assert_eq!(tier_dir_name(&SnapshotTier::Archived), "archived");
    }
}
