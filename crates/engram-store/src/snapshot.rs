use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

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
    write_embeddings_bin(&emb_path, &[embedding.to_vec()], dimensions, crate::PRECISION_F32)?;
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

/// Report returned by [`compact_snapshots`] with counts of promoted snapshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactReport {
    /// Number of active snapshots compressed and moved to `compressed/`.
    pub active_to_compressed: usize,
    /// Number of compressed snapshots moved to `archived/`.
    pub compressed_to_archived: usize,
}

/// Age threshold for promoting active → compressed (7 days).
const ACTIVE_MAX_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Age threshold for promoting compressed → archived (90 days).
const COMPRESSED_MAX_AGE: Duration = Duration::from_secs(90 * 24 * 60 * 60);

/// Determine a file's age by its filesystem modified time.
fn file_age(path: &Path) -> Result<Duration> {
    let metadata = fs::metadata(path)?;
    let modified = metadata
        .modified()
        .map_err(|e| EngramError::Store(format!("cannot read mtime: {e}")))?;
    Ok(SystemTime::now()
        .duration_since(modified)
        .unwrap_or(Duration::ZERO))
}

/// Compact snapshots by promoting them through tiers based on age.
///
/// - Snapshots in `active/` older than 7 days are zstd-compressed and moved to
///   `compressed/` (original deleted).
/// - Snapshots in `compressed/` older than 90 days are moved to `archived/`.
///
/// Returns a [`CompactReport`] with promotion counts.
pub fn compact_snapshots(store_root: &Path) -> Result<CompactReport> {
    let mut report = CompactReport {
        active_to_compressed: 0,
        compressed_to_archived: 0,
    };

    let active_dir = store_root.join("snapshots/active");
    let compressed_dir = store_root.join("snapshots/compressed");
    let archived_dir = store_root.join("snapshots/archived");

    // Ensure target directories exist
    fs::create_dir_all(&compressed_dir)?;
    fs::create_dir_all(&archived_dir)?;

    // Phase 1: active → compressed (zstd compress .yaml files older than 7 days)
    if active_dir.is_dir() {
        for entry in fs::read_dir(&active_dir)? {
            let entry = entry?;
            let path = entry.path();
            let fname = path.file_name().unwrap_or_default().to_string_lossy();

            if !fname.ends_with(".yaml") {
                continue;
            }

            if file_age(&path)? >= ACTIVE_MAX_AGE {
                let yaml_bytes = fs::read(&path)?;
                let compressed = zstd::encode_all(&yaml_bytes[..], 3)
                    .map_err(|e| EngramError::Store(format!("zstd compress error: {e}")))?;

                let dest = compressed_dir.join(format!("{fname}.zst"));
                fs::write(&dest, compressed)?;

                // Move associated embedding file if present
                let emb_path = snapshot_embedding_path(&path);
                if emb_path.exists() {
                    let emb_dest = compressed_dir.join(
                        emb_path.file_name().unwrap(),
                    );
                    fs::rename(&emb_path, &emb_dest)?;
                }

                fs::remove_file(&path)?;
                report.active_to_compressed += 1;
            }
        }
    }

    // Phase 2: compressed → archived (move files older than 90 days)
    if compressed_dir.is_dir() {
        for entry in fs::read_dir(&compressed_dir)? {
            let entry = entry?;
            let path = entry.path();
            let fname = path.file_name().unwrap_or_default().to_string_lossy();

            // Only process .yaml.zst snapshot files (not embedding bins)
            if !fname.ends_with(".yaml.zst") {
                continue;
            }

            if file_age(&path)? >= COMPRESSED_MAX_AGE {
                let dest = archived_dir.join(fname.as_ref());
                fs::rename(&path, &dest)?;

                // Move associated embedding file if present
                // Embedding files use the original yaml stem, e.g. foo.embedding.bin
                let stem = fname.trim_end_matches(".yaml.zst");
                let emb_name = format!("{stem}.embedding.bin");
                let emb_path = compressed_dir.join(&emb_name);
                if emb_path.exists() {
                    let emb_dest = archived_dir.join(&emb_name);
                    fs::rename(&emb_path, &emb_dest)?;
                }

                report.compressed_to_archived += 1;
            }
        }
    }

    Ok(report)
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

    /// Helper: set a file's modified time to `age` ago from now.
    fn set_file_age(path: &Path, age: Duration) {
        use std::fs::{File, FileTimes};
        use std::time::SystemTime;
        let mtime = SystemTime::now() - age;
        let times = FileTimes::new().set_modified(mtime);
        let file = File::options().write(true).open(path).unwrap();
        file.set_times(times).unwrap();
    }

    #[test]
    fn test_compact_no_old_snapshots() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        // Write a fresh snapshot (age < 7 days)
        let snapshot = sample_snapshot();
        write_snapshot(root, &snapshot).unwrap();

        let report = compact_snapshots(root).unwrap();
        assert_eq!(report.active_to_compressed, 0);
        assert_eq!(report.compressed_to_archived, 0);

        // Original file still in active/
        assert!(root
            .join("snapshots/active/2026-03-09T14-30-00Z_session-001.yaml")
            .exists());
    }

    #[test]
    fn test_compact_active_to_compressed() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        let snapshot = sample_snapshot();
        let yaml_path = write_snapshot(root, &snapshot).unwrap();

        // Make the file 8 days old
        set_file_age(&yaml_path, Duration::from_secs(8 * 24 * 60 * 60));

        let report = compact_snapshots(root).unwrap();
        assert_eq!(report.active_to_compressed, 1);
        assert_eq!(report.compressed_to_archived, 0);

        // Original removed from active/
        assert!(!yaml_path.exists());

        // Compressed file exists in compressed/
        let compressed_path = root.join(
            "snapshots/compressed/2026-03-09T14-30-00Z_session-001.yaml.zst",
        );
        assert!(compressed_path.exists());

        // Verify the compressed content is valid
        let compressed = fs::read(&compressed_path).unwrap();
        let mut decoder = zstd::Decoder::new(&compressed[..]).unwrap();
        let mut yaml = String::new();
        decoder.read_to_string(&mut yaml).unwrap();
        let loaded: Snapshot = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(loaded.session_id, "session-001");
    }

    #[test]
    fn test_compact_compressed_to_archived() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        // Place a .yaml.zst file in compressed/ tier
        let snapshot = sample_snapshot();
        let yaml = serde_yaml::to_string(&snapshot).unwrap();
        let compressed_data = zstd::encode_all(yaml.as_bytes(), 3).unwrap();
        let compressed_path = root.join(
            "snapshots/compressed/2026-03-09T14-30-00Z_session-001.yaml.zst",
        );
        fs::write(&compressed_path, compressed_data).unwrap();

        // Make it 91 days old
        set_file_age(&compressed_path, Duration::from_secs(91 * 24 * 60 * 60));

        let report = compact_snapshots(root).unwrap();
        assert_eq!(report.active_to_compressed, 0);
        assert_eq!(report.compressed_to_archived, 1);

        // Moved from compressed/ to archived/
        assert!(!compressed_path.exists());
        assert!(root
            .join("snapshots/archived/2026-03-09T14-30-00Z_session-001.yaml.zst")
            .exists());
    }

    #[test]
    fn test_compact_moves_embedding_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        let snapshot = sample_snapshot();
        let embedding = vec![0.1f32, 0.2, 0.3];
        let yaml_path =
            write_snapshot_with_embedding(root, &snapshot, &embedding, 3).unwrap();
        let emb_path = snapshot_embedding_path(&yaml_path);

        // Make both files 8 days old
        set_file_age(&yaml_path, Duration::from_secs(8 * 24 * 60 * 60));
        set_file_age(&emb_path, Duration::from_secs(8 * 24 * 60 * 60));

        let report = compact_snapshots(root).unwrap();
        assert_eq!(report.active_to_compressed, 1);

        // Embedding moved to compressed/
        assert!(!emb_path.exists());
        assert!(root
            .join("snapshots/compressed/2026-03-09T14-30-00Z_session-001.embedding.bin")
            .exists());
    }

    #[test]
    fn test_compact_empty_store() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        // No snapshot dirs at all
        let report = compact_snapshots(root).unwrap();
        assert_eq!(report.active_to_compressed, 0);
        assert_eq!(report.compressed_to_archived, 0);
    }

    #[test]
    fn test_compact_multiple_snapshots_mixed_ages() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_snapshot_dirs(root);

        // Fresh snapshot (keep in active)
        let s1 = sample_snapshot();
        let p1 = write_snapshot(root, &s1).unwrap();

        // Old snapshot (should compact)
        let s2 = Snapshot {
            session_id: "session-old".to_string(),
            summary: "Old snapshot".to_string(),
            key_context: vec![],
            full_transcript: String::new(),
            created_at: "2026-02-01T10:00:00Z".to_string(),
            tier: SnapshotTier::Active,
            embedding_ref: None,
        };
        let p2 = write_snapshot(root, &s2).unwrap();
        set_file_age(&p2, Duration::from_secs(10 * 24 * 60 * 60));

        let report = compact_snapshots(root).unwrap();
        assert_eq!(report.active_to_compressed, 1);

        // Fresh one stays
        assert!(p1.exists());
        // Old one removed from active
        assert!(!p2.exists());
    }

    #[test]
    fn test_compact_report_values() {
        let report = CompactReport {
            active_to_compressed: 3,
            compressed_to_archived: 1,
        };
        assert_eq!(report.active_to_compressed, 3);
        assert_eq!(report.compressed_to_archived, 1);
    }
}
