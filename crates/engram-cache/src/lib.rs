use std::fs;
use std::path::Path;

use engram_core::{ChunkMetadata, EngramError, Result};
use engram_query::{Bm25Index, HnswIndex};

/// Write compiled index caches to the cache directory for fast subsequent boots.
///
/// Writes four files to `cache_dir`:
/// - `fingerprint`: the SHA-256 manifest hash for cache validation
/// - `hnsw.index`: the serialized HNSW vector index
/// - `bm25.index`: the serialized BM25 keyword index
/// - `metadata.bin`: the serialized chunk metadata
pub fn write_cache(
    cache_dir: &Path,
    hnsw: &HnswIndex,
    bm25: &Bm25Index,
    metadata: &[ChunkMetadata],
    manifest_hash: &str,
) -> Result<()> {
    fs::create_dir_all(cache_dir)?;

    // Write fingerprint (SHA-256 of manifest.json)
    fs::write(cache_dir.join("fingerprint"), manifest_hash)?;

    // Write HNSW index
    hnsw.save(&cache_dir.join("hnsw.index"))?;

    // Write BM25 index
    bm25.save(&cache_dir.join("bm25.index"))?;

    // Write metadata as serialized JSON bytes
    let metadata_bytes = serde_json::to_vec(metadata)
        .map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(cache_dir.join("metadata.bin"), metadata_bytes)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::{ChunkKind, ChunkMetadata};
    use engram_query::{Bm25Document, Bm25Index, HnswIndex};
    use tempfile::TempDir;

    fn make_chunk(id: &str, name: &str) -> ChunkMetadata {
        ChunkMetadata {
            chunk_id: id.to_string(),
            kind: ChunkKind::Function,
            name: name.to_string(),
            signature: Some(format!("fn {name}()")),
            start_line: 1,
            end_line: 10,
            content_hash: "abc123".to_string(),
            tags: vec!["test".to_string()],
            indexed_at: "2026-03-09T00:00:00Z".to_string(),
            source_commit: "deadbeef".to_string(),
            embedding_offset: 0,
        }
    }

    fn make_vector(dims: usize, seed: f32) -> Vec<f32> {
        (0..dims)
            .map(|i| ((i as f32 + seed) * 0.1).sin())
            .collect()
    }

    #[test]
    fn write_cache_creates_all_files() {
        let tmp = TempDir::new().unwrap();
        let cache_dir = tmp.path().join(".engram-cache");

        let dims = 32;
        let v0 = make_vector(dims, 0.0);
        let v1 = make_vector(dims, 1.0);
        let entries: Vec<(u64, &[f32])> = vec![(0, &v0), (1, &v1)];
        let hnsw = HnswIndex::build(&entries, dims).unwrap();

        let docs = vec![
            Bm25Document {
                key: 0,
                name: "foo".to_string(),
                signature: Some("fn foo()".to_string()),
                tags: vec![],
            },
            Bm25Document {
                key: 1,
                name: "bar".to_string(),
                signature: Some("fn bar()".to_string()),
                tags: vec![],
            },
        ];
        let bm25 = Bm25Index::build(&docs);

        let chunks = vec![
            make_chunk("repo#src/lib.rs#foo", "foo"),
            make_chunk("repo#src/lib.rs#bar", "bar"),
        ];

        write_cache(&cache_dir, &hnsw, &bm25, &chunks, "abc123hash").unwrap();

        assert!(cache_dir.join("fingerprint").exists());
        assert!(cache_dir.join("hnsw.index").exists());
        assert!(cache_dir.join("bm25.index").exists());
        assert!(cache_dir.join("metadata.bin").exists());
    }

    #[test]
    fn fingerprint_contains_manifest_hash() {
        let tmp = TempDir::new().unwrap();
        let cache_dir = tmp.path().join(".engram-cache");

        let hnsw = HnswIndex::build(&[], 32).unwrap();
        let bm25 = Bm25Index::build(&[]);

        write_cache(&cache_dir, &hnsw, &bm25, &[], "sha256_manifest_hash").unwrap();

        let fingerprint = fs::read_to_string(cache_dir.join("fingerprint")).unwrap();
        assert_eq!(fingerprint, "sha256_manifest_hash");
    }

    #[test]
    fn creates_cache_directory_if_missing() {
        let tmp = TempDir::new().unwrap();
        let cache_dir = tmp.path().join("nested").join(".engram-cache");

        let hnsw = HnswIndex::build(&[], 32).unwrap();
        let bm25 = Bm25Index::build(&[]);

        assert!(!cache_dir.exists());
        write_cache(&cache_dir, &hnsw, &bm25, &[], "hash").unwrap();
        assert!(cache_dir.exists());
    }

    #[test]
    fn metadata_bin_contains_serialized_chunks() {
        let tmp = TempDir::new().unwrap();
        let cache_dir = tmp.path().join(".engram-cache");

        let hnsw = HnswIndex::build(&[], 32).unwrap();
        let bm25 = Bm25Index::build(&[]);

        let chunks = vec![make_chunk("repo#src/lib.rs#foo", "foo")];

        write_cache(&cache_dir, &hnsw, &bm25, &chunks, "hash").unwrap();

        let bytes = fs::read(cache_dir.join("metadata.bin")).unwrap();
        let loaded: Vec<ChunkMetadata> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].chunk_id, "repo#src/lib.rs#foo");
    }

    #[test]
    fn overwrites_existing_cache() {
        let tmp = TempDir::new().unwrap();
        let cache_dir = tmp.path().join(".engram-cache");

        let hnsw = HnswIndex::build(&[], 32).unwrap();
        let bm25 = Bm25Index::build(&[]);

        write_cache(&cache_dir, &hnsw, &bm25, &[], "hash1").unwrap();
        assert_eq!(
            fs::read_to_string(cache_dir.join("fingerprint")).unwrap(),
            "hash1"
        );

        write_cache(&cache_dir, &hnsw, &bm25, &[], "hash2").unwrap();
        assert_eq!(
            fs::read_to_string(cache_dir.join("fingerprint")).unwrap(),
            "hash2"
        );
    }
}
