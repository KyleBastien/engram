use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use engram_core::{ChunkMetadata, EngramError, Result, StoreConfig};
use engram_store::{read_chunks_jsonl, read_embeddings_bin, read_manifest};
use sha2::{Digest, Sha256};

use crate::{Bm25Document, Bm25Index, ChunkEntry, HnswIndex, HybridSearch, MetadataIndex};

/// The live in-memory index manager holding HNSW, BM25, and metadata indexes.
///
/// Created via [`IndexManager::boot`], which loads from the git store
/// (fast path via compiled cache, slow path via deserialization from index files).
pub struct IndexManager {
    hnsw: HnswIndex,
    bm25: Bm25Index,
    metadata: MetadataIndex,
    /// Ordered chunks matching HNSW/BM25 key order (key = index position).
    chunks: Vec<ChunkMetadata>,
    /// Boot timing in milliseconds.
    pub boot_time_ms: u64,
    /// Whether cache was used during boot ("hit" or "cold").
    pub cache_status: String,
}

impl IndexManager {
    /// Boot the index manager by loading indexes from the store.
    ///
    /// Sequence:
    /// 1. Read `manifest.json` from the store
    /// 2. Compute SHA-256 hash of the manifest
    /// 3. Try loading compiled cache (fast path)
    /// 4. On cache miss: walk `index/`, deserialize all `.chunks.jsonl` and
    ///    `.embeddings.bin` files, build HNSW/BM25/metadata indexes, write cache
    pub async fn boot(store_root: &Path, _config: &StoreConfig) -> Result<Self> {
        let start = Instant::now();

        // 1. Read manifest
        let manifest = read_manifest(store_root)?
            .ok_or_else(|| EngramError::Store("no manifest.json found".into()))?;

        // 2. Compute manifest hash
        let manifest_json = serde_json::to_string(&manifest)
            .map_err(|e| EngramError::Serialize(e.to_string()))?;
        let manifest_hash = {
            let hash = Sha256::digest(manifest_json.as_bytes());
            format!("{:x}", hash)
        };

        let dimensions = manifest.dimensions;
        let cache_dir = store_root.join(".engram-cache");

        // 3. Try compiled cache (fast path)
        if let Some((hnsw, bm25, chunks)) =
            try_load_cache(&cache_dir, &manifest_hash, dimensions)?
        {
            let metadata = MetadataIndex::build(&chunks);
            let elapsed = start.elapsed();
            eprintln!("Boot completed in {}ms (cache hit)", elapsed.as_millis());
            return Ok(Self {
                hnsw,
                bm25,
                metadata,
                chunks,
                boot_time_ms: elapsed.as_millis() as u64,
                cache_status: "hit".to_string(),
            });
        }

        // 4. Cold boot — walk index/ directory
        let (all_chunks, all_vectors) = walk_index(store_root, dimensions)?;

        // 5. Build HNSW index
        let entries: Vec<(u64, &[f32])> = (0..all_chunks.len())
            .map(|i| {
                let offset = i * dimensions;
                (i as u64, &all_vectors[offset..offset + dimensions])
            })
            .collect();
        let hnsw = HnswIndex::build(&entries, dimensions)?;

        // 6. Build BM25 index
        let bm25_docs: Vec<Bm25Document> = all_chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| Bm25Document {
                key: i as u64,
                name: chunk.name.clone(),
                signature: chunk.signature.clone(),
                tags: chunk.tags.clone(),
            })
            .collect();
        let bm25 = Bm25Index::build(&bm25_docs);

        // 7. Build metadata index
        let metadata = MetadataIndex::build(&all_chunks);

        // 8. Write compiled cache for next boot
        write_cache(
            &cache_dir,
            &hnsw,
            &bm25,
            &all_chunks,
            dimensions,
            &manifest_hash,
        )?;

        // 9. Report timing
        let chunk_count = all_chunks.len();
        let elapsed = start.elapsed();
        if elapsed.as_secs() >= 1 {
            eprintln!(
                "Boot completed in {:.1}s (cold, {} chunks)",
                elapsed.as_secs_f64(),
                chunk_count
            );
        } else {
            eprintln!(
                "Boot completed in {}ms (cold, {} chunks)",
                elapsed.as_millis(),
                chunk_count
            );
        }

        Ok(Self {
            hnsw,
            bm25,
            metadata,
            chunks: all_chunks,
            boot_time_ms: elapsed.as_millis() as u64,
            cache_status: "cold".to_string(),
        })
    }

    /// Access the HNSW vector index.
    pub fn hnsw(&self) -> &HnswIndex {
        &self.hnsw
    }

    /// Access the BM25 keyword index.
    pub fn bm25(&self) -> &Bm25Index {
        &self.bm25
    }

    /// Access the metadata lookup index.
    pub fn metadata(&self) -> &MetadataIndex {
        &self.metadata
    }

    /// Returns the total number of indexed chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Consume the IndexManager and produce a HybridSearch instance.
    ///
    /// Converts the ordered chunk metadata into the HashMap<u64, ChunkEntry>
    /// that HybridSearch expects, where keys match the HNSW/BM25 index positions.
    pub fn into_hybrid_search(self) -> HybridSearch {
        let metadata = self
            .chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let (repo, file) = extract_repo_file(&chunk.chunk_id);
                (
                    i as u64,
                    ChunkEntry {
                        chunk_id: chunk.chunk_id,
                        kind: chunk.kind,
                        name: chunk.name,
                        signature: chunk.signature,
                        file,
                        repo,
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        stale: false,
                    },
                )
            })
            .collect();
        HybridSearch::new(self.hnsw, self.bm25, metadata)
    }
}

/// Extract repo name and file path from a chunk_id.
///
/// chunk_id format: `{repo}#{file_path}#{chunk_name}`
fn extract_repo_file(chunk_id: &str) -> (String, String) {
    let first = chunk_id.find('#');
    let last = chunk_id.rfind('#');
    match (first, last) {
        (Some(f), Some(l)) if f < l => (
            chunk_id[..f].to_string(),
            chunk_id[f + 1..l].to_string(),
        ),
        _ => ("unknown".to_string(), chunk_id.to_string()),
    }
}

/// Walk the `index/` directory to collect all chunks and their embedding vectors.
fn walk_index(store_root: &Path, dimensions: usize) -> Result<(Vec<ChunkMetadata>, Vec<f32>)> {
    let index_dir = store_root.join("index");
    let mut all_chunks: Vec<ChunkMetadata> = Vec::new();
    let mut all_vectors: Vec<f32> = Vec::new();

    if !index_dir.is_dir() {
        return Ok((all_chunks, all_vectors));
    }

    let chunk_files = find_chunk_files(&index_dir)?;

    for chunk_path in &chunk_files {
        let chunks = read_chunks_jsonl(chunk_path)?;
        if chunks.is_empty() {
            continue;
        }

        // Derive the corresponding .embeddings.bin path
        let chunk_str = chunk_path.to_string_lossy();
        let emb_path = if let Some(base) = chunk_str.strip_suffix(".chunks.jsonl") {
            PathBuf::from(format!("{base}.embeddings.bin"))
        } else {
            continue;
        };

        if !emb_path.exists() {
            continue;
        }

        let emb_file = read_embeddings_bin(&emb_path)?;

        for chunk in &chunks {
            let offset = chunk.embedding_offset as usize * dimensions;
            if offset + dimensions <= emb_file.vectors.len() {
                all_vectors.extend_from_slice(&emb_file.vectors[offset..offset + dimensions]);
                all_chunks.push(chunk.clone());
            }
        }
    }

    Ok((all_chunks, all_vectors))
}

/// Recursively find all `.chunks.jsonl` files under a directory.
fn find_chunk_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut results = Vec::new();
    collect_chunk_files(dir, &mut results)?;
    results.sort();
    Ok(results)
}

fn collect_chunk_files(dir: &Path, results: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_chunk_files(&path, results)?;
        } else if path.to_string_lossy().ends_with(".chunks.jsonl") {
            results.push(path);
        }
    }
    Ok(())
}

// --- Inline cache functions (engram-cache depends on engram-query, so we inline to avoid cycles) ---

fn try_load_cache(
    cache_dir: &Path,
    manifest_hash: &str,
    dimensions: usize,
) -> Result<Option<(HnswIndex, Bm25Index, Vec<ChunkMetadata>)>> {
    let fingerprint = match fs::read_to_string(cache_dir.join("fingerprint")) {
        Ok(fp) => fp,
        Err(_) => return Ok(None),
    };
    if fingerprint != manifest_hash {
        return Ok(None);
    }

    let hnsw = match HnswIndex::load(&cache_dir.join("hnsw.index"), dimensions) {
        Ok(idx) => idx,
        Err(_) => return Ok(None),
    };

    let bm25 = match Bm25Index::load(&cache_dir.join("bm25.index")) {
        Ok(idx) => idx,
        Err(_) => return Ok(None),
    };

    let metadata_bytes = match fs::read(cache_dir.join("metadata.bin")) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(None),
    };
    let metadata: Vec<ChunkMetadata> = match serde_json::from_slice(&metadata_bytes) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };

    Ok(Some((hnsw, bm25, metadata)))
}

fn write_cache(
    cache_dir: &Path,
    hnsw: &HnswIndex,
    bm25: &Bm25Index,
    metadata: &[ChunkMetadata],
    dimensions: usize,
    manifest_hash: &str,
) -> Result<()> {
    fs::create_dir_all(cache_dir)?;
    fs::write(cache_dir.join("fingerprint"), manifest_hash)?;
    fs::write(cache_dir.join("dimensions"), dimensions.to_string())?;
    hnsw.save(&cache_dir.join("hnsw.index"))?;
    bm25.save(&cache_dir.join("bm25.index"))?;
    let metadata_bytes =
        serde_json::to_vec(metadata).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(cache_dir.join("metadata.bin"), metadata_bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::{ChunkKind, Manifest};
    use engram_store::{write_chunks_jsonl, write_embeddings_bin, write_manifest};
    use tempfile::TempDir;

    fn make_chunk(id: &str, name: &str, offset: usize) -> ChunkMetadata {
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
            embedding_offset: offset as u64,
        }
    }

    fn make_vector(dims: usize, seed: f32) -> Vec<f32> {
        (0..dims)
            .map(|i| ((i as f32 + seed) * 0.1).sin())
            .collect()
    }

    /// Set up a minimal store with manifest, chunks, and embeddings.
    fn setup_store(tmp: &TempDir, dims: usize) -> PathBuf {
        let store_root = tmp.path().join("store");
        let src_dir = store_root.join("index").join("repo").join("src");
        fs::create_dir_all(&src_dir).unwrap();

        let chunks = vec![
            make_chunk("repo#src/lib.rs#foo", "foo", 0),
            make_chunk("repo#src/lib.rs#bar", "bar", 1),
        ];
        write_chunks_jsonl(&src_dir.join("lib.rs.chunks.jsonl"), &chunks).unwrap();

        let vectors = vec![make_vector(dims, 0.0), make_vector(dims, 1.0)];
        write_embeddings_bin(&src_dir.join("lib.rs.embeddings.bin"), &vectors, dims).unwrap();

        let manifest = Manifest {
            chunk_count: 2,
            last_indexed_commit: Some("deadbeef".to_string()),
            model_name: "test-model".to_string(),
            dimensions: dims,
            source_repos: vec!["repo".to_string()],
            created_at: "2026-03-09T00:00:00Z".to_string(),
            updated_at: "2026-03-09T00:00:00Z".to_string(),
        };
        write_manifest(&store_root, &manifest).unwrap();

        store_root
    }

    #[tokio::test]
    async fn boot_cold_populates_indexes() {
        let tmp = TempDir::new().unwrap();
        let dims = 32;
        let store_root = setup_store(&tmp, dims);
        let config = StoreConfig::default();

        let mgr = IndexManager::boot(&store_root, &config).await.unwrap();

        assert_eq!(mgr.hnsw().len(), 2);
        assert_eq!(mgr.bm25().len(), 2);
        assert!(mgr.metadata().lookup_by_id("repo#src/lib.rs#foo").is_some());
        assert!(mgr.metadata().lookup_by_id("repo#src/lib.rs#bar").is_some());
    }

    #[tokio::test]
    async fn boot_writes_cache_on_cold() {
        let tmp = TempDir::new().unwrap();
        let dims = 32;
        let store_root = setup_store(&tmp, dims);
        let config = StoreConfig::default();

        IndexManager::boot(&store_root, &config).await.unwrap();

        let cache_dir = store_root.join(".engram-cache");
        assert!(cache_dir.join("fingerprint").exists());
        assert!(cache_dir.join("hnsw.index").exists());
        assert!(cache_dir.join("bm25.index").exists());
        assert!(cache_dir.join("metadata.bin").exists());
        assert!(cache_dir.join("dimensions").exists());
    }

    #[tokio::test]
    async fn boot_cache_hit_on_second_boot() {
        let tmp = TempDir::new().unwrap();
        let dims = 32;
        let store_root = setup_store(&tmp, dims);
        let config = StoreConfig::default();

        // First boot (cold)
        IndexManager::boot(&store_root, &config).await.unwrap();

        // Second boot (should be cache hit)
        let mgr = IndexManager::boot(&store_root, &config).await.unwrap();
        assert_eq!(mgr.hnsw().len(), 2);
        assert_eq!(mgr.bm25().len(), 2);
        assert!(mgr.metadata().lookup_by_id("repo#src/lib.rs#foo").is_some());
    }

    #[tokio::test]
    async fn boot_no_manifest_returns_error() {
        let tmp = TempDir::new().unwrap();
        let store_root = tmp.path().join("empty-store");
        fs::create_dir_all(&store_root).unwrap();
        let config = StoreConfig::default();

        let result = IndexManager::boot(&store_root, &config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn boot_empty_index_directory() {
        let tmp = TempDir::new().unwrap();
        let store_root = tmp.path().join("store");
        fs::create_dir_all(store_root.join("index")).unwrap();

        let manifest = Manifest {
            chunk_count: 0,
            last_indexed_commit: None,
            model_name: "test-model".to_string(),
            dimensions: 32,
            source_repos: vec![],
            created_at: "2026-03-09T00:00:00Z".to_string(),
            updated_at: "2026-03-09T00:00:00Z".to_string(),
        };
        write_manifest(&store_root, &manifest).unwrap();

        let config = StoreConfig::default();
        let mgr = IndexManager::boot(&store_root, &config).await.unwrap();
        assert_eq!(mgr.hnsw().len(), 0);
        assert_eq!(mgr.bm25().len(), 0);
    }

    #[tokio::test]
    async fn boot_multiple_source_files() {
        let tmp = TempDir::new().unwrap();
        let dims = 32;
        let store_root = tmp.path().join("store");
        let src_dir = store_root.join("index").join("repo").join("src");
        fs::create_dir_all(&src_dir).unwrap();

        // File 1
        let chunks1 = vec![make_chunk("repo#src/a.rs#alpha", "alpha", 0)];
        write_chunks_jsonl(&src_dir.join("a.rs.chunks.jsonl"), &chunks1).unwrap();
        write_embeddings_bin(
            &src_dir.join("a.rs.embeddings.bin"),
            &[make_vector(dims, 0.0)],
            dims,
        )
        .unwrap();

        // File 2
        let chunks2 = vec![
            make_chunk("repo#src/b.rs#beta", "beta", 0),
            make_chunk("repo#src/b.rs#gamma", "gamma", 1),
        ];
        write_chunks_jsonl(&src_dir.join("b.rs.chunks.jsonl"), &chunks2).unwrap();
        write_embeddings_bin(
            &src_dir.join("b.rs.embeddings.bin"),
            &[make_vector(dims, 1.0), make_vector(dims, 2.0)],
            dims,
        )
        .unwrap();

        let manifest = Manifest {
            chunk_count: 3,
            last_indexed_commit: Some("abc".to_string()),
            model_name: "test".to_string(),
            dimensions: dims,
            source_repos: vec!["repo".to_string()],
            created_at: "2026-03-09T00:00:00Z".to_string(),
            updated_at: "2026-03-09T00:00:00Z".to_string(),
        };
        write_manifest(&store_root, &manifest).unwrap();

        let config = StoreConfig::default();
        let mgr = IndexManager::boot(&store_root, &config).await.unwrap();
        assert_eq!(mgr.hnsw().len(), 3);
        assert_eq!(mgr.bm25().len(), 3);
        assert!(mgr.metadata().lookup_by_id("repo#src/a.rs#alpha").is_some());
        assert!(mgr.metadata().lookup_by_id("repo#src/b.rs#beta").is_some());
        assert!(mgr.metadata().lookup_by_id("repo#src/b.rs#gamma").is_some());
    }

    #[tokio::test]
    async fn boot_search_works_after_cold_boot() {
        let tmp = TempDir::new().unwrap();
        let dims = 32;
        let store_root = setup_store(&tmp, dims);
        let config = StoreConfig::default();

        let mgr = IndexManager::boot(&store_root, &config).await.unwrap();

        // Vector search should work
        let query = make_vector(dims, 0.0);
        let results = mgr.hnsw().search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);

        // BM25 search should work
        let bm25_results = mgr.bm25().search("foo", 2);
        assert!(!bm25_results.is_empty());
    }
}
