use std::path::Path;

use engram_core::{EngramError, Result};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// A wrapper around usearch's HNSW index for semantic vector search.
pub struct HnswIndex {
    index: Index,
    dimensions: usize,
}

impl HnswIndex {
    /// Build a new HNSW index from chunk ID keys and their embedding vectors.
    ///
    /// `entries` is a slice of `(key, vector)` pairs where key is a u64 identifier
    /// and vector is a slice of f32 values of length `dimensions`.
    pub fn build(entries: &[(u64, &[f32])], dimensions: usize) -> Result<Self> {
        let options = IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: false,
        };

        let index = Index::new(&options)
            .map_err(|e| EngramError::Index(format!("failed to create HNSW index: {e}")))?;

        if !entries.is_empty() {
            index
                .reserve(entries.len())
                .map_err(|e| EngramError::Index(format!("failed to reserve capacity: {e}")))?;

            for &(key, vector) in entries {
                index
                    .add(key, vector)
                    .map_err(|e| EngramError::Index(format!("failed to add vector {key}: {e}")))?;
            }
        }

        Ok(Self { index, dimensions })
    }

    /// Search the index for the `top_k` nearest neighbors to `query`.
    ///
    /// Returns a vector of `(key, distance)` pairs sorted by ascending distance.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>> {
        let matches = self
            .index
            .search(query, top_k)
            .map_err(|e| EngramError::Index(format!("search failed: {e}")))?;

        let results: Vec<(u64, f32)> = matches
            .keys
            .into_iter()
            .zip(matches.distances)
            .collect();

        Ok(results)
    }

    /// Save the index to a file at the given path.
    pub fn save(&self, path: &Path) -> Result<()> {
        let path_str = path
            .to_str()
            .ok_or_else(|| EngramError::Index("path is not valid UTF-8".to_string()))?;

        self.index
            .save(path_str)
            .map_err(|e| EngramError::Index(format!("failed to save index: {e}")))?;

        Ok(())
    }

    /// Load an index from a file at the given path.
    pub fn load(path: &Path, dimensions: usize) -> Result<Self> {
        let path_str = path
            .to_str()
            .ok_or_else(|| EngramError::Index("path is not valid UTF-8".to_string()))?;

        let options = IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: false,
        };

        let index = Index::new(&options)
            .map_err(|e| EngramError::Index(format!("failed to create index for load: {e}")))?;

        index
            .load(path_str)
            .map_err(|e| EngramError::Index(format!("failed to load index: {e}")))?;

        Ok(Self { index, dimensions })
    }

    /// Add a single vector to the index after initial build.
    pub fn add(&self, key: u64, vector: &[f32]) -> Result<()> {
        self.index
            .reserve(self.index.size() + 1)
            .map_err(|e| EngramError::Index(format!("failed to reserve for add: {e}")))?;
        self.index
            .add(key, vector)
            .map_err(|e| EngramError::Index(format!("failed to add vector {key}: {e}")))?;
        Ok(())
    }

    /// Returns the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.index.size() == 0
    }

    /// Returns the dimensionality of vectors in this index.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_vector(dimensions: usize, seed: f32) -> Vec<f32> {
        (0..dimensions)
            .map(|i| ((i as f32 + seed) * 0.1).sin())
            .collect()
    }

    #[test]
    fn build_and_search() {
        let dims = 64;
        let v0 = make_vector(dims, 0.0);
        let v1 = make_vector(dims, 1.0);
        let v2 = make_vector(dims, 2.0);

        let entries: Vec<(u64, &[f32])> = vec![(0, &v0), (1, &v1), (2, &v2)];

        let index = HnswIndex::build(&entries, dims).unwrap();
        assert_eq!(index.len(), 3);

        // Search for v0 — should find key 0 as the closest match
        let results = index.search(&v0, 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0, "closest match to v0 should be key 0");
        assert!(results[0].1 <= results[1].1, "results should be sorted by distance");
    }

    #[test]
    fn build_empty_index() {
        let index = HnswIndex::build(&[], 64).unwrap();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn search_returns_ascending_distance() {
        let dims = 32;
        let mut entries = Vec::new();
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| make_vector(dims, i as f32 * 5.0)).collect();
        for (i, v) in vectors.iter().enumerate() {
            entries.push((i as u64, v.as_slice()));
        }

        let index = HnswIndex::build(&entries, dims).unwrap();
        let results = index.search(&vectors[0], 10).unwrap();

        // Distances should be non-decreasing
        for window in results.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "distances must be ascending: {} <= {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn save_and_load() {
        let dims = 32;
        let v0 = make_vector(dims, 0.0);
        let v1 = make_vector(dims, 1.0);
        let entries: Vec<(u64, &[f32])> = vec![(10, &v0), (20, &v1)];

        let index = HnswIndex::build(&entries, dims).unwrap();

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.hnsw");
        index.save(&path).unwrap();

        let loaded = HnswIndex::load(&path, dims).unwrap();
        assert_eq!(loaded.len(), 2);

        // Search should return same results after load
        let results = loaded.search(&v0, 2).unwrap();
        assert_eq!(results[0].0, 10);
    }

    #[test]
    fn dimensions_accessor() {
        let index = HnswIndex::build(&[], 768).unwrap();
        assert_eq!(index.dimensions(), 768);
    }

    #[test]
    fn search_top_k_limits_results() {
        let dims = 16;
        let vectors: Vec<Vec<f32>> = (0..20).map(|i| make_vector(dims, i as f32)).collect();
        let entries: Vec<(u64, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, v.as_slice()))
            .collect();

        let index = HnswIndex::build(&entries, dims).unwrap();
        let results = index.search(&vectors[0], 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn single_vector_search() {
        let dims = 32;
        let v0 = make_vector(dims, 42.0);
        let entries: Vec<(u64, &[f32])> = vec![(99, &v0)];

        let index = HnswIndex::build(&entries, dims).unwrap();
        let results = index.search(&v0, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 99);
        // Distance to self should be near zero for cosine
        assert!(results[0].1 < 0.001, "self-distance should be ~0");
    }

    #[test]
    fn load_nonexistent_file_returns_error() {
        let result = HnswIndex::load(Path::new("/nonexistent/path.hnsw"), 64);
        assert!(result.is_err());
    }
}
