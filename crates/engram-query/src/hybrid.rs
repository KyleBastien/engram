use std::collections::HashMap;

use engram_core::{ChunkKind, Result};

use crate::{Bm25Index, HnswIndex};

/// Default alpha value favoring semantic search (0.7 = 70% vector, 30% BM25).
pub const DEFAULT_ALPHA: f64 = 0.7;

/// Result from a hybrid search query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk_id: String,
    pub score: f64,
    pub kind: ChunkKind,
    pub name: String,
    pub signature: Option<String>,
    pub file: String,
    pub repo: String,
    pub lines: (u32, u32),
    pub stale: bool,
}

/// Metadata entry for a chunk, used to build SearchResults.
pub struct ChunkEntry {
    pub chunk_id: String,
    pub kind: ChunkKind,
    pub name: String,
    pub signature: Option<String>,
    pub file: String,
    pub repo: String,
    pub start_line: u32,
    pub end_line: u32,
    pub stale: bool,
}

/// Hybrid search combining HNSW vector search and BM25 keyword search.
pub struct HybridSearch {
    hnsw: HnswIndex,
    bm25: Bm25Index,
    metadata: HashMap<u64, ChunkEntry>,
}

impl HybridSearch {
    /// Create a new HybridSearch from pre-built indexes and chunk metadata.
    pub fn new(hnsw: HnswIndex, bm25: Bm25Index, metadata: HashMap<u64, ChunkEntry>) -> Self {
        Self {
            hnsw,
            bm25,
            metadata,
        }
    }

    /// Look up a single chunk by its chunk_id.
    pub fn lookup_by_chunk_id(&self, chunk_id: &str) -> Option<&ChunkEntry> {
        self.metadata
            .values()
            .find(|entry| entry.chunk_id == chunk_id)
    }

    /// Look up all chunks belonging to a file path.
    pub fn lookup_by_file(&self, file_path: &str) -> Vec<&ChunkEntry> {
        self.metadata
            .values()
            .filter(|entry| entry.file == file_path)
            .collect()
    }

    /// Look up all chunks with a given symbol name.
    pub fn lookup_by_symbol(&self, symbol_name: &str) -> Vec<&ChunkEntry> {
        self.metadata
            .values()
            .filter(|entry| entry.name == symbol_name)
            .collect()
    }

    /// Search combining vector similarity and keyword relevance.
    ///
    /// `query` is the text query for BM25 keyword search.
    /// `query_embedding` is the vector embedding of the query for HNSW search.
    /// `top_k` limits the number of results returned.
    /// `alpha` controls the weight: alpha * vector_score + (1 - alpha) * bm25_score.
    pub async fn search(
        &self,
        query: &str,
        query_embedding: &[f32],
        top_k: usize,
        alpha: f64,
    ) -> Result<Vec<SearchResult>> {
        // Fetch more candidates than needed to allow merging and deduplication
        let fetch_k = top_k * 3;

        // HNSW vector search — returns (key, distance) ascending
        let hnsw_results = self.hnsw.search(query_embedding, fetch_k)?;

        // BM25 keyword search — returns (key, score) descending
        let bm25_results = self.bm25.search(query, fetch_k);

        // Normalize both to [0,1] scores (higher = better)
        let hnsw_scores = normalize_vector_scores(&hnsw_results);
        let bm25_scores = normalize_bm25_scores(&bm25_results);

        // Combine scores, deduplicating by key
        let mut combined: HashMap<u64, f64> = HashMap::new();

        for (key, score) in &hnsw_scores {
            *combined.entry(*key).or_insert(0.0) += alpha * score;
        }

        for (key, score) in &bm25_scores {
            *combined.entry(*key).or_insert(0.0) += (1.0 - alpha) * score;
        }

        // Sort by combined score descending, take top_k
        let mut sorted: Vec<(u64, f64)> = combined.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(top_k);

        // Build SearchResults from metadata
        let results: Vec<SearchResult> = sorted
            .into_iter()
            .filter_map(|(key, score)| {
                self.metadata.get(&key).map(|entry| SearchResult {
                    chunk_id: entry.chunk_id.clone(),
                    score,
                    kind: entry.kind.clone(),
                    name: entry.name.clone(),
                    signature: entry.signature.clone(),
                    file: entry.file.clone(),
                    repo: entry.repo.clone(),
                    lines: (entry.start_line, entry.end_line),
                    stale: entry.stale,
                })
            })
            .collect();

        Ok(results)
    }
}

/// Normalize HNSW distances (ascending, lower = better) to [0,1] scores (higher = better).
///
/// Uses min-max normalization: score = (max_dist - dist) / (max_dist - min_dist).
/// If all distances are equal, all scores are 1.0.
fn normalize_vector_scores(results: &[(u64, f32)]) -> Vec<(u64, f64)> {
    if results.is_empty() {
        return Vec::new();
    }
    if results.len() == 1 {
        return vec![(results[0].0, 1.0)];
    }

    let min_dist = results.iter().map(|r| r.1).fold(f32::INFINITY, f32::min);
    let max_dist = results
        .iter()
        .map(|r| r.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = max_dist - min_dist;

    results
        .iter()
        .map(|&(key, dist)| {
            let score = if range < f32::EPSILON {
                1.0
            } else {
                (max_dist - dist) as f64 / range as f64
            };
            (key, score)
        })
        .collect()
}

/// Normalize BM25 scores (descending, higher = better) to [0,1] range.
///
/// Uses min-max normalization: normalized = (score - min) / (max - min).
/// If all scores are equal, all normalized values are 1.0.
fn normalize_bm25_scores(results: &[(u64, f64)]) -> Vec<(u64, f64)> {
    if results.is_empty() {
        return Vec::new();
    }
    if results.len() == 1 {
        return vec![(results[0].0, 1.0)];
    }

    let min_score = results.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let max_score = results
        .iter()
        .map(|r| r.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max_score - min_score;

    results
        .iter()
        .map(|&(key, score)| {
            let normalized = if range < f64::EPSILON {
                1.0
            } else {
                (score - min_score) / range
            };
            (key, normalized)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bm25Document;

    fn make_vector(dimensions: usize, seed: f32) -> Vec<f32> {
        (0..dimensions)
            .map(|i| ((i as f32 + seed) * 0.1).sin())
            .collect()
    }

    fn make_entry(key: u64, name: &str, file: &str) -> (u64, ChunkEntry) {
        (
            key,
            ChunkEntry {
                chunk_id: format!("repo#{}#{}", file, name),
                kind: ChunkKind::Function,
                name: name.to_string(),
                signature: Some(format!("fn {}()", name)),
                file: file.to_string(),
                repo: "test-repo".to_string(),
                start_line: 1,
                end_line: 10,
                stale: false,
            },
        )
    }

    fn make_doc(key: u64, name: &str, signature: Option<&str>, tags: &[&str]) -> Bm25Document {
        Bm25Document {
            key,
            name: name.to_string(),
            signature: signature.map(|s| s.to_string()),
            tags: tags.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn build_test_search() -> HybridSearch {
        let dims = 32;
        let v0 = make_vector(dims, 0.0);
        let v1 = make_vector(dims, 1.0);
        let v2 = make_vector(dims, 2.0);

        let entries: Vec<(u64, &[f32])> = vec![(0, &v0), (1, &v1), (2, &v2)];
        let hnsw = HnswIndex::build(&entries, dims).unwrap();

        let docs = vec![
            make_doc(0, "calculate_total", Some("fn calculate_total(items: &[Item]) -> f64"), &["math"]),
            make_doc(1, "render_button", Some("fn render_button(label: &str)"), &["ui"]),
            make_doc(2, "parse_config", Some("fn parse_config(path: &Path) -> Config"), &["config"]),
        ];
        let bm25 = Bm25Index::build(&docs);

        let metadata: HashMap<u64, ChunkEntry> = vec![
            make_entry(0, "calculate_total", "src/math.rs"),
            make_entry(1, "render_button", "src/ui.rs"),
            make_entry(2, "parse_config", "src/config.rs"),
        ]
        .into_iter()
        .collect();

        HybridSearch::new(hnsw, bm25, metadata)
    }

    #[tokio::test]
    async fn hybrid_search_returns_results() {
        let search = build_test_search();
        let query_vec = make_vector(32, 0.0);
        let results = search.search("calculate", &query_vec, 3, DEFAULT_ALPHA).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[tokio::test]
    async fn results_sorted_by_descending_score() {
        let search = build_test_search();
        let query_vec = make_vector(32, 0.0);
        let results = search.search("calculate", &query_vec, 3, DEFAULT_ALPHA).await.unwrap();

        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "scores must be descending: {} >= {}",
                window[0].score,
                window[1].score
            );
        }
    }

    #[tokio::test]
    async fn default_alpha_is_0_7() {
        assert!((DEFAULT_ALPHA - 0.7).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn search_result_contains_all_fields() {
        let search = build_test_search();
        let query_vec = make_vector(32, 0.0);
        let results = search.search("calculate", &query_vec, 1, DEFAULT_ALPHA).await.unwrap();
        assert!(!results.is_empty());

        let r = &results[0];
        assert!(!r.chunk_id.is_empty());
        assert!(r.score > 0.0);
        assert!(!r.name.is_empty());
        assert!(!r.file.is_empty());
        assert!(!r.repo.is_empty());
        assert!(r.lines.0 <= r.lines.1);
    }

    #[tokio::test]
    async fn deduplication_by_chunk_id() {
        let search = build_test_search();
        let query_vec = make_vector(32, 0.0);
        // Both HNSW and BM25 may return the same keys — results should be deduplicated
        let results = search.search("calculate total math", &query_vec, 10, DEFAULT_ALPHA).await.unwrap();

        let mut seen_ids: Vec<&str> = Vec::new();
        for r in &results {
            assert!(
                !seen_ids.contains(&r.chunk_id.as_str()),
                "duplicate chunk_id: {}",
                r.chunk_id
            );
            seen_ids.push(&r.chunk_id);
        }
    }

    #[tokio::test]
    async fn alpha_1_uses_only_vector() {
        let search = build_test_search();
        // Use a vector close to key 0
        let query_vec = make_vector(32, 0.0);
        let results = search.search("render_button ui", &query_vec, 3, 1.0).await.unwrap();

        // With alpha=1.0, only vector scores matter. The closest vector to seed 0.0 is key 0.
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "repo#src/math.rs#calculate_total");
    }

    #[tokio::test]
    async fn alpha_0_uses_only_bm25() {
        let search = build_test_search();
        // Use a vector close to key 2 but query text matching key 1
        let query_vec = make_vector(32, 2.0);
        let results = search.search("render_button ui", &query_vec, 3, 0.0).await.unwrap();

        // With alpha=0.0, only BM25 scores matter. "render_button ui" should match key 1.
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "repo#src/ui.rs#render_button");
    }

    #[tokio::test]
    async fn empty_indexes_return_empty() {
        let hnsw = HnswIndex::build(&[], 32).unwrap();
        let bm25 = Bm25Index::build(&[]);
        let metadata = HashMap::new();

        let search = HybridSearch::new(hnsw, bm25, metadata);
        let query_vec = make_vector(32, 0.0);
        let results = search.search("anything", &query_vec, 10, DEFAULT_ALPHA).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn top_k_limits_results() {
        let dims = 16;
        let vectors: Vec<Vec<f32>> = (0..20).map(|i| make_vector(dims, i as f32)).collect();
        let entries: Vec<(u64, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, v.as_slice()))
            .collect();
        let hnsw = HnswIndex::build(&entries, dims).unwrap();

        let docs: Vec<Bm25Document> = (0..20)
            .map(|i| make_doc(i, &format!("func_{i}"), None, &["common"]))
            .collect();
        let bm25 = Bm25Index::build(&docs);

        let metadata: HashMap<u64, ChunkEntry> = (0..20)
            .map(|i| make_entry(i, &format!("func_{i}"), &format!("src/f{i}.rs")))
            .into_iter()
            .collect();

        let search = HybridSearch::new(hnsw, bm25, metadata);
        let query_vec = make_vector(dims, 0.0);
        let results = search.search("common", &query_vec, 5, DEFAULT_ALPHA).await.unwrap();
        assert!(results.len() <= 5);
    }

    #[tokio::test]
    async fn stale_field_propagated() {
        let dims = 32;
        let v0 = make_vector(dims, 0.0);
        let hnsw = HnswIndex::build(&[(0, v0.as_slice())], dims).unwrap();
        let bm25 = Bm25Index::build(&[make_doc(0, "stale_func", None, &[])]);

        let mut metadata = HashMap::new();
        metadata.insert(
            0,
            ChunkEntry {
                chunk_id: "repo#file#stale_func".to_string(),
                kind: ChunkKind::Function,
                name: "stale_func".to_string(),
                signature: None,
                file: "src/lib.rs".to_string(),
                repo: "repo".to_string(),
                start_line: 1,
                end_line: 5,
                stale: true,
            },
        );

        let search = HybridSearch::new(hnsw, bm25, metadata);
        let results = search.search("stale_func", &v0, 1, DEFAULT_ALPHA).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].stale);
    }

    #[test]
    fn normalize_vector_single_result() {
        let results = vec![(42, 0.5f32)];
        let normalized = normalize_vector_scores(&results);
        assert_eq!(normalized.len(), 1);
        assert!((normalized[0].1 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_vector_range_0_to_1() {
        let results = vec![(0, 0.1f32), (1, 0.5f32), (2, 0.9f32)];
        let normalized = normalize_vector_scores(&results);

        for (_, score) in &normalized {
            assert!(*score >= 0.0 && *score <= 1.0, "score {score} not in [0,1]");
        }
        // Closest (lowest distance) should have highest score
        let key0_score = normalized.iter().find(|r| r.0 == 0).unwrap().1;
        let key2_score = normalized.iter().find(|r| r.0 == 2).unwrap().1;
        assert!(key0_score > key2_score);
    }

    #[test]
    fn normalize_bm25_range_0_to_1() {
        let results = vec![(0, 3.0), (1, 1.5), (2, 0.5)];
        let normalized = normalize_bm25_scores(&results);

        for (_, score) in &normalized {
            assert!(*score >= 0.0 && *score <= 1.0, "score {score} not in [0,1]");
        }
        // Highest BM25 score should remain highest after normalization
        let key0_score = normalized.iter().find(|r| r.0 == 0).unwrap().1;
        let key2_score = normalized.iter().find(|r| r.0 == 2).unwrap().1;
        assert!(key0_score > key2_score);
    }

    #[test]
    fn normalize_empty_returns_empty() {
        assert!(normalize_vector_scores(&[]).is_empty());
        assert!(normalize_bm25_scores(&[]).is_empty());
    }

    #[test]
    fn normalize_equal_distances_all_ones() {
        let results = vec![(0, 0.5f32), (1, 0.5f32), (2, 0.5f32)];
        let normalized = normalize_vector_scores(&results);
        for (_, score) in normalized {
            assert!((score - 1.0).abs() < f64::EPSILON);
        }
    }
}
