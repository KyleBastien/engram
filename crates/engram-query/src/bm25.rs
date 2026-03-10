use std::collections::HashMap;
use std::fs;
use std::path::Path;

use engram_core::Result;
use serde::{Deserialize, Serialize};

/// A BM25 inverted index for keyword search over chunk metadata.
///
/// Builds from chunk name, signature, and tags fields. Uses standard
/// BM25 parameters (k1=1.2, b=0.75) for scoring.
pub struct Bm25Index {
    /// BM25 parameter controlling term frequency saturation.
    k1: f64,
    /// BM25 parameter controlling document length normalization.
    b: f64,
    /// Mapping from term to list of (doc_index, term_frequency) pairs.
    inverted: HashMap<String, Vec<(usize, u32)>>,
    /// Total number of documents.
    doc_count: usize,
    /// Length of each document in tokens.
    doc_lengths: Vec<u32>,
    /// Average document length.
    avg_dl: f64,
    /// The u64 key for each document (parallel to doc_lengths).
    keys: Vec<u64>,
}

/// Serializable representation for save/load.
#[derive(Serialize, Deserialize)]
struct Bm25Data {
    k1: f64,
    b: f64,
    inverted: HashMap<String, Vec<(usize, u32)>>,
    doc_count: usize,
    doc_lengths: Vec<u32>,
    avg_dl: f64,
    keys: Vec<u64>,
}

/// Input document for building the BM25 index.
pub struct Bm25Document {
    /// Unique key identifying this chunk (same u64 key space as HNSW).
    pub key: u64,
    /// Chunk name.
    pub name: String,
    /// Optional signature text.
    pub signature: Option<String>,
    /// Tags associated with the chunk.
    pub tags: Vec<String>,
}

impl Bm25Index {
    /// Build a BM25 index from a slice of documents.
    ///
    /// Each document's searchable text is formed by concatenating its name,
    /// signature (if present), and tags. Text is tokenized on word boundaries
    /// and lowercased.
    pub fn build(documents: &[Bm25Document]) -> Self {
        let k1 = 1.2;
        let b = 0.75;
        let doc_count = documents.len();
        let mut inverted: HashMap<String, Vec<(usize, u32)>> = HashMap::new();
        let mut doc_lengths = Vec::with_capacity(doc_count);
        let mut keys = Vec::with_capacity(doc_count);

        for (doc_idx, doc) in documents.iter().enumerate() {
            keys.push(doc.key);

            // Build searchable text from name, signature, and tags
            let mut text = doc.name.clone();
            if let Some(ref sig) = doc.signature {
                text.push(' ');
                text.push_str(sig);
            }
            for tag in &doc.tags {
                text.push(' ');
                text.push_str(tag);
            }

            // Tokenize: split on non-alphanumeric boundaries, lowercase
            let tokens = tokenize(&text);
            doc_lengths.push(tokens.len() as u32);

            // Count term frequencies for this document
            let mut tf_map: HashMap<String, u32> = HashMap::new();
            for token in tokens {
                *tf_map.entry(token).or_insert(0) += 1;
            }

            // Add to inverted index
            for (term, tf) in tf_map {
                inverted.entry(term).or_default().push((doc_idx, tf));
            }
        }

        let avg_dl = if doc_count > 0 {
            doc_lengths.iter().map(|&l| l as f64).sum::<f64>() / doc_count as f64
        } else {
            0.0
        };

        Self {
            k1,
            b,
            inverted,
            doc_count,
            doc_lengths,
            avg_dl,
            keys,
        }
    }

    /// Search the index for the given query, returning up to `top_k` results.
    ///
    /// Returns `(chunk_id_key, score)` pairs sorted by descending BM25 score.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f64)> {
        if self.doc_count == 0 {
            return Vec::new();
        }

        let query_tokens = tokenize(query);
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for token in &query_tokens {
            if let Some(postings) = self.inverted.get(token) {
                let df = postings.len() as f64;
                // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                let idf =
                    ((self.doc_count as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(doc_idx, tf) in postings {
                    let dl = self.doc_lengths[doc_idx] as f64;
                    let tf_f = tf as f64;
                    // BM25 term score
                    let numerator = tf_f * (self.k1 + 1.0);
                    let denominator =
                        tf_f + self.k1 * (1.0 - self.b + self.b * dl / self.avg_dl);
                    let term_score = idf * numerator / denominator;

                    *scores.entry(doc_idx).or_insert(0.0) += term_score;
                }
            }
        }

        // Sort by descending score, take top_k
        let mut results: Vec<(u64, f64)> = scores
            .into_iter()
            .map(|(doc_idx, score)| (self.keys[doc_idx], score))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Save the index to a JSON file at the given path.
    pub fn save(&self, path: &Path) -> Result<()> {
        let data = Bm25Data {
            k1: self.k1,
            b: self.b,
            inverted: self.inverted.clone(),
            doc_count: self.doc_count,
            doc_lengths: self.doc_lengths.clone(),
            avg_dl: self.avg_dl,
            keys: self.keys.clone(),
        };
        let json = serde_json::to_string(&data)
            .map_err(|e| engram_core::EngramError::Serialize(e.to_string()))?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load an index from a JSON file at the given path.
    pub fn load(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let data: Bm25Data = serde_json::from_str(&json)
            .map_err(|e| engram_core::EngramError::Serialize(e.to_string()))?;
        Ok(Self {
            k1: data.k1,
            b: data.b,
            inverted: data.inverted,
            doc_count: data.doc_count,
            doc_lengths: data.doc_lengths,
            avg_dl: data.avg_dl,
            keys: data.keys,
        })
    }

    /// Returns the number of documents in the index.
    pub fn len(&self) -> usize {
        self.doc_count
    }

    /// Returns true if the index contains no documents.
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }
}

/// Tokenize text by splitting on non-alphanumeric boundaries and lowercasing.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_doc(key: u64, name: &str, signature: Option<&str>, tags: &[&str]) -> Bm25Document {
        Bm25Document {
            key,
            name: name.to_string(),
            signature: signature.map(|s| s.to_string()),
            tags: tags.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn build_and_search_basic() {
        let docs = vec![
            make_doc(1, "calculate_total", Some("fn calculate_total(items: &[Item]) -> f64"), &["math", "pricing"]),
            make_doc(2, "render_button", Some("fn render_button(label: &str)"), &["ui", "component"]),
            make_doc(3, "parse_config", Some("fn parse_config(path: &Path) -> Config"), &["config", "io"]),
        ];

        let index = Bm25Index::build(&docs);
        assert_eq!(index.len(), 3);

        let results = index.search("calculate total pricing", 3);
        assert!(!results.is_empty());
        // The first result should be the calculate_total chunk
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn search_empty_index() {
        let index = Bm25Index::build(&[]);
        assert!(index.is_empty());
        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn results_sorted_by_descending_score() {
        let docs = vec![
            make_doc(1, "foo bar baz", None, &[]),
            make_doc(2, "foo", None, &[]),
            make_doc(3, "foo foo foo bar bar baz", None, &[]),
        ];

        let index = Bm25Index::build(&docs);
        let results = index.search("foo bar baz", 3);

        // Scores should be descending
        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "scores must be descending: {} >= {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn bm25_uses_standard_parameters() {
        let docs = vec![make_doc(1, "test", None, &[])];
        let index = Bm25Index::build(&docs);
        // Verify k1=1.2 and b=0.75 are used (internal, but we test via scoring behavior)
        assert_eq!(index.k1, 1.2);
        assert_eq!(index.b, 0.75);
    }

    #[test]
    fn tokenizes_on_word_boundaries_and_lowercases() {
        let tokens = tokenize("Hello_World foo-bar BAZ.qux");
        assert_eq!(tokens, vec!["hello_world", "foo", "bar", "baz", "qux"]);
    }

    #[test]
    fn search_uses_tags() {
        let docs = vec![
            make_doc(1, "process", None, &["database", "migration"]),
            make_doc(2, "process", None, &["ui", "render"]),
        ];

        let index = Bm25Index::build(&docs);
        let results = index.search("database migration", 2);
        assert_eq!(results[0].0, 1, "doc with matching tags should rank first");
    }

    #[test]
    fn search_uses_signature() {
        let docs = vec![
            make_doc(1, "run", Some("fn run(connection: DatabaseConnection)"), &[]),
            make_doc(2, "run", Some("fn run(button: UiButton)"), &[]),
        ];

        let index = Bm25Index::build(&docs);
        let results = index.search("DatabaseConnection", 2);
        assert_eq!(results[0].0, 1, "doc with matching signature should rank first");
    }

    #[test]
    fn top_k_limits_results() {
        let docs: Vec<Bm25Document> = (0..20)
            .map(|i| make_doc(i, &format!("func_{i}"), None, &["common"]))
            .collect();

        let index = Bm25Index::build(&docs);
        let results = index.search("common", 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn save_and_load_round_trip() {
        let docs = vec![
            make_doc(10, "alpha", Some("fn alpha()"), &["core"]),
            make_doc(20, "beta", Some("fn beta()"), &["util"]),
        ];

        let index = Bm25Index::build(&docs);

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bm25.json");
        index.save(&path).unwrap();

        let loaded = Bm25Index::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);

        // Search should produce same results after load
        let original_results = index.search("alpha core", 2);
        let loaded_results = loaded.search("alpha core", 2);
        assert_eq!(original_results.len(), loaded_results.len());
        assert_eq!(original_results[0].0, loaded_results[0].0);
        assert!((original_results[0].1 - loaded_results[0].1).abs() < 1e-10);
    }

    #[test]
    fn no_match_returns_empty() {
        let docs = vec![make_doc(1, "hello", None, &["world"])];
        let index = Bm25Index::build(&docs);
        let results = index.search("zzzzz_nonexistent", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn case_insensitive_search() {
        let docs = vec![make_doc(1, "HttpRequest", Some("struct HttpRequest"), &["http"])];
        let index = Bm25Index::build(&docs);

        let results = index.search("httprequest", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }
}
