use engram_core::ChunkMetadata;
use std::collections::HashMap;

/// In-memory metadata indexes for direct O(1) lookups by chunk ID, file path,
/// symbol name, tag, and source repo.
pub struct MetadataIndex {
    by_id: HashMap<String, ChunkMetadata>,
    by_file: HashMap<String, Vec<ChunkMetadata>>,
    by_symbol: HashMap<String, Vec<ChunkMetadata>>,
    by_tag: HashMap<String, Vec<ChunkMetadata>>,
    by_repo: HashMap<String, Vec<ChunkMetadata>>,
}

impl MetadataIndex {
    /// Build all lookup indexes from a slice of chunk metadata.
    ///
    /// File path is extracted from chunk_id format: `{source}#{path}#{name}`.
    /// Symbol name uses the chunk's `name` field.
    pub fn build(chunks: &[ChunkMetadata]) -> Self {
        let mut by_id = HashMap::new();
        let mut by_file: HashMap<String, Vec<ChunkMetadata>> = HashMap::new();
        let mut by_symbol: HashMap<String, Vec<ChunkMetadata>> = HashMap::new();
        let mut by_tag: HashMap<String, Vec<ChunkMetadata>> = HashMap::new();
        let mut by_repo: HashMap<String, Vec<ChunkMetadata>> = HashMap::new();

        for chunk in chunks {
            by_id.insert(chunk.chunk_id.clone(), chunk.clone());

            let (repo, file_path) = extract_repo_file(&chunk.chunk_id);
            by_file
                .entry(file_path)
                .or_default()
                .push(chunk.clone());

            by_repo
                .entry(repo)
                .or_default()
                .push(chunk.clone());

            by_symbol
                .entry(chunk.name.clone())
                .or_default()
                .push(chunk.clone());

            for tag in &chunk.tags {
                by_tag
                    .entry(tag.clone())
                    .or_default()
                    .push(chunk.clone());
            }
        }

        Self {
            by_id,
            by_file,
            by_symbol,
            by_tag,
            by_repo,
        }
    }

    /// Look up a single chunk by its unique chunk_id.
    pub fn lookup_by_id(&self, chunk_id: &str) -> Option<&ChunkMetadata> {
        self.by_id.get(chunk_id)
    }

    /// Look up all chunks belonging to a file path.
    pub fn lookup_by_file(&self, file_path: &str) -> Option<&[ChunkMetadata]> {
        self.by_file.get(file_path).map(|v| v.as_slice())
    }

    /// Look up all chunks with a given symbol name.
    pub fn lookup_by_symbol(&self, symbol_name: &str) -> Option<&[ChunkMetadata]> {
        self.by_symbol.get(symbol_name).map(|v| v.as_slice())
    }

    /// Look up all chunks tagged with a given tag.
    pub fn lookup_by_tag(&self, tag: &str) -> Option<&[ChunkMetadata]> {
        self.by_tag.get(tag).map(|v| v.as_slice())
    }

    /// Look up all chunks belonging to a source repo.
    pub fn lookup_by_repo(&self, repo: &str) -> Option<&[ChunkMetadata]> {
        self.by_repo.get(repo).map(|v| v.as_slice())
    }
}

/// Extract the source repo and file path components from a chunk_id.
///
/// chunk_id format: `{source_name}#{relative_path}#{chunk_name}`
/// Returns (source_name, relative_path).
/// If the format doesn't match, returns ("unknown", full chunk_id) as fallback.
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

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::ChunkKind;

    fn make_chunk(id: &str, name: &str, kind: ChunkKind, tags: Vec<&str>) -> ChunkMetadata {
        ChunkMetadata {
            chunk_id: id.to_string(),
            kind,
            name: name.to_string(),
            signature: Some(format!("fn {name}()")),
            start_line: 1,
            end_line: 10,
            content_hash: "abc123".to_string(),
            tags: tags.into_iter().map(String::from).collect(),
            indexed_at: "2026-03-09T00:00:00Z".to_string(),
            source_commit: "deadbeef".to_string(),
            embedding_offset: 0,
        }
    }

    #[test]
    fn build_and_lookup_by_id() {
        let chunks = vec![
            make_chunk("repo#src/lib.rs#foo", "foo", ChunkKind::Function, vec![]),
            make_chunk("repo#src/lib.rs#bar", "bar", ChunkKind::Function, vec![]),
        ];
        let index = MetadataIndex::build(&chunks);

        let result = index.lookup_by_id("repo#src/lib.rs#foo");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "foo");

        assert!(index.lookup_by_id("nonexistent").is_none());
    }

    #[test]
    fn lookup_by_file_groups_chunks() {
        let chunks = vec![
            make_chunk("repo#src/lib.rs#foo", "foo", ChunkKind::Function, vec![]),
            make_chunk("repo#src/lib.rs#bar", "bar", ChunkKind::Function, vec![]),
            make_chunk("repo#src/main.rs#main", "main", ChunkKind::Function, vec![]),
        ];
        let index = MetadataIndex::build(&chunks);

        let lib_chunks = index.lookup_by_file("src/lib.rs").unwrap();
        assert_eq!(lib_chunks.len(), 2);

        let main_chunks = index.lookup_by_file("src/main.rs").unwrap();
        assert_eq!(main_chunks.len(), 1);

        assert!(index.lookup_by_file("nonexistent.rs").is_none());
    }

    #[test]
    fn lookup_by_symbol_groups_across_files() {
        let chunks = vec![
            make_chunk("repo#src/a.rs#new", "new", ChunkKind::Function, vec![]),
            make_chunk("repo#src/b.rs#new", "new", ChunkKind::Function, vec![]),
            make_chunk("repo#src/a.rs#drop", "drop", ChunkKind::Function, vec![]),
        ];
        let index = MetadataIndex::build(&chunks);

        let new_chunks = index.lookup_by_symbol("new").unwrap();
        assert_eq!(new_chunks.len(), 2);

        let drop_chunks = index.lookup_by_symbol("drop").unwrap();
        assert_eq!(drop_chunks.len(), 1);

        assert!(index.lookup_by_symbol("nonexistent").is_none());
    }

    #[test]
    fn lookup_by_tag() {
        let chunks = vec![
            make_chunk(
                "repo#src/lib.rs#foo",
                "foo",
                ChunkKind::Function,
                vec!["async", "public"],
            ),
            make_chunk(
                "repo#src/lib.rs#bar",
                "bar",
                ChunkKind::Function,
                vec!["public"],
            ),
        ];
        let index = MetadataIndex::build(&chunks);

        let public_chunks = index.lookup_by_tag("public").unwrap();
        assert_eq!(public_chunks.len(), 2);

        let async_chunks = index.lookup_by_tag("async").unwrap();
        assert_eq!(async_chunks.len(), 1);

        assert!(index.lookup_by_tag("nonexistent").is_none());
    }

    #[test]
    fn empty_chunks_builds_empty_index() {
        let index = MetadataIndex::build(&[]);
        assert!(index.lookup_by_id("anything").is_none());
        assert!(index.lookup_by_file("anything").is_none());
        assert!(index.lookup_by_symbol("anything").is_none());
        assert!(index.lookup_by_tag("anything").is_none());
        assert!(index.lookup_by_repo("anything").is_none());
    }

    #[test]
    fn lookup_by_repo_groups_chunks() {
        let chunks = vec![
            make_chunk("repo-a#src/lib.rs#foo", "foo", ChunkKind::Function, vec![]),
            make_chunk("repo-a#src/main.rs#main", "main", ChunkKind::Function, vec![]),
            make_chunk("repo-b#src/lib.rs#bar", "bar", ChunkKind::Function, vec![]),
        ];
        let index = MetadataIndex::build(&chunks);

        let repo_a_chunks = index.lookup_by_repo("repo-a").unwrap();
        assert_eq!(repo_a_chunks.len(), 2);

        let repo_b_chunks = index.lookup_by_repo("repo-b").unwrap();
        assert_eq!(repo_b_chunks.len(), 1);
        assert_eq!(repo_b_chunks[0].name, "bar");

        assert!(index.lookup_by_repo("nonexistent").is_none());
    }

    #[test]
    fn extract_repo_file_from_chunk_id() {
        assert_eq!(
            extract_repo_file("repo#src/lib.rs#foo"),
            ("repo".to_string(), "src/lib.rs".to_string())
        );
        assert_eq!(
            extract_repo_file("my-repo#src/deep/nested/file.ts#MyClass"),
            ("my-repo".to_string(), "src/deep/nested/file.ts".to_string())
        );
    }

    #[test]
    fn extract_repo_file_fallback() {
        // No hash separators — returns ("unknown", full string) as fallback
        assert_eq!(
            extract_repo_file("no-hash"),
            ("unknown".to_string(), "no-hash".to_string())
        );
        // Single hash — no second separator, returns fallback
        assert_eq!(
            extract_repo_file("one#hash"),
            ("unknown".to_string(), "one#hash".to_string())
        );
    }

    #[test]
    fn all_lookups_are_hashmap_based() {
        // This test verifies the index uses HashMap (O(1) lookups)
        // by checking the struct fields exist and work correctly
        // with a large-ish dataset
        let chunks: Vec<ChunkMetadata> = (0..1000)
            .map(|i| {
                make_chunk(
                    &format!("repo#src/file_{}.rs#func_{i}", i % 10),
                    &format!("func_{i}"),
                    ChunkKind::Function,
                    vec!["test"],
                )
            })
            .collect();
        let index = MetadataIndex::build(&chunks);

        // All 1000 chunks individually addressable
        assert!(index.lookup_by_id("repo#src/file_5.rs#func_5").is_some());
        assert!(index.lookup_by_id("repo#src/file_9.rs#func_999").is_some());

        // 10 unique file paths, each with 100 chunks
        let file_chunks = index.lookup_by_file("src/file_0.rs").unwrap();
        assert_eq!(file_chunks.len(), 100);

        // 1000 unique symbol names
        assert!(index.lookup_by_symbol("func_500").is_some());

        // All 1000 chunks tagged "test"
        let tagged = index.lookup_by_tag("test").unwrap();
        assert_eq!(tagged.len(), 1000);
    }

    #[test]
    fn chunk_with_no_tags_not_in_tag_index() {
        let chunks = vec![make_chunk(
            "repo#src/lib.rs#foo",
            "foo",
            ChunkKind::Function,
            vec![],
        )];
        let index = MetadataIndex::build(&chunks);

        // Chunk exists in other indexes
        assert!(index.lookup_by_id("repo#src/lib.rs#foo").is_some());
        // But nothing in tag index since chunk has no tags
        assert!(index.lookup_by_tag("foo").is_none());
    }
}
