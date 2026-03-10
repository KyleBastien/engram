use serde::{Deserialize, Serialize};

/// The kind of semantic chunk extracted from source code or documentation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkKind {
    Function,
    Class,
    Method,
    Type,
    Impl,
    Module,
    DocSection,
    Readme,
    CommentBlock,
    Knowledge,
    Other,
}

/// Metadata for a single semantic chunk stored in the index.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChunkMetadata {
    pub chunk_id: String,
    pub kind: ChunkKind,
    pub name: String,
    pub signature: Option<String>,
    pub start_line: u32,
    pub end_line: u32,
    pub content_hash: String,
    pub tags: Vec<String>,
    pub indexed_at: String,
    pub source_commit: String,
    pub embedding_offset: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunk(kind: ChunkKind, name: &str) -> ChunkMetadata {
        ChunkMetadata {
            chunk_id: format!("chunk-{name}"),
            kind,
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

    #[test]
    fn round_trip_all_chunk_kinds() {
        let kinds = vec![
            (ChunkKind::Function, "function"),
            (ChunkKind::Class, "class"),
            (ChunkKind::Method, "method"),
            (ChunkKind::Type, "type_def"),
            (ChunkKind::Impl, "impl_block"),
            (ChunkKind::Module, "module"),
            (ChunkKind::DocSection, "doc_section"),
            (ChunkKind::Readme, "readme"),
            (ChunkKind::CommentBlock, "comment_block"),
            (ChunkKind::Knowledge, "knowledge_item"),
            (ChunkKind::Other, "other"),
        ];

        for (kind, name) in kinds {
            let chunk = sample_chunk(kind, name);
            let json = serde_json::to_string(&chunk).expect("serialize");
            let deserialized: ChunkMetadata =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(chunk, deserialized, "round-trip failed for {name}");
        }
    }

    #[test]
    fn jsonl_round_trip() {
        let chunks = vec![
            sample_chunk(ChunkKind::Function, "foo"),
            sample_chunk(ChunkKind::Class, "Bar"),
            sample_chunk(ChunkKind::Other, "misc"),
        ];

        // Write as JSONL (one JSON object per line)
        let jsonl: String = chunks
            .iter()
            .map(|c| serde_json::to_string(c).unwrap())
            .collect::<Vec<_>>()
            .join("\n");

        // Read back from JSONL
        let parsed: Vec<ChunkMetadata> = jsonl
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();

        assert_eq!(chunks, parsed);
    }

    #[test]
    fn optional_signature() {
        let mut chunk = sample_chunk(ChunkKind::Module, "mod");
        chunk.signature = None;

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: ChunkMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk, deserialized);
        assert!(deserialized.signature.is_none());
    }
}
