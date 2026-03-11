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
    Snapshot,
    Other,
}

/// Index partition for separating chunk kinds into query-time groups.
///
/// Each partition maps to a key range in the HNSW/BM25 indexes, allowing
/// searches to be filtered to specific partitions for parallel querying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Partition {
    /// Code chunks: functions, classes, methods, types, impls, modules (key range 0..DOCS_KEY_OFFSET)
    Code = 0,
    /// Documentation chunks: doc_sections, readmes, comment_blocks (key range DOCS_KEY_OFFSET..KNOWLEDGE_KEY_OFFSET)
    Docs = 1,
    /// Knowledge items: decisions, lessons, patterns (key range KNOWLEDGE_KEY_OFFSET..SNAPSHOT_KEY_OFFSET)
    Knowledge = 2,
    /// Snapshot items (key range SNAPSHOT_KEY_OFFSET..)
    Snapshots = 3,
}

impl ChunkKind {
    /// Returns the index partition this chunk kind belongs to.
    pub fn partition(&self) -> Partition {
        match self {
            ChunkKind::Function
            | ChunkKind::Class
            | ChunkKind::Method
            | ChunkKind::Type
            | ChunkKind::Impl
            | ChunkKind::Module
            | ChunkKind::Other => Partition::Code,
            ChunkKind::DocSection | ChunkKind::Readme | ChunkKind::CommentBlock => Partition::Docs,
            ChunkKind::Knowledge => Partition::Knowledge,
            ChunkKind::Snapshot => Partition::Snapshots,
        }
    }
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
            (ChunkKind::Snapshot, "snapshot_item"),
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
    fn partition_mapping_code() {
        let code_kinds = [
            ChunkKind::Function,
            ChunkKind::Class,
            ChunkKind::Method,
            ChunkKind::Type,
            ChunkKind::Impl,
            ChunkKind::Module,
            ChunkKind::Other,
        ];
        for kind in &code_kinds {
            assert_eq!(kind.partition(), Partition::Code, "{kind:?} should be Code");
        }
    }

    #[test]
    fn partition_mapping_docs() {
        let doc_kinds = [
            ChunkKind::DocSection,
            ChunkKind::Readme,
            ChunkKind::CommentBlock,
        ];
        for kind in &doc_kinds {
            assert_eq!(kind.partition(), Partition::Docs, "{kind:?} should be Docs");
        }
    }

    #[test]
    fn partition_mapping_knowledge_and_snapshots() {
        assert_eq!(ChunkKind::Knowledge.partition(), Partition::Knowledge);
        assert_eq!(ChunkKind::Snapshot.partition(), Partition::Snapshots);
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
