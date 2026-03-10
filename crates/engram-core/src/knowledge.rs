use serde::{Deserialize, Serialize};

/// A recorded architectural or design decision.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Decision {
    pub id: String,
    pub title: String,
    pub status: String,
    pub context: String,
    pub decision: String,
    pub consequences: Vec<String>,
    pub related_files: Vec<String>,
    pub contributed_by: String,
    pub created_at: String,
    pub embedding_ref: Option<String>,
}

/// A lesson learned from past experience.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Lesson {
    pub id: String,
    pub title: String,
    pub description: String,
    pub trigger: String,
    pub resolution: String,
    pub related_files: Vec<String>,
    pub contributed_by: String,
    pub created_at: String,
    pub embedding_ref: Option<String>,
}

/// A recognized code or architecture pattern.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Pattern {
    pub id: String,
    pub name: String,
    pub description: String,
    pub examples: Vec<String>,
    pub anti_patterns: Vec<String>,
    pub contributed_by: String,
    pub created_at: String,
    pub embedding_ref: Option<String>,
}

/// A glossary entry defining domain terminology.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GlossaryEntry {
    pub term: String,
    pub definition: String,
    pub context: String,
    pub contributed_by: String,
    pub created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_yaml_round_trip() {
        let decision = Decision {
            id: "DEC-001".to_string(),
            title: "Use HNSW for vector search".to_string(),
            status: "accepted".to_string(),
            context: "Need fast approximate nearest neighbor search".to_string(),
            decision: "Use HNSW index with ef_construction=200".to_string(),
            consequences: vec![
                "Fast query times".to_string(),
                "Higher memory usage".to_string(),
            ],
            related_files: vec!["crates/engram-query/src/hnsw.rs".to_string()],
            contributed_by: "agent-session-1".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
            embedding_ref: Some("emb-001".to_string()),
        };

        let yaml = serde_yaml::to_string(&decision).expect("serialize to YAML");
        let deserialized: Decision = serde_yaml::from_str(&yaml).expect("deserialize from YAML");
        assert_eq!(decision, deserialized);
    }

    #[test]
    fn lesson_yaml_round_trip() {
        let lesson = Lesson {
            id: "LES-001".to_string(),
            title: "Always scope git2 borrows".to_string(),
            description: "git2 Repository borrows must be scoped before moving".to_string(),
            trigger: "Borrow checker error when moving Repository".to_string(),
            resolution: "Scope borrows in a block before moving".to_string(),
            related_files: vec!["crates/engram-store/src/git.rs".to_string()],
            contributed_by: "agent-session-2".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
            embedding_ref: None,
        };

        let yaml = serde_yaml::to_string(&lesson).expect("serialize to YAML");
        let deserialized: Lesson = serde_yaml::from_str(&yaml).expect("deserialize from YAML");
        assert_eq!(lesson, deserialized);
    }

    #[test]
    fn pattern_yaml_round_trip() {
        let pattern = Pattern {
            id: "PAT-001".to_string(),
            name: "One module per concept".to_string(),
            description: "Each concept gets its own module file".to_string(),
            examples: vec![
                "error.rs for error types".to_string(),
                "chunk.rs for chunk types".to_string(),
            ],
            anti_patterns: vec!["Putting all types in lib.rs".to_string()],
            contributed_by: "agent-session-3".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
            embedding_ref: Some("emb-003".to_string()),
        };

        let yaml = serde_yaml::to_string(&pattern).expect("serialize to YAML");
        let deserialized: Pattern = serde_yaml::from_str(&yaml).expect("deserialize from YAML");
        assert_eq!(pattern, deserialized);
    }

    #[test]
    fn glossary_entry_yaml_round_trip() {
        let entry = GlossaryEntry {
            term: "chunk".to_string(),
            definition: "A semantic unit of code extracted by tree-sitter".to_string(),
            context: "Used throughout engram for indexing and search".to_string(),
            contributed_by: "agent-session-4".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
        };

        let yaml = serde_yaml::to_string(&entry).expect("serialize to YAML");
        let deserialized: GlossaryEntry =
            serde_yaml::from_str(&yaml).expect("deserialize from YAML");
        assert_eq!(entry, deserialized);
    }

    #[test]
    fn decision_optional_embedding_ref() {
        let decision = Decision {
            id: "DEC-002".to_string(),
            title: "No embedding yet".to_string(),
            status: "proposed".to_string(),
            context: "Early stage".to_string(),
            decision: "TBD".to_string(),
            consequences: vec![],
            related_files: vec![],
            contributed_by: "agent".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
            embedding_ref: None,
        };

        let yaml = serde_yaml::to_string(&decision).expect("serialize");
        let deserialized: Decision = serde_yaml::from_str(&yaml).expect("deserialize");
        assert!(deserialized.embedding_ref.is_none());
    }
}
