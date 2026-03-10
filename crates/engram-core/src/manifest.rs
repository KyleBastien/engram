use serde::{Deserialize, Serialize};

/// Global index manifest describing the state of the indexed data.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Manifest {
    pub chunk_count: usize,
    pub last_indexed_commit: Option<String>,
    pub model_name: String,
    pub dimensions: usize,
    pub source_repos: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> Manifest {
        Manifest {
            chunk_count: 42,
            last_indexed_commit: Some("abc123def456".to_string()),
            model_name: "nomic-embed-text".to_string(),
            dimensions: 768,
            source_repos: vec![
                "/home/user/project-a".to_string(),
                "/home/user/project-b".to_string(),
            ],
            created_at: "2026-03-09T00:00:00Z".to_string(),
            updated_at: "2026-03-09T12:00:00Z".to_string(),
        }
    }

    #[test]
    fn json_round_trip() {
        let manifest = sample_manifest();
        let json = serde_json::to_string_pretty(&manifest).expect("serialize");
        let deserialized: Manifest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(manifest.chunk_count, deserialized.chunk_count);
        assert_eq!(manifest.last_indexed_commit, deserialized.last_indexed_commit);
        assert_eq!(manifest.model_name, deserialized.model_name);
        assert_eq!(manifest.dimensions, deserialized.dimensions);
        assert_eq!(manifest.source_repos, deserialized.source_repos);
        assert_eq!(manifest.created_at, deserialized.created_at);
        assert_eq!(manifest.updated_at, deserialized.updated_at);
    }

    #[test]
    fn manifest_json_file_format() {
        let manifest = sample_manifest();
        let json = serde_json::to_string_pretty(&manifest).unwrap();

        // Verify it contains expected field names
        assert!(json.contains("\"chunk_count\""));
        assert!(json.contains("\"last_indexed_commit\""));
        assert!(json.contains("\"model_name\""));
        assert!(json.contains("\"dimensions\""));
        assert!(json.contains("\"source_repos\""));
        assert!(json.contains("\"created_at\""));
        assert!(json.contains("\"updated_at\""));
    }

    #[test]
    fn manifest_with_no_indexed_commit() {
        let mut manifest = sample_manifest();
        manifest.last_indexed_commit = None;

        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: Manifest = serde_json::from_str(&json).unwrap();
        assert!(deserialized.last_indexed_commit.is_none());
    }

    #[test]
    fn manifest_empty_source_repos() {
        let mut manifest = sample_manifest();
        manifest.source_repos = vec![];

        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: Manifest = serde_json::from_str(&json).unwrap();
        assert!(deserialized.source_repos.is_empty());
    }
}
