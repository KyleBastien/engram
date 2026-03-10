use serde::{Deserialize, Serialize};

/// The storage tier for a conversation snapshot.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotTier {
    Active,
    Compressed,
    Archived,
}

/// A conversation snapshot capturing session context at a point in time.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub session_id: String,
    pub summary: String,
    pub key_context: Vec<String>,
    pub full_transcript: String,
    pub created_at: String,
    pub tier: SnapshotTier,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_tier_variants() {
        let active = SnapshotTier::Active;
        let compressed = SnapshotTier::Compressed;
        let archived = SnapshotTier::Archived;

        let json = serde_json::to_string(&active).unwrap();
        assert_eq!(json, "\"active\"");
        let json = serde_json::to_string(&compressed).unwrap();
        assert_eq!(json, "\"compressed\"");
        let json = serde_json::to_string(&archived).unwrap();
        assert_eq!(json, "\"archived\"");
    }

    #[test]
    fn active_snapshot_yaml_round_trip() {
        let snapshot = Snapshot {
            session_id: "session-001".to_string(),
            summary: "Implemented snapshot types for engram-core".to_string(),
            key_context: vec![
                "Added SnapshotTier enum".to_string(),
                "Added Snapshot struct".to_string(),
            ],
            full_transcript: "User: Add snapshot types\nAssistant: Done.".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
            tier: SnapshotTier::Active,
        };

        let yaml = serde_yaml::to_string(&snapshot).expect("serialize to YAML");
        assert!(yaml.contains("session_id: session-001"));
        assert!(yaml.contains("tier: active"));

        let deserialized: Snapshot = serde_yaml::from_str(&yaml).expect("deserialize from YAML");
        assert_eq!(snapshot, deserialized);
    }

    #[test]
    fn snapshot_all_tiers_yaml_round_trip() {
        for tier in [
            SnapshotTier::Active,
            SnapshotTier::Compressed,
            SnapshotTier::Archived,
        ] {
            let snapshot = Snapshot {
                session_id: "session-002".to_string(),
                summary: "Test".to_string(),
                key_context: vec![],
                full_transcript: String::new(),
                created_at: "2026-03-09T00:00:00Z".to_string(),
                tier: tier.clone(),
            };

            let yaml = serde_yaml::to_string(&snapshot).expect("serialize");
            let deserialized: Snapshot = serde_yaml::from_str(&yaml).expect("deserialize");
            assert_eq!(snapshot.tier, deserialized.tier);
        }
    }

    #[test]
    fn snapshot_derives() {
        let snapshot = Snapshot {
            session_id: "s1".to_string(),
            summary: "sum".to_string(),
            key_context: vec!["ctx".to_string()],
            full_transcript: "transcript".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
            tier: SnapshotTier::Active,
        };

        // Debug
        let debug = format!("{:?}", snapshot);
        assert!(debug.contains("Snapshot"));

        // Clone
        let cloned = snapshot.clone();
        assert_eq!(snapshot, cloned);
    }
}
