use serde::{Deserialize, Serialize};

/// The mode for a benchmark session.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMode {
    Baseline,
    Assisted,
}

/// The outcome of a benchmarked task.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TaskOutcome {
    Success,
    Failure,
    Partial,
}

/// A benchmark session tracking a single task run.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BenchmarkSession {
    pub session_id: String,
    pub mode: BenchmarkMode,
    pub task_description: String,
    pub started_at: String,
    #[serde(default)]
    pub ended_at: Option<String>,
    #[serde(default)]
    pub task_outcome: Option<TaskOutcome>,
    #[serde(default)]
    pub notes: Option<String>,
}

/// A single event recorded during a benchmark session.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BenchmarkEvent {
    pub timestamp: String,
    pub event_type: String,
    pub query: String,
    pub tokens_used: u64,
    pub files_read: u64,
    pub chunks_returned: u64,
    #[serde(default)]
    pub hit: Option<bool>,
}

/// Computed metrics from a benchmark session.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BenchmarkReport {
    pub session_id: String,
    pub mode: BenchmarkMode,
    pub token_efficiency: f64,
    pub retrieval_precision: f64,
    pub retrieval_recall: f64,
    pub time_to_first_edit_ms: u64,
    pub file_read_count: u64,
    pub search_to_read_ratio: f64,
    pub context_waste_tokens: u64,
}

/// Comparison of a single metric between baseline and assisted runs.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MetricComparison {
    pub metric_name: String,
    pub baseline_value: f64,
    pub assisted_value: f64,
    pub absolute_diff: f64,
    pub percentage_diff: f64,
    pub assisted_better: bool,
}

/// Comparison report between a baseline and assisted benchmark run.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ComparisonReport {
    pub baseline_session_id: String,
    pub assisted_session_id: String,
    pub metrics: Vec<MetricComparison>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_session_json_round_trip() {
        let session = BenchmarkSession {
            session_id: "bench-001".to_string(),
            mode: BenchmarkMode::Baseline,
            task_description: "Fix the login bug".to_string(),
            started_at: "2026-03-10T10:00:00Z".to_string(),
            ended_at: Some("2026-03-10T10:30:00Z".to_string()),
            task_outcome: Some(TaskOutcome::Success),
            notes: Some("Completed without search".to_string()),
        };

        let json = serde_json::to_string(&session).expect("serialize");
        let deserialized: BenchmarkSession = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(session.session_id, deserialized.session_id);
        assert_eq!(session.mode, deserialized.mode);
        assert_eq!(session.task_description, deserialized.task_description);
        assert_eq!(session.ended_at, deserialized.ended_at);
        assert_eq!(session.task_outcome, deserialized.task_outcome);
        assert_eq!(session.notes, deserialized.notes);
    }

    #[test]
    fn benchmark_session_optional_fields_default() {
        let json = r#"{
            "session_id": "bench-002",
            "mode": "assisted",
            "task_description": "Add feature X",
            "started_at": "2026-03-10T10:00:00Z"
        }"#;

        let session: BenchmarkSession = serde_json::from_str(json).expect("deserialize");
        assert_eq!(session.session_id, "bench-002");
        assert_eq!(session.mode, BenchmarkMode::Assisted);
        assert!(session.ended_at.is_none());
        assert!(session.task_outcome.is_none());
        assert!(session.notes.is_none());
    }

    #[test]
    fn benchmark_event_json_round_trip() {
        let event = BenchmarkEvent {
            timestamp: "2026-03-10T10:05:00Z".to_string(),
            event_type: "search".to_string(),
            query: "login handler".to_string(),
            tokens_used: 150,
            files_read: 3,
            chunks_returned: 5,
            hit: Some(true),
        };

        let json = serde_json::to_string(&event).expect("serialize");
        let deserialized: BenchmarkEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(event.timestamp, deserialized.timestamp);
        assert_eq!(event.tokens_used, deserialized.tokens_used);
        assert_eq!(event.hit, deserialized.hit);
    }

    #[test]
    fn benchmark_event_optional_hit() {
        let json = r#"{
            "timestamp": "2026-03-10T10:05:00Z",
            "event_type": "search",
            "query": "auth module",
            "tokens_used": 100,
            "files_read": 2,
            "chunks_returned": 3
        }"#;

        let event: BenchmarkEvent = serde_json::from_str(json).expect("deserialize");
        assert!(event.hit.is_none());
    }

    #[test]
    fn benchmark_report_json_round_trip() {
        let report = BenchmarkReport {
            session_id: "bench-001".to_string(),
            mode: BenchmarkMode::Assisted,
            token_efficiency: 0.85,
            retrieval_precision: 0.9,
            retrieval_recall: 0.75,
            time_to_first_edit_ms: 5000,
            file_read_count: 12,
            search_to_read_ratio: 0.6,
            context_waste_tokens: 200,
        };

        let json = serde_json::to_string(&report).expect("serialize");
        let deserialized: BenchmarkReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(report.session_id, deserialized.session_id);
        assert_eq!(report.mode, deserialized.mode);
        assert!((report.token_efficiency - deserialized.token_efficiency).abs() < f64::EPSILON);
        assert!((report.retrieval_precision - deserialized.retrieval_precision).abs() < f64::EPSILON);
        assert_eq!(report.time_to_first_edit_ms, deserialized.time_to_first_edit_ms);
        assert_eq!(report.context_waste_tokens, deserialized.context_waste_tokens);
    }

    #[test]
    fn benchmark_mode_serialization() {
        let baseline = BenchmarkMode::Baseline;
        let assisted = BenchmarkMode::Assisted;

        assert_eq!(serde_json::to_string(&baseline).unwrap(), "\"baseline\"");
        assert_eq!(serde_json::to_string(&assisted).unwrap(), "\"assisted\"");
    }

    #[test]
    fn task_outcome_serialization() {
        assert_eq!(serde_json::to_string(&TaskOutcome::Success).unwrap(), "\"success\"");
        assert_eq!(serde_json::to_string(&TaskOutcome::Failure).unwrap(), "\"failure\"");
        assert_eq!(serde_json::to_string(&TaskOutcome::Partial).unwrap(), "\"partial\"");
    }

    #[test]
    fn benchmark_types_derive_debug_clone() {
        let session = BenchmarkSession {
            session_id: "s1".to_string(),
            mode: BenchmarkMode::Baseline,
            task_description: "test".to_string(),
            started_at: "now".to_string(),
            ended_at: None,
            task_outcome: None,
            notes: None,
        };
        let debug = format!("{:?}", session);
        assert!(debug.contains("BenchmarkSession"));
        let _cloned = session.clone();

        let event = BenchmarkEvent {
            timestamp: "now".to_string(),
            event_type: "search".to_string(),
            query: "q".to_string(),
            tokens_used: 0,
            files_read: 0,
            chunks_returned: 0,
            hit: None,
        };
        let debug = format!("{:?}", event);
        assert!(debug.contains("BenchmarkEvent"));
        let _cloned = event.clone();

        let report = BenchmarkReport {
            session_id: "s1".to_string(),
            mode: BenchmarkMode::Assisted,
            token_efficiency: 0.0,
            retrieval_precision: 0.0,
            retrieval_recall: 0.0,
            time_to_first_edit_ms: 0,
            file_read_count: 0,
            search_to_read_ratio: 0.0,
            context_waste_tokens: 0,
        };
        let debug = format!("{:?}", report);
        assert!(debug.contains("BenchmarkReport"));
        let _cloned = report.clone();
    }
}
