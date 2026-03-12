use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use engram_core::{
    BenchmarkEvent, BenchmarkMode, BenchmarkReport, BenchmarkSession, ComparisonReport,
    MetricComparison, TaskOutcome,
};

struct ActiveSession {
    session: BenchmarkSession,
    events: Vec<BenchmarkEvent>,
}

/// Manages benchmark sessions for measuring engram effectiveness.
///
/// Supports multiple concurrent sessions via interior mutability.
pub struct BenchmarkHarness {
    store_root: PathBuf,
    sessions: Mutex<HashMap<String, ActiveSession>>,
}

impl Default for BenchmarkHarness {
    fn default() -> Self {
        Self::new(PathBuf::from("."))
    }
}

impl BenchmarkHarness {
    pub fn new(store_root: PathBuf) -> Self {
        Self {
            store_root,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Start a new benchmark session. Returns the session_id.
    pub fn start(&self, mode: BenchmarkMode, task_description: String) -> String {
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = BenchmarkSession {
            session_id: session_id.clone(),
            mode,
            task_description,
            started_at: now_iso8601(),
            ended_at: None,
            task_outcome: None,
            notes: None,
        };
        let active = ActiveSession {
            session,
            events: Vec::new(),
        };
        self.sessions
            .lock()
            .expect("session lock poisoned")
            .insert(session_id.clone(), active);
        session_id
    }

    /// Log an event to an active session.
    ///
    /// Panics if session_id does not exist.
    pub fn log_event(&self, session_id: &str, event: BenchmarkEvent) {
        let mut sessions = self.sessions.lock().expect("session lock poisoned");
        let active = sessions
            .get_mut(session_id)
            .unwrap_or_else(|| panic!("no active session with id: {session_id}"));
        active.events.push(event);
    }

    /// End a session, compute metrics, write the report, and return it.
    ///
    /// Panics if session_id does not exist.
    pub fn end(
        &self,
        session_id: &str,
        outcome: TaskOutcome,
        notes: Option<String>,
    ) -> BenchmarkReport {
        let active = {
            let mut sessions = self.sessions.lock().expect("session lock poisoned");
            sessions
                .remove(session_id)
                .unwrap_or_else(|| panic!("no active session with id: {session_id}"))
        };

        let mut session = active.session;
        session.ended_at = Some(now_iso8601());
        session.task_outcome = Some(outcome);
        session.notes = notes;

        let report = compute_metrics(&active.events, &session);
        self.write_report(&report, &session, &active.events);
        report
    }

    fn write_report(
        &self,
        report: &BenchmarkReport,
        session: &BenchmarkSession,
        events: &[BenchmarkEvent],
    ) {
        let runs_dir = self.store_root.join("metrics").join("runs");
        fs::create_dir_all(&runs_dir).expect("failed to create metrics/runs directory");

        let timestamp = sanitize_timestamp(&session.started_at);
        let filename = format!("{}_{}.jsonl", timestamp, session.session_id);
        let path = runs_dir.join(filename);

        let mut lines = Vec::new();
        lines.push(serde_json::to_string(session).expect("serialize session"));
        for event in events {
            lines.push(serde_json::to_string(event).expect("serialize event"));
        }
        lines.push(serde_json::to_string(report).expect("serialize report"));

        let content = lines.join("\n") + "\n";
        fs::write(path, content).expect("failed to write benchmark report");
    }
}

/// A stored benchmark run loaded from disk.
#[derive(Debug, Clone)]
pub struct StoredRun {
    pub session: BenchmarkSession,
    pub events: Vec<BenchmarkEvent>,
    pub report: BenchmarkReport,
    pub filename: String,
}

/// List all stored benchmark runs, sorted by filename (most recent last).
pub fn list_runs(store_root: &std::path::Path) -> Vec<StoredRun> {
    let runs_dir = store_root.join("metrics").join("runs");
    let entries = match fs::read_dir(&runs_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "jsonl")
        })
        .collect();

    paths.sort_by_key(|e| e.file_name());

    paths
        .into_iter()
        .filter_map(|entry| load_run_from_path(&entry.path(), &entry.file_name().to_string_lossy()))
        .collect()
}

/// Load a single run from a JSONL file path.
fn load_run_from_path(path: &std::path::Path, filename: &str) -> Option<StoredRun> {
    let content = fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 2 {
        return None;
    }
    let session: BenchmarkSession = serde_json::from_str(lines[0]).ok()?;
    let report: BenchmarkReport = serde_json::from_str(lines[lines.len() - 1]).ok()?;
    let events: Vec<BenchmarkEvent> = lines[1..lines.len() - 1]
        .iter()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    Some(StoredRun {
        session,
        events,
        report,
        filename: filename.to_string(),
    })
}

/// Format a benchmark report as a human-readable table.
pub fn format_report_table(report: &BenchmarkReport, session: &BenchmarkSession) -> String {
    let mut out = String::new();
    out.push_str(&format!("Session:   {}\n", report.session_id));
    out.push_str(&format!("Mode:      {:?}\n", report.mode));
    out.push_str(&format!("Task:      {}\n", session.task_description));
    out.push_str(&format!("Started:   {}\n", session.started_at));
    if let Some(ref ended) = session.ended_at {
        out.push_str(&format!("Ended:     {ended}\n"));
    }
    if let Some(ref outcome) = session.task_outcome {
        out.push_str(&format!("Outcome:   {outcome:?}\n"));
    }
    out.push('\n');
    out.push_str(&format!("{:<25} {}\n", "Metric", "Value"));
    out.push_str(&format!("{:<25} {}\n", "-------------------------", "----------"));
    out.push_str(&format!("{:<25} {:.2}\n", "Token Efficiency", report.token_efficiency));
    out.push_str(&format!("{:<25} {:.4}\n", "Retrieval Precision", report.retrieval_precision));
    out.push_str(&format!("{:<25} {:.4}\n", "Retrieval Recall", report.retrieval_recall));
    out.push_str(&format!("{:<25} {}ms\n", "Time to First Edit", report.time_to_first_edit_ms));
    out.push_str(&format!("{:<25} {}\n", "File Read Count", report.file_read_count));
    out.push_str(&format!("{:<25} {:.4}\n", "Search-to-Read Ratio", report.search_to_read_ratio));
    out.push_str(&format!("{:<25} {}\n", "Context Waste Tokens", report.context_waste_tokens));
    out
}

/// Format a comparison report as a human-readable side-by-side table.
pub fn format_comparison_table(report: &ComparisonReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("Baseline:  {}\n", report.baseline_session_id));
    out.push_str(&format!("Assisted:  {}\n", report.assisted_session_id));
    out.push('\n');
    out.push_str(&format!(
        "{:<25} {:>12} {:>12} {:>10} {:>8}\n",
        "Metric", "Baseline", "Assisted", "Diff", "Better?"
    ));
    out.push_str(&format!(
        "{:<25} {:>12} {:>12} {:>10} {:>8}\n",
        "-------------------------", "------------", "------------", "----------", "--------"
    ));
    for m in &report.metrics {
        let better = if m.assisted_better { "  yes" } else { "   no" };
        out.push_str(&format!(
            "{:<25} {:>12.2} {:>12.2} {:>+9.2}% {}\n",
            m.metric_name, m.baseline_value, m.assisted_value, m.percentage_diff, better
        ));
    }
    out
}

/// Format a listing of all runs as a human-readable table.
pub fn format_runs_list(runs: &[StoredRun]) -> String {
    if runs.is_empty() {
        return "No benchmark runs found.\n".to_string();
    }
    let mut out = String::new();
    out.push_str(&format!(
        "{:<38} {:<12} {:<25} {}\n",
        "Session ID", "Mode", "Started", "Task"
    ));
    out.push_str(&format!(
        "{:<38} {:<12} {:<25} {}\n",
        "--------------------------------------", "------------", "-------------------------", "----"
    ));
    for run in runs {
        let mode = format!("{:?}", run.report.mode);
        let task = if run.session.task_description.len() > 40 {
            format!("{}...", &run.session.task_description[..37])
        } else {
            run.session.task_description.clone()
        };
        out.push_str(&format!(
            "{:<38} {:<12} {:<25} {}\n",
            run.session.session_id, mode, run.session.started_at, task
        ));
    }
    out
}

/// Compute metrics from a list of events and a completed session.
pub fn compute_metrics(events: &[BenchmarkEvent], session: &BenchmarkSession) -> BenchmarkReport {
    let token_efficiency: f64 = events.iter().map(|e| e.tokens_used as f64).sum();

    let events_with_hit: Vec<_> = events.iter().filter(|e| e.hit.is_some()).collect();
    let retrieval_precision = if events_with_hit.is_empty() {
        0.0
    } else {
        let hits = events_with_hit
            .iter()
            .filter(|e| e.hit.is_some_and(|h| h))
            .count();
        hits as f64 / events_with_hit.len() as f64
    };

    let file_read_count: u64 = events.iter().map(|e| e.files_read).sum();

    let search_count = events
        .iter()
        .filter(|e| e.event_type == "search")
        .count() as f64;
    let search_to_read_ratio = if file_read_count == 0 {
        0.0
    } else {
        search_count / file_read_count as f64
    };

    let time_to_first_edit_ms = events
        .iter()
        .find(|e| e.event_type == "edit")
        .map(|_| 0_u64)
        .unwrap_or(0);

    let context_waste_tokens = 0_u64;

    BenchmarkReport {
        session_id: session.session_id.clone(),
        mode: session.mode.clone(),
        token_efficiency,
        retrieval_precision,
        retrieval_recall: 0.0,
        time_to_first_edit_ms,
        file_read_count,
        search_to_read_ratio,
        context_waste_tokens,
    }
}

/// Compare a baseline report with an assisted report, producing a metric-by-metric comparison.
///
/// For each metric, computes absolute and percentage differences.
/// `assisted_better` is true when the assisted value is an improvement:
/// - Lower is better for: token_efficiency, time_to_first_edit_ms, file_read_count, context_waste_tokens
/// - Higher is better for: retrieval_precision, retrieval_recall, search_to_read_ratio
pub fn compare(baseline: &BenchmarkReport, assisted: &BenchmarkReport) -> ComparisonReport {
    let metrics = vec![
        make_comparison("token_efficiency", baseline.token_efficiency, assisted.token_efficiency, false),
        make_comparison("retrieval_precision", baseline.retrieval_precision, assisted.retrieval_precision, true),
        make_comparison("retrieval_recall", baseline.retrieval_recall, assisted.retrieval_recall, true),
        make_comparison("time_to_first_edit_ms", baseline.time_to_first_edit_ms as f64, assisted.time_to_first_edit_ms as f64, false),
        make_comparison("file_read_count", baseline.file_read_count as f64, assisted.file_read_count as f64, false),
        make_comparison("search_to_read_ratio", baseline.search_to_read_ratio, assisted.search_to_read_ratio, true),
        make_comparison("context_waste_tokens", baseline.context_waste_tokens as f64, assisted.context_waste_tokens as f64, false),
    ];

    ComparisonReport {
        baseline_session_id: baseline.session_id.clone(),
        assisted_session_id: assisted.session_id.clone(),
        metrics,
    }
}

fn make_comparison(name: &str, baseline_value: f64, assisted_value: f64, higher_is_better: bool) -> MetricComparison {
    let absolute_diff = assisted_value - baseline_value;
    let percentage_diff = if baseline_value.abs() < f64::EPSILON {
        0.0
    } else {
        (absolute_diff / baseline_value) * 100.0
    };
    let assisted_better = if higher_is_better {
        assisted_value > baseline_value
    } else {
        assisted_value < baseline_value
    };

    MetricComparison {
        metric_name: name.to_string(),
        baseline_value,
        assisted_value,
        absolute_diff,
        percentage_diff,
        assisted_better,
    }
}

/// Write a comparison report as YAML to the metrics/ directory.
pub fn write_comparison_report(store_root: &std::path::Path, report: &ComparisonReport) {
    let comparisons_dir = store_root.join("metrics").join("comparisons");
    fs::create_dir_all(&comparisons_dir).expect("failed to create metrics/comparisons directory");

    let filename = format!(
        "{}_vs_{}.yaml",
        sanitize_timestamp(&report.baseline_session_id),
        sanitize_timestamp(&report.assisted_session_id),
    );
    let path = comparisons_dir.join(filename);
    let yaml = serde_yaml::to_string(report).expect("serialize comparison report to YAML");
    fs::write(path, yaml).expect("failed to write comparison report");
}

fn now_iso8601() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before epoch");
    let secs = duration.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    // Simple date calculation from days since epoch
    let (year, month, day) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn days_to_ymd(days_since_epoch: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn sanitize_timestamp(ts: &str) -> String {
    ts.replace(':', "-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn read_report_file(path: &Path) -> (BenchmarkSession, Vec<BenchmarkEvent>, BenchmarkReport) {
        let content = fs::read_to_string(path).expect("read report file");
        let lines: Vec<&str> = content.lines().collect();
        assert!(
            lines.len() >= 2,
            "report file must have at least session + report lines"
        );
        let session: BenchmarkSession =
            serde_json::from_str(lines[0]).expect("deserialize session line");
        let report: BenchmarkReport =
            serde_json::from_str(lines[lines.len() - 1]).expect("deserialize report line");
        let events: Vec<BenchmarkEvent> = lines[1..lines.len() - 1]
            .iter()
            .map(|line| serde_json::from_str(line).expect("deserialize event line"))
            .collect();
        (session, events, report)
    }

    #[test]
    fn start_returns_unique_session_ids() {
        let harness = BenchmarkHarness::default();
        let id1 = harness.start(BenchmarkMode::Baseline, "task 1".to_string());
        let id2 = harness.start(BenchmarkMode::Assisted, "task 2".to_string());
        assert_ne!(id1, id2);
    }

    #[test]
    fn log_event_appends_to_session() {
        let harness = BenchmarkHarness::default();
        let sid = harness.start(BenchmarkMode::Baseline, "test".to_string());

        harness.log_event(
            &sid,
            BenchmarkEvent {
                timestamp: "2026-03-10T10:00:00Z".to_string(),
                event_type: "search".to_string(),
                query: "find foo".to_string(),
                tokens_used: 100,
                files_read: 2,
                chunks_returned: 3,
                hit: Some(true),
            },
        );

        let sessions = harness.sessions.lock().unwrap();
        assert_eq!(sessions[&sid].events.len(), 1);
    }

    #[test]
    fn end_returns_report_and_removes_session() {
        let dir = tempfile::tempdir().unwrap();
        let harness = BenchmarkHarness::new(dir.path().to_path_buf());
        let sid = harness.start(BenchmarkMode::Assisted, "test task".to_string());

        harness.log_event(
            &sid,
            BenchmarkEvent {
                timestamp: "2026-03-10T10:00:00Z".to_string(),
                event_type: "search".to_string(),
                query: "q".to_string(),
                tokens_used: 50,
                files_read: 1,
                chunks_returned: 2,
                hit: Some(true),
            },
        );

        let report = harness.end(&sid, TaskOutcome::Success, Some("good".to_string()));

        assert_eq!(report.session_id, sid);
        assert_eq!(report.mode, BenchmarkMode::Assisted);
        assert!((report.token_efficiency - 50.0).abs() < f64::EPSILON);
        assert!((report.retrieval_precision - 1.0).abs() < f64::EPSILON);

        // Session removed
        let sessions = harness.sessions.lock().unwrap();
        assert!(!sessions.contains_key(&sid));
    }

    #[test]
    fn end_writes_jsonl_report_file() {
        let dir = tempfile::tempdir().unwrap();
        let harness = BenchmarkHarness::new(dir.path().to_path_buf());
        let sid = harness.start(BenchmarkMode::Baseline, "file test".to_string());

        harness.log_event(
            &sid,
            BenchmarkEvent {
                timestamp: "2026-03-10T10:01:00Z".to_string(),
                event_type: "search".to_string(),
                query: "bar".to_string(),
                tokens_used: 75,
                files_read: 3,
                chunks_returned: 1,
                hit: Some(false),
            },
        );

        let report = harness.end(&sid, TaskOutcome::Failure, None);

        // Find the written file
        let runs_dir = dir.path().join("metrics").join("runs");
        let entries: Vec<_> = fs::read_dir(&runs_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);

        let file_name = entries[0].file_name().to_string_lossy().to_string();
        assert!(file_name.ends_with(&format!("{}.jsonl", sid)));

        let (session, events, file_report) = read_report_file(&entries[0].path());
        assert_eq!(session.session_id, sid);
        assert_eq!(events.len(), 1);
        assert_eq!(file_report.session_id, report.session_id);
    }

    #[test]
    fn multiple_concurrent_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let harness = BenchmarkHarness::new(dir.path().to_path_buf());

        let sid1 = harness.start(BenchmarkMode::Baseline, "session 1".to_string());
        let sid2 = harness.start(BenchmarkMode::Assisted, "session 2".to_string());

        harness.log_event(
            &sid1,
            BenchmarkEvent {
                timestamp: "t1".to_string(),
                event_type: "search".to_string(),
                query: "a".to_string(),
                tokens_used: 10,
                files_read: 1,
                chunks_returned: 1,
                hit: Some(true),
            },
        );

        harness.log_event(
            &sid2,
            BenchmarkEvent {
                timestamp: "t2".to_string(),
                event_type: "search".to_string(),
                query: "b".to_string(),
                tokens_used: 20,
                files_read: 2,
                chunks_returned: 2,
                hit: Some(false),
            },
        );

        let r1 = harness.end(&sid1, TaskOutcome::Success, None);
        let r2 = harness.end(&sid2, TaskOutcome::Partial, None);

        assert_eq!(r1.session_id, sid1);
        assert_eq!(r2.session_id, sid2);
        assert!((r1.token_efficiency - 10.0).abs() < f64::EPSILON);
        assert!((r2.token_efficiency - 20.0).abs() < f64::EPSILON);

        // Both files written
        let runs_dir = dir.path().join("metrics").join("runs");
        let entries: Vec<_> = fs::read_dir(&runs_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn compute_metrics_empty_events() {
        let session = BenchmarkSession {
            session_id: "s1".to_string(),
            mode: BenchmarkMode::Baseline,
            task_description: "empty".to_string(),
            started_at: "t".to_string(),
            ended_at: None,
            task_outcome: None,
            notes: None,
        };
        let report = compute_metrics(&[], &session);
        assert!((report.token_efficiency - 0.0).abs() < f64::EPSILON);
        assert!((report.retrieval_precision - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.file_read_count, 0);
    }

    #[test]
    fn compute_metrics_precision_calculation() {
        let session = BenchmarkSession {
            session_id: "s2".to_string(),
            mode: BenchmarkMode::Assisted,
            task_description: "precision".to_string(),
            started_at: "t".to_string(),
            ended_at: None,
            task_outcome: None,
            notes: None,
        };
        let events = vec![
            BenchmarkEvent {
                timestamp: "t1".to_string(),
                event_type: "search".to_string(),
                query: "a".to_string(),
                tokens_used: 100,
                files_read: 2,
                chunks_returned: 3,
                hit: Some(true),
            },
            BenchmarkEvent {
                timestamp: "t2".to_string(),
                event_type: "search".to_string(),
                query: "b".to_string(),
                tokens_used: 50,
                files_read: 1,
                chunks_returned: 2,
                hit: Some(false),
            },
            BenchmarkEvent {
                timestamp: "t3".to_string(),
                event_type: "read".to_string(),
                query: "c".to_string(),
                tokens_used: 25,
                files_read: 1,
                chunks_returned: 0,
                hit: None, // no hit info
            },
        ];
        let report = compute_metrics(&events, &session);
        // token_efficiency = 100 + 50 + 25 = 175
        assert!((report.token_efficiency - 175.0).abs() < f64::EPSILON);
        // precision = 1 hit out of 2 events with hit info = 0.5
        assert!((report.retrieval_precision - 0.5).abs() < f64::EPSILON);
        // file_read_count = 2 + 1 + 1 = 4
        assert_eq!(report.file_read_count, 4);
        // search_to_read_ratio = 2 searches / 4 reads = 0.5
        assert!((report.search_to_read_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn compare_produces_correct_metric_comparisons() {
        let baseline = BenchmarkReport {
            session_id: "baseline-1".to_string(),
            mode: BenchmarkMode::Baseline,
            token_efficiency: 200.0,
            retrieval_precision: 0.4,
            retrieval_recall: 0.3,
            time_to_first_edit_ms: 10000,
            file_read_count: 20,
            search_to_read_ratio: 0.3,
            context_waste_tokens: 500,
        };
        let assisted = BenchmarkReport {
            session_id: "assisted-1".to_string(),
            mode: BenchmarkMode::Assisted,
            token_efficiency: 100.0,
            retrieval_precision: 0.8,
            retrieval_recall: 0.6,
            time_to_first_edit_ms: 5000,
            file_read_count: 10,
            search_to_read_ratio: 0.6,
            context_waste_tokens: 200,
        };

        let report = compare(&baseline, &assisted);
        assert_eq!(report.baseline_session_id, "baseline-1");
        assert_eq!(report.assisted_session_id, "assisted-1");
        assert_eq!(report.metrics.len(), 7);

        // token_efficiency: lower is better, 200 -> 100, assisted is better
        let te = &report.metrics[0];
        assert_eq!(te.metric_name, "token_efficiency");
        assert!((te.absolute_diff - (-100.0)).abs() < f64::EPSILON);
        assert!((te.percentage_diff - (-50.0)).abs() < f64::EPSILON);
        assert!(te.assisted_better);

        // retrieval_precision: higher is better, 0.4 -> 0.8, assisted is better
        let rp = &report.metrics[1];
        assert_eq!(rp.metric_name, "retrieval_precision");
        assert!((rp.absolute_diff - 0.4).abs() < f64::EPSILON);
        assert!((rp.percentage_diff - 100.0).abs() < f64::EPSILON);
        assert!(rp.assisted_better);

        // file_read_count: lower is better, 20 -> 10, assisted is better
        let frc = &report.metrics[4];
        assert!(frc.assisted_better);
    }

    #[test]
    fn compare_marks_worse_metrics_as_not_better() {
        let baseline = BenchmarkReport {
            session_id: "b".to_string(),
            mode: BenchmarkMode::Baseline,
            token_efficiency: 50.0,
            retrieval_precision: 0.9,
            retrieval_recall: 0.8,
            time_to_first_edit_ms: 1000,
            file_read_count: 5,
            search_to_read_ratio: 0.8,
            context_waste_tokens: 10,
        };
        let assisted = BenchmarkReport {
            session_id: "a".to_string(),
            mode: BenchmarkMode::Assisted,
            token_efficiency: 100.0, // worse (higher)
            retrieval_precision: 0.5, // worse (lower)
            retrieval_recall: 0.4,
            time_to_first_edit_ms: 2000,
            file_read_count: 10,
            search_to_read_ratio: 0.4,
            context_waste_tokens: 50,
        };

        let report = compare(&baseline, &assisted);
        // All metrics should be worse
        for m in &report.metrics {
            assert!(!m.assisted_better, "metric {} should not be better", m.metric_name);
        }
    }

    #[test]
    fn compare_handles_zero_baseline() {
        let baseline = BenchmarkReport {
            session_id: "b".to_string(),
            mode: BenchmarkMode::Baseline,
            token_efficiency: 0.0,
            retrieval_precision: 0.0,
            retrieval_recall: 0.0,
            time_to_first_edit_ms: 0,
            file_read_count: 0,
            search_to_read_ratio: 0.0,
            context_waste_tokens: 0,
        };
        let assisted = BenchmarkReport {
            session_id: "a".to_string(),
            mode: BenchmarkMode::Assisted,
            token_efficiency: 100.0,
            retrieval_precision: 0.5,
            retrieval_recall: 0.0,
            time_to_first_edit_ms: 0,
            file_read_count: 0,
            search_to_read_ratio: 0.0,
            context_waste_tokens: 0,
        };

        let report = compare(&baseline, &assisted);
        // Zero baseline => percentage_diff should be 0.0 (no division by zero)
        for m in &report.metrics {
            assert!(m.percentage_diff.is_finite(), "metric {} has non-finite percentage", m.metric_name);
        }
    }

    #[test]
    fn write_comparison_report_creates_yaml_file() {
        let dir = tempfile::tempdir().unwrap();
        let report = ComparisonReport {
            baseline_session_id: "baseline-1".to_string(),
            assisted_session_id: "assisted-1".to_string(),
            metrics: vec![MetricComparison {
                metric_name: "token_efficiency".to_string(),
                baseline_value: 200.0,
                assisted_value: 100.0,
                absolute_diff: -100.0,
                percentage_diff: -50.0,
                assisted_better: true,
            }],
        };

        write_comparison_report(dir.path(), &report);

        let comparisons_dir = dir.path().join("metrics").join("comparisons");
        let entries: Vec<_> = fs::read_dir(&comparisons_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);

        let content = fs::read_to_string(entries[0].path()).unwrap();
        let deserialized: ComparisonReport =
            serde_yaml::from_str(&content).expect("deserialize YAML");
        assert_eq!(deserialized.baseline_session_id, "baseline-1");
        assert_eq!(deserialized.metrics.len(), 1);
        assert!(deserialized.metrics[0].assisted_better);
    }

    #[test]
    #[should_panic(expected = "no active session")]
    fn log_event_panics_for_unknown_session() {
        let harness = BenchmarkHarness::default();
        harness.log_event(
            "nonexistent",
            BenchmarkEvent {
                timestamp: "t".to_string(),
                event_type: "search".to_string(),
                query: "q".to_string(),
                tokens_used: 0,
                files_read: 0,
                chunks_returned: 0,
                hit: None,
            },
        );
    }

    #[test]
    #[should_panic(expected = "no active session")]
    fn end_panics_for_unknown_session() {
        let harness = BenchmarkHarness::default();
        harness.end("nonexistent", TaskOutcome::Failure, None);
    }
}
