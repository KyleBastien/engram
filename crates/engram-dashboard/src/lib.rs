use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, StatusCode, Uri};
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::Embed;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};

use engram_core::DashboardConfig;

#[derive(Embed)]
#[folder = "src/assets/"]
struct Assets;

/// Events broadcast to dashboard WebSocket clients.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DashboardEvent {
    SearchQuery {
        query: String,
        code_result_count: usize,
        knowledge_result_count: usize,
        search_time_ms: u64,
    },
    KnowledgeWrite {
        kind: String,
        title: String,
    },
    ReindexProgress {
        source: String,
        chunks_indexed: usize,
        total_chunks: usize,
    },
    SessionConnect {
        client_name: String,
    },
    SessionDisconnect {
        client_name: String,
    },
}

/// Broadcast channel for sending dashboard events to all connected WebSocket clients.
///
/// Clone and share between the MCP server (producer) and the dashboard (consumer).
#[derive(Clone)]
pub struct EventBroadcaster {
    tx: broadcast::Sender<DashboardEvent>,
}

impl EventBroadcaster {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Send an event to all connected WebSocket clients.
    /// Returns silently if no clients are connected.
    pub fn broadcast(&self, event: DashboardEvent) {
        let _ = self.tx.send(event);
    }

    fn subscribe(&self) -> broadcast::Receiver<DashboardEvent> {
        self.tx.subscribe()
    }
}

impl Default for EventBroadcaster {
    fn default() -> Self {
        Self::new(256)
    }
}

/// Per-source-repo index statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RepoHealth {
    pub name: String,
    pub chunk_count: usize,
    pub stale_chunks: usize,
    pub last_indexed_commit: String,
    pub last_reindex_time: String,
}

/// A file with stale chunks.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StaleFile {
    pub file: String,
    pub repo: String,
    pub stale_chunks: usize,
    pub total_chunks: usize,
    pub oldest_indexed_at: String,
}

/// Embedding provider status.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderStatus {
    pub name: String,
    pub model: String,
    pub reachable: bool,
}

/// Cache status information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStatus {
    pub status: String,
    pub size_bytes: u64,
}

/// A single knowledge write-back event.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeWriteback {
    pub kind: String,
    pub title: String,
    pub timestamp: String,
}

/// Onboarding status for a source repository.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OnboardingStatus {
    pub source: String,
    pub status: String,
    pub chunks_indexed: usize,
}

/// Snapshot of knowledge activity data served via REST API.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeSnapshot {
    pub items_by_category: std::collections::HashMap<String, usize>,
    pub recent_writebacks: Vec<KnowledgeWriteback>,
    pub onboarding_status: Vec<OnboardingStatus>,
}

/// A single data point for a time-series chart.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimeSeriesPoint {
    pub timestamp: String,
    pub value: f64,
}

/// Snapshot of search analytics data served via REST API.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchAnalyticsSnapshot {
    pub query_frequency: Vec<TimeSeriesPoint>,
    pub avg_relevance: Vec<TimeSeriesPoint>,
    pub cache_hit_rate: f64,
    pub sidecar_hit_rate: f64,
    pub total_queries: usize,
    pub avg_search_time_ms: f64,
}

/// Summary of a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkRunSummary {
    pub session_id: String,
    pub task: String,
    pub started_at: String,
    pub status: String,
    pub baseline_tokens: u64,
    pub assisted_tokens: u64,
    pub token_savings_pct: f64,
    pub duration_ms: u64,
    pub event_count: usize,
}

/// A single data point for token savings trend over time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenSavingsPoint {
    pub session_id: String,
    pub completed_at: String,
    pub savings_pct: f64,
}

/// Snapshot of benchmark data served via REST API.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkSnapshot {
    pub active_sessions: Vec<BenchmarkRunSummary>,
    pub historical_runs: Vec<BenchmarkRunSummary>,
    pub token_savings_trend: Vec<TokenSavingsPoint>,
}

/// Snapshot of index health data served via REST API.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthSnapshot {
    pub total_chunks: usize,
    pub repos: Vec<RepoHealth>,
    pub stale_files: Vec<StaleFile>,
    pub provider: ProviderStatus,
    pub cache: CacheStatus,
    pub boot_time_ms: u64,
    pub store_path: String,
}

/// Shared state for the dashboard: event broadcaster + health data + knowledge data + search analytics + benchmarks.
#[derive(Clone)]
pub struct DashboardState {
    pub broadcaster: EventBroadcaster,
    pub health: Arc<RwLock<HealthSnapshot>>,
    pub knowledge: Arc<RwLock<KnowledgeSnapshot>>,
    pub search_analytics: Arc<RwLock<SearchAnalyticsSnapshot>>,
    pub benchmarks: Arc<RwLock<BenchmarkSnapshot>>,
}

impl DashboardState {
    pub fn new(broadcaster: EventBroadcaster, health: HealthSnapshot) -> Self {
        Self {
            broadcaster,
            health: Arc::new(RwLock::new(health)),
            knowledge: Arc::new(RwLock::new(KnowledgeSnapshot::default())),
            search_analytics: Arc::new(RwLock::new(SearchAnalyticsSnapshot::default())),
            benchmarks: Arc::new(RwLock::new(BenchmarkSnapshot::default())),
        }
    }

    pub fn with_knowledge(
        broadcaster: EventBroadcaster,
        health: HealthSnapshot,
        knowledge: KnowledgeSnapshot,
    ) -> Self {
        Self {
            broadcaster,
            health: Arc::new(RwLock::new(health)),
            knowledge: Arc::new(RwLock::new(knowledge)),
            search_analytics: Arc::new(RwLock::new(SearchAnalyticsSnapshot::default())),
            benchmarks: Arc::new(RwLock::new(BenchmarkSnapshot::default())),
        }
    }
}

/// Serve the dashboard HTTP server on the configured port.
///
/// Serves static HTML/JS/CSS files bundled into the binary via rust-embed.
/// The dashboard is accessible at `http://localhost:{port}/dashboard`.
/// WebSocket endpoint is at `ws://localhost:{port}/ws`.
/// REST API endpoints are at `/api/health`.
pub async fn serve_dashboard(
    config: &DashboardConfig,
    state: DashboardState,
) -> engram_core::Result<()> {
    let app = build_router_with_state(state);
    let addr = format!("0.0.0.0:{}", config.port);

    eprintln!(
        "engram-dashboard: listening on http://localhost:{}/dashboard",
        config.port
    );

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| engram_core::EngramError::Mcp(format!("Failed to bind {addr}: {e}")))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| engram_core::EngramError::Mcp(format!("Dashboard server error: {e}")))?;

    Ok(())
}

/// Build the axum Router for the dashboard (without WebSocket support).
///
/// Exposed publicly for testing.
pub fn build_router() -> Router {
    let dashboard_routes = Router::new()
        .route("/", get(index_handler))
        .fallback(get(static_handler));

    Router::new().nest("/dashboard", dashboard_routes)
}

/// Build the axum Router with WebSocket event broadcasting (legacy).
pub fn build_router_with_events(broadcaster: EventBroadcaster) -> Router {
    let state = DashboardState::new(broadcaster, HealthSnapshot::default());
    build_router_with_state(state)
}

/// Build the axum Router with full dashboard state (events + health data).
pub fn build_router_with_state(state: DashboardState) -> Router {
    let shared = Arc::new(state);

    let dashboard_routes = Router::<Arc<DashboardState>>::new()
        .route("/", get(index_handler))
        .fallback(get(static_handler));

    Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/health", get(health_handler))
        .route("/api/knowledge", get(knowledge_handler))
        .route("/api/search_analytics", get(search_analytics_handler))
        .route("/api/benchmarks", get(benchmarks_handler))
        .nest("/dashboard", dashboard_routes)
        .with_state(shared)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let rx = state.broadcaster.subscribe();
    ws.on_upgrade(|socket| handle_ws_connection(socket, rx))
}

async fn health_handler(
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let snapshot = state.health.read().await;
    Json(snapshot.clone())
}

async fn knowledge_handler(
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let snapshot = state.knowledge.read().await;
    Json(snapshot.clone())
}

async fn search_analytics_handler(
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let snapshot = state.search_analytics.read().await;
    Json(snapshot.clone())
}

async fn benchmarks_handler(
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let snapshot = state.benchmarks.read().await;
    Json(snapshot.clone())
}

async fn handle_ws_connection(
    mut socket: WebSocket,
    mut rx: broadcast::Receiver<DashboardEvent>,
) {
    while let Ok(event) = rx.recv().await {
        if let Ok(json) = serde_json::to_string(&event) {
            if socket.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    }
}

async fn index_handler() -> impl IntoResponse {
    match Assets::get("index.html") {
        Some(content) => Html(content.data.to_vec()).into_response(),
        None => (StatusCode::INTERNAL_SERVER_ERROR, "index.html not found").into_response(),
    }
}

async fn static_handler(uri: Uri) -> impl IntoResponse {
    // Strip prefix to get the asset path — axum nest() strips the /dashboard prefix,
    // leaving paths like /knowledge.html; trim any leading slash to match asset filenames.
    let path = uri
        .path()
        .trim_start_matches("/dashboard/")
        .trim_start_matches('/');

    match Assets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            Response::builder()
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(axum::body::Body::from(content.data.to_vec()))
                .unwrap()
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn start_test_server() -> (String, tokio::task::JoinHandle<()>) {
        let app = build_router();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        (format!("http://{addr}"), handle)
    }

    async fn start_ws_test_server(
        broadcaster: EventBroadcaster,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let app = build_router_with_events(broadcaster);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        (format!("127.0.0.1:{}", addr.port()), handle)
    }

    #[tokio::test]
    async fn test_dashboard_index_served() {
        let (base, handle) = start_test_server().await;
        let client = reqwest::Client::new();

        let resp = client
            .get(format!("{base}/dashboard"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        assert!(body.contains("Engram Dashboard"));

        handle.abort();
    }

    #[tokio::test]
    async fn test_dashboard_unknown_path_returns_404() {
        let (base, handle) = start_test_server().await;
        let client = reqwest::Client::new();

        let resp = client
            .get(format!("{base}/dashboard/nonexistent.js"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        handle.abort();
    }

    #[tokio::test]
    async fn test_build_router_creates_valid_router() {
        let router = build_router();
        // Router should be constructable without panic
        let _ = router;
    }

    #[tokio::test]
    async fn test_serve_dashboard_config() {
        // Verify DashboardConfig defaults
        let config = DashboardConfig::default();
        assert!(config.enabled);
        assert_eq!(config.port, 3200);
    }

    #[tokio::test]
    async fn test_embedded_assets_contain_index() {
        assert!(Assets::get("index.html").is_some());
    }

    #[tokio::test]
    async fn test_event_broadcaster_default() {
        let broadcaster = EventBroadcaster::default();
        // Broadcasting with no subscribers should not panic
        broadcaster.broadcast(DashboardEvent::SessionConnect {
            client_name: "test".to_string(),
        });
    }

    #[tokio::test]
    async fn test_event_serialization_has_type_field() {
        let event = DashboardEvent::SearchQuery {
            query: "find foo".to_string(),
            code_result_count: 3,
            knowledge_result_count: 1,
            search_time_ms: 42,
        };
        let json = serde_json::to_string(&event).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["type"], "search_query");
        assert_eq!(value["query"], "find foo");
        assert_eq!(value["code_result_count"], 3);
    }

    #[tokio::test]
    async fn test_all_event_variants_serialize_with_type() {
        let events: Vec<DashboardEvent> = vec![
            DashboardEvent::SearchQuery {
                query: "q".to_string(),
                code_result_count: 0,
                knowledge_result_count: 0,
                search_time_ms: 0,
            },
            DashboardEvent::KnowledgeWrite {
                kind: "decision".to_string(),
                title: "Use Rust".to_string(),
            },
            DashboardEvent::ReindexProgress {
                source: "repo".to_string(),
                chunks_indexed: 50,
                total_chunks: 100,
            },
            DashboardEvent::SessionConnect {
                client_name: "claude".to_string(),
            },
            DashboardEvent::SessionDisconnect {
                client_name: "claude".to_string(),
            },
        ];

        let expected_types = [
            "search_query",
            "knowledge_write",
            "reindex_progress",
            "session_connect",
            "session_disconnect",
        ];

        for (event, expected_type) in events.into_iter().zip(expected_types.iter()) {
            let json = serde_json::to_string(&event).unwrap();
            let value: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(value["type"].as_str().unwrap(), *expected_type);
        }
    }

    #[tokio::test]
    async fn test_broadcast_received_by_subscriber() {
        let broadcaster = EventBroadcaster::new(16);
        let mut rx = broadcaster.subscribe();

        broadcaster.broadcast(DashboardEvent::SessionConnect {
            client_name: "test-client".to_string(),
        });

        let event = rx.recv().await.unwrap();
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("session_connect"));
        assert!(json.contains("test-client"));
    }

    #[tokio::test]
    async fn test_broadcast_multiple_subscribers() {
        let broadcaster = EventBroadcaster::new(16);
        let mut rx1 = broadcaster.subscribe();
        let mut rx2 = broadcaster.subscribe();

        broadcaster.broadcast(DashboardEvent::SearchQuery {
            query: "hello".to_string(),
            code_result_count: 1,
            knowledge_result_count: 0,
            search_time_ms: 10,
        });

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();

        let j1 = serde_json::to_string(&e1).unwrap();
        let j2 = serde_json::to_string(&e2).unwrap();
        assert_eq!(j1, j2);
        assert!(j1.contains("hello"));
    }

    #[tokio::test]
    async fn test_websocket_receives_events() {
        use futures_util::StreamExt;
        use tokio_tungstenite::connect_async;

        let broadcaster = EventBroadcaster::new(16);
        let (addr, handle) = start_ws_test_server(broadcaster.clone()).await;

        let url = format!("ws://{addr}/ws");
        let (mut ws_stream, _) = connect_async(&url).await.expect("Failed to connect");

        // Give the WebSocket connection time to establish
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        broadcaster.broadcast(DashboardEvent::SearchQuery {
            query: "test query".to_string(),
            code_result_count: 5,
            knowledge_result_count: 2,
            search_time_ms: 15,
        });

        let msg = tokio::time::timeout(std::time::Duration::from_secs(2), ws_stream.next())
            .await
            .expect("Timeout waiting for WS message")
            .expect("Stream ended")
            .expect("WS error");

        let text = msg.into_text().expect("Expected text message");
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["type"], "search_query");
        assert_eq!(value["query"], "test query");
        assert_eq!(value["code_result_count"], 5);

        ws_stream.close(None).await.ok();
        handle.abort();
    }

    #[tokio::test]
    async fn test_websocket_multiple_clients() {
        use futures_util::StreamExt;
        use tokio_tungstenite::connect_async;

        let broadcaster = EventBroadcaster::new(16);
        let (addr, handle) = start_ws_test_server(broadcaster.clone()).await;

        let url = format!("ws://{addr}/ws");
        let (mut ws1, _) = connect_async(&url).await.expect("Client 1 failed");
        let (mut ws2, _) = connect_async(&url).await.expect("Client 2 failed");

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        broadcaster.broadcast(DashboardEvent::SessionConnect {
            client_name: "multi-test".to_string(),
        });

        let timeout = std::time::Duration::from_secs(2);

        let msg1 = tokio::time::timeout(timeout, ws1.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let msg2 = tokio::time::timeout(timeout, ws2.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        let t1 = msg1.into_text().unwrap();
        let t2 = msg2.into_text().unwrap();
        assert_eq!(t1, t2);
        assert!(t1.contains("multi-test"));

        ws1.close(None).await.ok();
        ws2.close(None).await.ok();
        handle.abort();
    }

    #[tokio::test]
    async fn test_build_router_with_events_creates_valid_router() {
        let broadcaster = EventBroadcaster::default();
        let router = build_router_with_events(broadcaster);
        let _ = router;
    }

    async fn start_state_test_server(
        state: DashboardState,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let app = build_router_with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        (format!("http://{addr}"), handle)
    }

    #[tokio::test]
    async fn test_health_api_returns_json() {
        let health = HealthSnapshot {
            total_chunks: 150,
            repos: vec![RepoHealth {
                name: "my-repo".to_string(),
                chunk_count: 150,
                stale_chunks: 10,
                last_indexed_commit: "abc123".to_string(),
                last_reindex_time: "2026-03-10T00:00:00Z".to_string(),
            }],
            stale_files: vec![StaleFile {
                file: "src/main.rs".to_string(),
                repo: "my-repo".to_string(),
                stale_chunks: 3,
                total_chunks: 5,
                oldest_indexed_at: "2026-03-09T00:00:00Z".to_string(),
            }],
            provider: ProviderStatus {
                name: "ollama".to_string(),
                model: "nomic-embed-text".to_string(),
                reachable: true,
            },
            cache: CacheStatus {
                status: "hit".to_string(),
                size_bytes: 1024,
            },
            boot_time_ms: 42,
            store_path: "/tmp/store".to_string(),
        };
        let state = DashboardState::new(EventBroadcaster::default(), health);
        let (base, handle) = start_state_test_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}/api/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["total_chunks"], 150);
        assert_eq!(body["repos"][0]["name"], "my-repo");
        assert_eq!(body["repos"][0]["stale_chunks"], 10);
        assert_eq!(body["stale_files"][0]["file"], "src/main.rs");
        assert_eq!(body["provider"]["name"], "ollama");
        assert_eq!(body["cache"]["status"], "hit");

        handle.abort();
    }

    #[tokio::test]
    async fn test_health_snapshot_default_empty() {
        let snapshot = HealthSnapshot::default();
        assert_eq!(snapshot.total_chunks, 0);
        assert!(snapshot.repos.is_empty());
        assert!(snapshot.stale_files.is_empty());
    }

    #[tokio::test]
    async fn test_dashboard_state_new() {
        let broadcaster = EventBroadcaster::default();
        let state = DashboardState::new(broadcaster, HealthSnapshot::default());
        let health = state.health.read().await;
        assert_eq!(health.total_chunks, 0);
    }

    #[tokio::test]
    async fn test_knowledge_snapshot_default_empty() {
        let snapshot = KnowledgeSnapshot::default();
        assert!(snapshot.items_by_category.is_empty());
        assert!(snapshot.recent_writebacks.is_empty());
        assert!(snapshot.onboarding_status.is_empty());
    }

    #[tokio::test]
    async fn test_knowledge_api_returns_json() {
        let mut items = std::collections::HashMap::new();
        items.insert("decision".to_string(), 5);
        items.insert("lesson".to_string(), 3);
        items.insert("pattern".to_string(), 2);

        let knowledge = KnowledgeSnapshot {
            items_by_category: items,
            recent_writebacks: vec![
                KnowledgeWriteback {
                    kind: "decision".to_string(),
                    title: "Use Rust".to_string(),
                    timestamp: "2026-03-10T00:00:00Z".to_string(),
                },
                KnowledgeWriteback {
                    kind: "lesson".to_string(),
                    title: "Pin dependencies".to_string(),
                    timestamp: "2026-03-09T12:00:00Z".to_string(),
                },
            ],
            onboarding_status: vec![OnboardingStatus {
                source: "my-repo".to_string(),
                status: "complete".to_string(),
                chunks_indexed: 150,
            }],
        };

        let state = DashboardState::with_knowledge(
            EventBroadcaster::default(),
            HealthSnapshot::default(),
            knowledge,
        );
        let (base, handle) = start_state_test_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}/api/knowledge"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["items_by_category"]["decision"], 5);
        assert_eq!(body["items_by_category"]["lesson"], 3);
        assert_eq!(body["recent_writebacks"][0]["kind"], "decision");
        assert_eq!(body["recent_writebacks"][0]["title"], "Use Rust");
        assert_eq!(body["onboarding_status"][0]["source"], "my-repo");
        assert_eq!(body["onboarding_status"][0]["status"], "complete");
        assert_eq!(body["onboarding_status"][0]["chunks_indexed"], 150);

        handle.abort();
    }

    #[tokio::test]
    async fn test_embedded_assets_contain_knowledge() {
        assert!(Assets::get("knowledge.html").is_some());
    }

    #[tokio::test]
    async fn test_embedded_assets_contain_search() {
        assert!(Assets::get("search.html").is_some());
    }

    #[tokio::test]
    async fn test_knowledge_html_served() {
        let state = DashboardState::new(EventBroadcaster::default(), HealthSnapshot::default());
        let (base, handle) = start_state_test_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}/dashboard/knowledge.html"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        assert!(body.contains("Knowledge Activity"));

        handle.abort();
    }

    #[tokio::test]
    async fn test_search_analytics_snapshot_default_empty() {
        let snapshot = SearchAnalyticsSnapshot::default();
        assert_eq!(snapshot.total_queries, 0);
        assert!(snapshot.query_frequency.is_empty());
        assert!(snapshot.avg_relevance.is_empty());
        assert_eq!(snapshot.cache_hit_rate, 0.0);
        assert_eq!(snapshot.sidecar_hit_rate, 0.0);
    }

    #[tokio::test]
    async fn test_search_analytics_api_returns_json() {
        let state = DashboardState::new(EventBroadcaster::default(), HealthSnapshot::default());
        {
            let mut analytics = state.search_analytics.write().await;
            analytics.total_queries = 42;
            analytics.avg_search_time_ms = 15.5;
            analytics.cache_hit_rate = 0.75;
            analytics.sidecar_hit_rate = 0.3;
            analytics.query_frequency = vec![
                TimeSeriesPoint {
                    timestamp: "2026-03-10T00:00:00Z".to_string(),
                    value: 10.0,
                },
                TimeSeriesPoint {
                    timestamp: "2026-03-10T01:00:00Z".to_string(),
                    value: 15.0,
                },
            ];
            analytics.avg_relevance = vec![TimeSeriesPoint {
                timestamp: "2026-03-10T00:00:00Z".to_string(),
                value: 0.85,
            }];
        }
        let (base, handle) = start_state_test_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}/api/search_analytics"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["total_queries"], 42);
        assert_eq!(body["avg_search_time_ms"], 15.5);
        assert_eq!(body["cache_hit_rate"], 0.75);
        assert_eq!(body["sidecar_hit_rate"], 0.3);
        assert_eq!(body["query_frequency"].as_array().unwrap().len(), 2);
        assert_eq!(body["avg_relevance"][0]["value"], 0.85);

        handle.abort();
    }

    #[tokio::test]
    async fn test_search_html_served() {
        let state = DashboardState::new(EventBroadcaster::default(), HealthSnapshot::default());
        let (base, handle) = start_state_test_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}/dashboard/search.html"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        assert!(body.contains("Search Analytics"));

        handle.abort();
    }

    #[tokio::test]
    async fn test_benchmark_snapshot_default_empty() {
        let snapshot = BenchmarkSnapshot::default();
        assert!(snapshot.active_sessions.is_empty());
        assert!(snapshot.historical_runs.is_empty());
        assert!(snapshot.token_savings_trend.is_empty());
    }

    #[tokio::test]
    async fn test_benchmarks_api_returns_json() {
        let state = DashboardState::new(EventBroadcaster::default(), HealthSnapshot::default());
        {
            let mut benchmarks = state.benchmarks.write().await;
            benchmarks.active_sessions = vec![BenchmarkRunSummary {
                session_id: "sess-001".to_string(),
                task: "Implement auth".to_string(),
                started_at: "2026-03-10T10:00:00Z".to_string(),
                status: "running".to_string(),
                baseline_tokens: 0,
                assisted_tokens: 0,
                token_savings_pct: 0.0,
                duration_ms: 0,
                event_count: 15,
            }];
            benchmarks.historical_runs = vec![BenchmarkRunSummary {
                session_id: "sess-000".to_string(),
                task: "Add tests".to_string(),
                started_at: "2026-03-09T08:00:00Z".to_string(),
                status: "complete".to_string(),
                baseline_tokens: 5000,
                assisted_tokens: 3200,
                token_savings_pct: 36.0,
                duration_ms: 120000,
                event_count: 42,
            }];
            benchmarks.token_savings_trend = vec![TokenSavingsPoint {
                session_id: "sess-000".to_string(),
                completed_at: "2026-03-09T10:00:00Z".to_string(),
                savings_pct: 36.0,
            }];
        }
        let (base, handle) = start_state_test_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}/api/benchmarks"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["active_sessions"][0]["session_id"], "sess-001");
        assert_eq!(body["active_sessions"][0]["event_count"], 15);
        assert_eq!(body["historical_runs"][0]["task"], "Add tests");
        assert_eq!(body["historical_runs"][0]["token_savings_pct"], 36.0);
        assert_eq!(body["token_savings_trend"][0]["savings_pct"], 36.0);

        handle.abort();
    }

    #[tokio::test]
    async fn test_embedded_assets_contain_benchmark() {
        assert!(Assets::get("benchmark.html").is_some());
    }

    #[tokio::test]
    async fn test_benchmark_html_served() {
        let state = DashboardState::new(EventBroadcaster::default(), HealthSnapshot::default());
        let (base, handle) = start_state_test_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}/dashboard/benchmark.html"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        assert!(body.contains("Benchmark Dashboard"));

        handle.abort();
    }
}
