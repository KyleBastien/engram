use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, StatusCode, Uri};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::Embed;
use serde::Serialize;
use tokio::sync::broadcast;

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

/// Serve the dashboard HTTP server on the configured port.
///
/// Serves static HTML/JS/CSS files bundled into the binary via rust-embed.
/// The dashboard is accessible at `http://localhost:{port}/dashboard`.
/// WebSocket endpoint is at `ws://localhost:{port}/ws`.
pub async fn serve_dashboard(
    config: &DashboardConfig,
    broadcaster: EventBroadcaster,
) -> engram_core::Result<()> {
    let app = build_router_with_events(broadcaster);
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

/// Build the axum Router with WebSocket event broadcasting.
pub fn build_router_with_events(broadcaster: EventBroadcaster) -> Router {
    let shared = Arc::new(broadcaster);

    let dashboard_routes = Router::<Arc<EventBroadcaster>>::new()
        .route("/", get(index_handler))
        .fallback(get(static_handler));

    Router::new()
        .route("/ws", get(ws_handler))
        .nest("/dashboard", dashboard_routes)
        .with_state(shared)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(broadcaster): State<Arc<EventBroadcaster>>,
) -> impl IntoResponse {
    let rx = broadcaster.subscribe();
    ws.on_upgrade(|socket| handle_ws_connection(socket, rx))
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
    // Strip the /dashboard/ prefix to get the asset path
    let path = uri.path().trim_start_matches("/dashboard/");

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
}
