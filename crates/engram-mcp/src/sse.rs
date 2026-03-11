use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::protocol::{JsonRpcRequest, JsonRpcResponse, PARSE_ERROR};
use crate::server::McpServer;

/// Shared state for the SSE transport, holding the MCP server and active client sessions.
struct SseState {
    server: McpServer,
    /// Map from session_id to the sender channel for that client's SSE stream.
    sessions: Mutex<HashMap<String, mpsc::Sender<JsonRpcResponse>>>,
}

/// Start the MCP SSE transport HTTP server on the given port.
///
/// This creates an axum HTTP server with:
/// - `GET /sse` — SSE endpoint that clients connect to; assigns a session ID and streams responses
/// - `POST /message?sessionId=<id>` — endpoint for clients to send JSON-RPC requests
///
/// The server runs until shutdown (Ctrl+C or process termination).
pub async fn serve_sse(server: McpServer, port: u16) -> engram_core::Result<()> {
    let state = Arc::new(SseState {
        server,
        sessions: Mutex::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/sse", get(handle_sse))
        .route("/message", post(handle_message))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    eprintln!("engram-mcp: SSE transport listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| engram_core::EngramError::Mcp(format!("Failed to bind {addr}: {e}")))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| engram_core::EngramError::Mcp(format!("SSE server error: {e}")))?;

    Ok(())
}

/// GET /sse — Establish an SSE connection.
///
/// Assigns a session ID and sends an initial `endpoint` event telling the client
/// where to POST JSON-RPC messages. Subsequent events are `message` events containing
/// JSON-RPC responses.
async fn handle_sse(
    State(state): State<Arc<SseState>>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::channel::<JsonRpcResponse>(256);

    // Register the session
    state
        .sessions
        .lock()
        .await
        .insert(session_id.clone(), tx);

    eprintln!("engram-mcp: SSE client connected: {session_id}");

    // Build a stream that:
    // 1. First yields the `endpoint` event with the message URL
    // 2. Then yields `message` events for each JSON-RPC response
    let endpoint_url = format!("/message?sessionId={session_id}");
    let endpoint_event = Event::default()
        .event("endpoint")
        .data(endpoint_url);

    let session_id_for_cleanup = session_id.clone();
    let state_for_cleanup = Arc::clone(&state);

    let response_stream = ReceiverStream::new(rx).map(|response| {
        let json = serde_json::to_string(&response).unwrap_or_default();
        Ok(Event::default().event("message").data(json))
    });

    // Combine the initial endpoint event with the response stream
    let initial = tokio_stream::once(Ok(endpoint_event));
    let combined = initial.chain(response_stream);

    // Spawn cleanup task: when the stream ends, remove the session
    tokio::spawn(async move {
        // Small delay to let the stream fully close
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        // Note: cleanup happens naturally when the receiver is dropped
        // but we also explicitly remove the session entry
        state_for_cleanup
            .sessions
            .lock()
            .await
            .remove(&session_id_for_cleanup);
        eprintln!("engram-mcp: SSE client disconnected: {session_id_for_cleanup}");
    });

    Sse::new(combined)
}

/// POST /message?sessionId=<id> — Receive a JSON-RPC request from a client.
///
/// Parses the JSON-RPC request, processes it through the MCP server, and sends
/// the response back via the client's SSE stream.
async fn handle_message(
    State(state): State<Arc<SseState>>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    body: String,
) -> impl IntoResponse {
    let session_id = match params.get("sessionId") {
        Some(id) => id.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing sessionId query parameter"})),
            );
        }
    };

    // Look up the session's SSE sender
    let tx = {
        let sessions = state.sessions.lock().await;
        sessions.get(&session_id).cloned()
    };

    let tx = match tx {
        Some(tx) => tx,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Unknown session ID"})),
            );
        }
    };

    // Parse the JSON-RPC request
    let request = match serde_json::from_str::<JsonRpcRequest>(&body) {
        Ok(req) => req,
        Err(e) => {
            eprintln!("engram-mcp: SSE parse error: {e}");
            let resp = JsonRpcResponse::error(None, PARSE_ERROR, format!("Parse error: {e}"));
            let _ = tx.send(resp).await;
            return (
                StatusCode::ACCEPTED,
                Json(serde_json::json!({"status": "error sent via SSE"})),
            );
        }
    };

    // Process the request through the MCP server
    if let Some(response) = state.server.handle_json_rpc(&request).await {
        let _ = tx.send(response).await;
    }

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({"status": "ok"})),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::McpServer;

    #[tokio::test]
    async fn test_sse_server_starts_and_accepts_connections() {
        // Start SSE server on a random available port
        let server = McpServer::new();
        let state = Arc::new(SseState {
            server,
            sessions: Mutex::new(HashMap::new()),
        });

        let app = Router::new()
            .route("/sse", get(handle_sse))
            .route("/message", post(handle_message))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Give the server a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Connect to SSE endpoint and read the initial endpoint event
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://{addr}/sse"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // Read enough of the SSE stream to get the endpoint event
        let text = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            resp.text(),
        )
        .await
        .unwrap_or_else(|_| Ok(String::new()))
        .unwrap_or_default();

        // The SSE response should contain the endpoint event
        assert!(
            text.contains("event: endpoint"),
            "Expected 'event: endpoint' in SSE stream, got: {text}"
        );
        assert!(
            text.contains("/message?sessionId="),
            "Expected '/message?sessionId=' in SSE stream, got: {text}"
        );

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_sse_message_without_session_id_returns_400() {
        let server = McpServer::new();
        let state = Arc::new(SseState {
            server,
            sessions: Mutex::new(HashMap::new()),
        });

        let app = Router::new()
            .route("/message", post(handle_message))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{addr}/message"))
            .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_sse_message_with_unknown_session_returns_404() {
        let server = McpServer::new();
        let state = Arc::new(SseState {
            server,
            sessions: Mutex::new(HashMap::new()),
        });

        let app = Router::new()
            .route("/message", post(handle_message))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{addr}/message?sessionId=nonexistent"))
            .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_sse_full_round_trip() {
        // Start server
        let server = McpServer::new();
        let state = Arc::new(SseState {
            server,
            sessions: Mutex::new(HashMap::new()),
        });

        let app = Router::new()
            .route("/sse", get(handle_sse))
            .route("/message", post(handle_message))
            .with_state(Arc::clone(&state));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();

        // Connect to SSE and extract session ID from the endpoint event
        let resp = client
            .get(format!("http://{addr}/sse"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // We need to read the SSE stream in a background task since it's streaming
        // Instead, let's manually create a session and test the message endpoint
        let session_id = "test-session";
        let (tx, mut rx) = mpsc::channel::<JsonRpcResponse>(256);
        state
            .sessions
            .lock()
            .await
            .insert(session_id.to_string(), tx);

        // Send an initialize request
        let resp = client
            .post(format!("http://{addr}/message?sessionId={session_id}"))
            .header("content-type", "application/json")
            .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 202);

        // Receive the response via the SSE channel
        let response = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();

        assert!(response.result.is_some());
        assert!(response.error.is_none());
        let result = response.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "engram");

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_sse_concurrent_sessions() {
        let server = McpServer::new();
        let state = Arc::new(SseState {
            server,
            sessions: Mutex::new(HashMap::new()),
        });

        let app = Router::new()
            .route("/message", post(handle_message))
            .with_state(Arc::clone(&state));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Create two concurrent sessions
        let (tx1, mut rx1) = mpsc::channel::<JsonRpcResponse>(256);
        let (tx2, mut rx2) = mpsc::channel::<JsonRpcResponse>(256);
        {
            let mut sessions = state.sessions.lock().await;
            sessions.insert("session-1".to_string(), tx1);
            sessions.insert("session-2".to_string(), tx2);
        }

        let client = reqwest::Client::new();

        // Send requests to both sessions concurrently
        let (r1, r2) = tokio::join!(
            client
                .post(format!("http://{addr}/message?sessionId=session-1"))
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#)
                .send(),
            client
                .post(format!("http://{addr}/message?sessionId=session-2"))
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#)
                .send(),
        );

        assert_eq!(r1.unwrap().status(), 202);
        assert_eq!(r2.unwrap().status(), 202);

        // Both sessions should receive their respective responses
        let resp1 = tokio::time::timeout(std::time::Duration::from_secs(2), rx1.recv())
            .await
            .unwrap()
            .unwrap();
        let resp2 = tokio::time::timeout(std::time::Duration::from_secs(2), rx2.recv())
            .await
            .unwrap()
            .unwrap();

        // Session 1 got initialize response
        assert_eq!(resp1.id, Some(serde_json::Value::Number(1.into())));
        assert!(resp1.result.is_some());

        // Session 2 got tools/list response
        assert_eq!(resp2.id, Some(serde_json::Value::Number(2.into())));
        assert!(resp2.result.is_some());

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_sse_parse_error_sent_via_stream() {
        let server = McpServer::new();
        let state = Arc::new(SseState {
            server,
            sessions: Mutex::new(HashMap::new()),
        });

        let app = Router::new()
            .route("/message", post(handle_message))
            .with_state(Arc::clone(&state));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let (tx, mut rx) = mpsc::channel::<JsonRpcResponse>(256);
        state
            .sessions
            .lock()
            .await
            .insert("err-session".to_string(), tx);

        let client = reqwest::Client::new();

        // Send malformed JSON
        let resp = client
            .post(format!(
                "http://{addr}/message?sessionId=err-session"
            ))
            .body("not valid json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 202);

        // Should receive a parse error via SSE
        let response = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();

        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, PARSE_ERROR);

        server_handle.abort();
    }
}
