use axum::http::{header, StatusCode, Uri};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::Embed;

use engram_core::DashboardConfig;

#[derive(Embed)]
#[folder = "src/assets/"]
struct Assets;

/// Serve the dashboard HTTP server on the configured port.
///
/// Serves static HTML/JS/CSS files bundled into the binary via rust-embed.
/// The dashboard is accessible at `http://localhost:{port}/dashboard`.
pub async fn serve_dashboard(config: &DashboardConfig) -> engram_core::Result<()> {
    let app = build_router();
    let addr = format!("0.0.0.0:{}", config.port);

    eprintln!("engram-dashboard: listening on http://localhost:{}/dashboard", config.port);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| engram_core::EngramError::Mcp(format!("Failed to bind {addr}: {e}")))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| engram_core::EngramError::Mcp(format!("Dashboard server error: {e}")))?;

    Ok(())
}

/// Build the axum Router for the dashboard.
///
/// Exposed publicly for testing.
pub fn build_router() -> Router {
    let dashboard_routes = Router::new()
        .route("/", get(index_handler))
        .fallback(get(static_handler));

    Router::new().nest("/dashboard", dashboard_routes)
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

    #[tokio::test]
    async fn test_dashboard_index_served() {
        let (base, handle) = start_test_server().await;
        let client = reqwest::Client::new();

        let resp = client.get(format!("{base}/dashboard")).send().await.unwrap();
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
}
