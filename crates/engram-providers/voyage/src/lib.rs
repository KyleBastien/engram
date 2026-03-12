use async_trait::async_trait;
use engram_core::{EmbedError, EmbeddingProvider};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the Voyage embedding provider.
#[derive(Debug, Clone)]
pub struct VoyageConfig {
    /// Base URL for the Voyage API.
    pub base_url: String,
    /// Model name (e.g., "voyage-code-3").
    pub model: String,
    /// Dimensionality of output vectors.
    pub dimensions: usize,
    /// Maximum texts per batch call.
    pub max_batch_size: usize,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum number of retries on transient failures.
    pub max_retries: u32,
}

impl Default for VoyageConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.voyageai.com".to_string(),
            model: "voyage-code-3".to_string(),
            dimensions: 1024,
            max_batch_size: 128,
            timeout: Duration::from_secs(30),
            max_retries: 3,
        }
    }
}

/// Voyage AI embedding provider.
pub struct VoyageProvider {
    config: VoyageConfig,
    display_name: String,
    api_key: String,
    client: Client,
}

impl VoyageProvider {
    /// Creates a new VoyageProvider, reading the API key from `VOYAGE_API_KEY` env var.
    pub fn new(config: VoyageConfig) -> Result<Self, EmbedError> {
        let api_key = std::env::var("VOYAGE_API_KEY")
            .map_err(|_| EmbedError::Unavailable("VOYAGE_API_KEY not set".to_string()))?;
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| EmbedError::Api(format!("failed to build HTTP client: {e}")))?;
        let display_name = format!("voyage/{}", config.model);
        Ok(Self {
            config,
            display_name,
            api_key,
            client,
        })
    }

    /// Creates a new VoyageProvider with an explicit API key.
    pub fn with_api_key(config: VoyageConfig, api_key: String) -> Self {
        let display_name = format!("voyage/{}", config.model);
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_default();
        Self {
            config,
            display_name,
            api_key,
            client,
        }
    }

    /// Creates a new VoyageProvider with a custom reqwest Client (useful for testing).
    pub fn with_client(config: VoyageConfig, api_key: String, client: Client) -> Self {
        let display_name = format!("voyage/{}", config.model);
        Self {
            config,
            display_name,
            api_key,
            client,
        }
    }

    async fn send_request(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let url = format!("{}/v1/embeddings", self.config.base_url);

        let request_body = EmbedRequest {
            model: self.config.model.clone(),
            input: texts.iter().map(|s| s.to_string()).collect(),
        };

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| EmbedError::Unavailable(e.to_string()))?;

        let status = response.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(60);
            return Err(EmbedError::RateLimited(format!(
                "retry after {retry_after}s"
            )));
        }

        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read response body".to_string());
            return Err(EmbedError::Api(format!(
                "Voyage returned status {status}: {body}"
            )));
        }

        let embed_response: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbedError::Api(format!("failed to parse response: {e}")))?;

        // Voyage returns embeddings with an index field; sort by index to preserve order
        let mut data = embed_response.data;
        data.sort_by_key(|d| d.index);

        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[async_trait]
impl EmbeddingProvider for VoyageProvider {
    fn name(&self) -> &str {
        &self.display_name
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn max_batch_size(&self) -> usize {
        self.config.max_batch_size
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            match self.send_request(texts).await {
                Ok(result) => return Ok(result),
                Err(EmbedError::RateLimited(msg)) => {
                    return Err(EmbedError::RateLimited(msg));
                }
                Err(e) => {
                    if attempt < self.config.max_retries {
                        last_err = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| EmbedError::Api("unknown error".to_string())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_config(base_url: &str) -> VoyageConfig {
        VoyageConfig {
            base_url: base_url.to_string(),
            model: "voyage-code-3".to_string(),
            dimensions: 1024,
            max_batch_size: 128,
            timeout: Duration::from_secs(5),
            max_retries: 0,
        }
    }

    fn voyage_response(embeddings: &[Vec<f32>]) -> serde_json::Value {
        let data: Vec<serde_json::Value> = embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                serde_json::json!({
                    "object": "embedding",
                    "embedding": emb,
                    "index": i
                })
            })
            .collect();
        serde_json::json!({
            "object": "list",
            "data": data,
            "model": "voyage-code-3",
            "usage": { "total_tokens": 5 }
        })
    }

    #[tokio::test]
    async fn embed_single_text() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 1024];
        let response_body = voyage_response(&[embedding]);

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = VoyageProvider::with_client(
            test_config(&mock_server.uri()),
            "test-key".into(),
            Client::new(),
        );
        let result = provider.embed(&["hello world"]).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1024);
    }

    #[tokio::test]
    async fn embed_batch_texts() {
        let mock_server = MockServer::start().await;

        let embeddings: Vec<Vec<f32>> = (0..3).map(|_| vec![0.5_f32; 1024]).collect();
        let response_body = voyage_response(&embeddings);

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = VoyageProvider::with_client(
            test_config(&mock_server.uri()),
            "test-key".into(),
            Client::new(),
        );
        let result = provider.embed(&["hello", "world", "test"]).await.unwrap();

        assert_eq!(result.len(), 3);
        for vec in &result {
            assert_eq!(vec.len(), 1024);
        }
    }

    #[tokio::test]
    async fn embed_returns_unavailable_when_server_not_reachable() {
        let config = VoyageConfig {
            base_url: "http://127.0.0.1:1".to_string(),
            timeout: Duration::from_millis(100),
            max_retries: 0,
            ..VoyageConfig::default()
        };

        let provider = VoyageProvider::with_api_key(config, "test-key".into());
        let result = provider.embed(&["hello"]).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, EmbedError::Unavailable(_)),
            "expected Unavailable error, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn embed_returns_api_error_on_non_success_status() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = VoyageProvider::with_client(
            test_config(&mock_server.uri()),
            "test-key".into(),
            Client::new(),
        );
        let result = provider.embed(&["hello"]).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, EmbedError::Api(_)),
            "expected Api error, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn embed_returns_rate_limited_on_429() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "30")
                    .set_body_string("rate limited"),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = VoyageProvider::with_client(
            test_config(&mock_server.uri()),
            "test-key".into(),
            Client::new(),
        );
        let result = provider.embed(&["hello"]).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, EmbedError::RateLimited(_)),
            "expected RateLimited error, got: {err:?}"
        );
        assert!(err.to_string().contains("30"));
    }

    #[tokio::test]
    async fn name_returns_configured_model() {
        let config = VoyageConfig {
            model: "voyage-3".to_string(),
            ..VoyageConfig::default()
        };
        let provider = VoyageProvider::with_api_key(config, "test-key".into());
        assert_eq!(provider.name(), "voyage/voyage-3");
    }

    #[tokio::test]
    async fn dimensions_returns_configured_value() {
        let config = VoyageConfig {
            dimensions: 512,
            ..VoyageConfig::default()
        };
        let provider = VoyageProvider::with_api_key(config, "test-key".into());
        assert_eq!(provider.dimensions(), 512);
    }

    #[tokio::test]
    async fn default_config_values() {
        let config = VoyageConfig::default();
        assert_eq!(config.base_url, "https://api.voyageai.com");
        assert_eq!(config.model, "voyage-code-3");
        assert_eq!(config.dimensions, 1024);
        assert_eq!(config.max_batch_size, 128);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 3);
    }

    #[tokio::test]
    async fn sends_bearer_auth_header() {
        let mock_server = MockServer::start().await;

        let response_body = voyage_response(&[vec![0.1_f32; 1024]]);

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("authorization", "Bearer my-secret-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = VoyageProvider::with_client(
            test_config(&mock_server.uri()),
            "my-secret-key".into(),
            Client::new(),
        );
        let result = provider.embed(&["test"]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn retries_on_transient_failure() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .up_to_n_times(1)
            .expect(1)
            .mount(&mock_server)
            .await;

        let response_body = voyage_response(&[vec![0.1_f32; 1024]]);
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut config = test_config(&mock_server.uri());
        config.max_retries = 1;

        let provider = VoyageProvider::with_client(config, "test-key".into(), Client::new());
        let result = provider.embed(&["hello"]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn does_not_retry_on_rate_limit() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "60")
                    .set_body_string("rate limited"),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut config = test_config(&mock_server.uri());
        config.max_retries = 3;

        let provider = VoyageProvider::with_client(config, "test-key".into(), Client::new());
        let result = provider.embed(&["hello"]).await;

        assert!(matches!(result, Err(EmbedError::RateLimited(_))));
    }

    #[tokio::test]
    async fn preserves_embedding_order() {
        let mock_server = MockServer::start().await;

        // Return embeddings out of order
        let response_body = serde_json::json!({
            "object": "list",
            "data": [
                { "object": "embedding", "embedding": vec![0.2_f32; 1024], "index": 1 },
                { "object": "embedding", "embedding": vec![0.1_f32; 1024], "index": 0 },
            ],
            "model": "voyage-code-3",
            "usage": { "total_tokens": 5 }
        });

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = VoyageProvider::with_client(
            test_config(&mock_server.uri()),
            "test-key".into(),
            Client::new(),
        );
        let result = provider.embed(&["first", "second"]).await.unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0][0] - 0.1).abs() < f32::EPSILON);
        assert!((result[1][0] - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn new_fails_without_env_var() {
        std::env::remove_var("VOYAGE_API_KEY");
        let result = VoyageProvider::new(VoyageConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn request_body_format() {
        let req = EmbedRequest {
            model: "voyage-code-3".to_string(),
            input: vec!["hello".to_string(), "world".to_string()],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "voyage-code-3");
        assert_eq!(json["input"], serde_json::json!(["hello", "world"]));
    }
}
