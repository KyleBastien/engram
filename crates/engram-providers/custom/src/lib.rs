use async_trait::async_trait;
use engram_core::{EmbedError, EmbeddingProvider};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for the custom HTTP embedding provider.
#[derive(Debug, Clone)]
pub struct CustomConfig {
    /// Endpoint URL to POST embedding requests to.
    pub endpoint: String,
    /// Display name for the provider.
    pub name: String,
    /// Dimensionality of output vectors.
    pub dimensions: usize,
    /// Maximum texts per batch call.
    pub max_batch_size: usize,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum number of retries on transient failures.
    pub max_retries: u32,
    /// Custom headers. Values support `${ENV_VAR}` substitution.
    pub headers: HashMap<String, String>,
}

impl Default for CustomConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8080/embed".to_string(),
            name: "custom".to_string(),
            dimensions: 384,
            max_batch_size: 64,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            headers: HashMap::new(),
        }
    }
}

/// Substitutes `${ENV_VAR}` patterns in a string with environment variable values.
fn substitute_env_vars(input: &str) -> Result<String, EmbedError> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_name = String::new();
            let mut found_close = false;
            for ch in chars.by_ref() {
                if ch == '}' {
                    found_close = true;
                    break;
                }
                var_name.push(ch);
            }
            if !found_close {
                return Err(EmbedError::Api(format!(
                    "unclosed ${{}} in header value: {input}"
                )));
            }
            let value = std::env::var(&var_name).map_err(|_| {
                EmbedError::Unavailable(format!("environment variable {var_name} not set"))
            })?;
            result.push_str(&value);
        } else {
            result.push(c);
        }
    }

    Ok(result)
}

/// Build a HeaderMap from config headers, performing env var substitution.
fn build_headers(headers: &HashMap<String, String>) -> Result<HeaderMap, EmbedError> {
    let mut map = HeaderMap::new();
    for (key, value) in headers {
        let resolved = substitute_env_vars(value)?;
        let name = HeaderName::from_bytes(key.as_bytes())
            .map_err(|e| EmbedError::Api(format!("invalid header name '{key}': {e}")))?;
        let val = HeaderValue::from_str(&resolved)
            .map_err(|e| EmbedError::Api(format!("invalid header value for '{key}': {e}")))?;
        map.insert(name, val);
    }
    Ok(map)
}

/// Custom HTTP embedding provider that can point at any embedding API.
pub struct CustomProvider {
    config: CustomConfig,
    client: Client,
    resolved_headers: HeaderMap,
}

impl CustomProvider {
    /// Creates a new CustomProvider, resolving env vars in headers eagerly.
    pub fn new(config: CustomConfig) -> Result<Self, EmbedError> {
        let resolved_headers = build_headers(&config.headers)?;
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| EmbedError::Api(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            config,
            client,
            resolved_headers,
        })
    }

    /// Creates a new CustomProvider with a custom reqwest Client (useful for testing).
    pub fn with_client(config: CustomConfig, client: Client) -> Result<Self, EmbedError> {
        let resolved_headers = build_headers(&config.headers)?;
        Ok(Self {
            config,
            client,
            resolved_headers,
        })
    }

    async fn send_request(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let request_body = CustomEmbedRequest {
            texts: texts.iter().map(|s| s.to_string()).collect(),
        };

        let mut req = self.client.post(&self.config.endpoint).json(&request_body);

        for (name, value) in &self.resolved_headers {
            req = req.header(name.clone(), value.clone());
        }

        let response = req
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
                "custom endpoint returned status {status}: {body}"
            )));
        }

        let embed_response: CustomEmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbedError::Api(format!("failed to parse response: {e}")))?;

        Ok(embed_response.embeddings)
    }
}

#[derive(Serialize)]
struct CustomEmbedRequest {
    texts: Vec<String>,
}

#[derive(Deserialize)]
struct CustomEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl EmbeddingProvider for CustomProvider {
    fn name(&self) -> &str {
        &self.config.name
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
    use wiremock::matchers::{header, method};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_config(endpoint: &str) -> CustomConfig {
        CustomConfig {
            endpoint: endpoint.to_string(),
            name: "test-custom".to_string(),
            dimensions: 384,
            max_batch_size: 64,
            timeout: Duration::from_secs(5),
            max_retries: 0,
            headers: HashMap::new(),
        }
    }

    fn custom_response(embeddings: &[Vec<f32>]) -> serde_json::Value {
        serde_json::json!({ "embeddings": embeddings })
    }

    #[tokio::test]
    async fn embed_single_text() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 384];
        let response_body = custom_response(&[embedding]);

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = test_config(&format!("{}/embed", mock_server.uri()));
        let provider = CustomProvider::with_client(config, Client::new()).unwrap();
        let result = provider.embed(&["hello world"]).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);
    }

    #[tokio::test]
    async fn embed_batch_texts() {
        let mock_server = MockServer::start().await;

        let embeddings: Vec<Vec<f32>> = (0..3).map(|_| vec![0.5_f32; 384]).collect();
        let response_body = custom_response(&embeddings);

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = test_config(&format!("{}/embed", mock_server.uri()));
        let provider = CustomProvider::with_client(config, Client::new()).unwrap();
        let result = provider.embed(&["hello", "world", "test"]).await.unwrap();

        assert_eq!(result.len(), 3);
        for vec in &result {
            assert_eq!(vec.len(), 384);
        }
    }

    #[tokio::test]
    async fn embed_returns_unavailable_when_server_not_reachable() {
        let config = CustomConfig {
            endpoint: "http://127.0.0.1:1/embed".to_string(),
            timeout: Duration::from_millis(100),
            max_retries: 0,
            ..CustomConfig::default()
        };

        let provider = CustomProvider::new(config).unwrap();
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
            .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = test_config(&format!("{}/embed", mock_server.uri()));
        let provider = CustomProvider::with_client(config, Client::new()).unwrap();
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
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "30")
                    .set_body_string("rate limited"),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = test_config(&format!("{}/embed", mock_server.uri()));
        let provider = CustomProvider::with_client(config, Client::new()).unwrap();
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
    async fn name_returns_configured_name() {
        let config = CustomConfig {
            name: "my-embedder".to_string(),
            ..CustomConfig::default()
        };
        let provider = CustomProvider::new(config).unwrap();
        assert_eq!(provider.name(), "my-embedder");
    }

    #[tokio::test]
    async fn dimensions_returns_configured_value() {
        let config = CustomConfig {
            dimensions: 768,
            ..CustomConfig::default()
        };
        let provider = CustomProvider::new(config).unwrap();
        assert_eq!(provider.dimensions(), 768);
    }

    #[tokio::test]
    async fn max_batch_size_returns_configured_value() {
        let config = CustomConfig {
            max_batch_size: 32,
            ..CustomConfig::default()
        };
        let provider = CustomProvider::new(config).unwrap();
        assert_eq!(provider.max_batch_size(), 32);
    }

    #[tokio::test]
    async fn sends_custom_headers() {
        let mock_server = MockServer::start().await;

        let response_body = custom_response(&[vec![0.1_f32; 384]]);

        Mock::given(method("POST"))
            .and(header("x-api-key", "my-secret"))
            .and(header("x-custom", "value"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut headers = HashMap::new();
        headers.insert("x-api-key".to_string(), "my-secret".to_string());
        headers.insert("x-custom".to_string(), "value".to_string());

        let config = CustomConfig {
            endpoint: format!("{}/embed", mock_server.uri()),
            headers,
            timeout: Duration::from_secs(5),
            max_retries: 0,
            ..CustomConfig::default()
        };

        let provider = CustomProvider::with_client(config, Client::new()).unwrap();
        let result = provider.embed(&["test"]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn env_var_substitution_in_headers() {
        // Set a test env var
        std::env::set_var("TEST_CUSTOM_API_KEY", "resolved-key");

        let mock_server = MockServer::start().await;

        let response_body = custom_response(&[vec![0.1_f32; 384]]);

        Mock::given(method("POST"))
            .and(header("authorization", "Bearer resolved-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut headers = HashMap::new();
        headers.insert(
            "authorization".to_string(),
            "Bearer ${TEST_CUSTOM_API_KEY}".to_string(),
        );

        let config = CustomConfig {
            endpoint: format!("{}/embed", mock_server.uri()),
            headers,
            timeout: Duration::from_secs(5),
            max_retries: 0,
            ..CustomConfig::default()
        };

        let provider = CustomProvider::with_client(config, Client::new()).unwrap();
        let result = provider.embed(&["test"]).await.unwrap();
        assert_eq!(result.len(), 1);

        std::env::remove_var("TEST_CUSTOM_API_KEY");
    }

    #[test]
    fn env_var_substitution_missing_var() {
        std::env::remove_var("NONEXISTENT_VAR_FOR_TEST");
        let result = substitute_env_vars("Bearer ${NONEXISTENT_VAR_FOR_TEST}");
        assert!(result.is_err());
    }

    #[test]
    fn env_var_substitution_no_vars() {
        let result = substitute_env_vars("plain-value").unwrap();
        assert_eq!(result, "plain-value");
    }

    #[test]
    fn env_var_substitution_multiple_vars() {
        std::env::set_var("TEST_HOST_CUSTOM", "example.com");
        std::env::set_var("TEST_PORT_CUSTOM", "8080");

        let result = substitute_env_vars("${TEST_HOST_CUSTOM}:${TEST_PORT_CUSTOM}").unwrap();
        assert_eq!(result, "example.com:8080");

        std::env::remove_var("TEST_HOST_CUSTOM");
        std::env::remove_var("TEST_PORT_CUSTOM");
    }

    #[test]
    fn env_var_substitution_unclosed_brace() {
        let result = substitute_env_vars("Bearer ${UNCLOSED");
        assert!(result.is_err());
    }

    #[test]
    fn request_body_format() {
        let req = CustomEmbedRequest {
            texts: vec!["hello".to_string(), "world".to_string()],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["texts"], serde_json::json!(["hello", "world"]));
        // Ensure it sends "texts" not "input" or "model"
        assert!(json.get("model").is_none());
    }

    #[tokio::test]
    async fn retries_on_transient_failure() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .up_to_n_times(1)
            .expect(1)
            .mount(&mock_server)
            .await;

        let response_body = custom_response(&[vec![0.1_f32; 384]]);
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut config = test_config(&format!("{}/embed", mock_server.uri()));
        config.max_retries = 1;

        let provider = CustomProvider::with_client(config, Client::new()).unwrap();
        let result = provider.embed(&["hello"]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn does_not_retry_on_rate_limit() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "60")
                    .set_body_string("rate limited"),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut config = test_config(&format!("{}/embed", mock_server.uri()));
        config.max_retries = 3;

        let provider = CustomProvider::with_client(config, Client::new()).unwrap();
        let result = provider.embed(&["hello"]).await;

        assert!(matches!(result, Err(EmbedError::RateLimited(_))));
    }

    #[test]
    fn default_config_values() {
        let config = CustomConfig::default();
        assert_eq!(config.endpoint, "http://localhost:8080/embed");
        assert_eq!(config.name, "custom");
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 3);
        assert!(config.headers.is_empty());
    }
}
