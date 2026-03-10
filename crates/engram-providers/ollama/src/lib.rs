use async_trait::async_trait;
use engram_core::{EmbedError, EmbeddingProvider};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Configuration for the Ollama embedding provider.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub base_url: String,
    pub model: String,
    pub dimensions: usize,
    pub max_batch_size: usize,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
            dimensions: 768,
            max_batch_size: 512,
        }
    }
}

/// Ollama embedding provider that generates embeddings locally via the Ollama API.
pub struct OllamaProvider {
    config: OllamaConfig,
    display_name: String,
    client: Client,
}

impl OllamaProvider {
    /// Creates a new OllamaProvider with the given configuration.
    pub fn new(config: OllamaConfig) -> Self {
        let display_name = format!("ollama/{}", config.model);
        Self {
            config,
            display_name,
            client: Client::new(),
        }
    }

    /// Creates a new OllamaProvider with a custom reqwest Client (useful for testing).
    pub fn with_client(config: OllamaConfig, client: Client) -> Self {
        let display_name = format!("ollama/{}", config.model);
        Self {
            config,
            display_name,
            client,
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
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
        let url = format!("{}/api/embed", self.config.base_url);

        let request_body = EmbedRequest {
            model: self.config.model.clone(),
            input: texts.iter().map(|s| s.to_string()).collect(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| EmbedError::Unavailable(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read response body".to_string());
            return Err(EmbedError::Api(format!(
                "Ollama returned status {status}: {body}"
            )));
        }

        let embed_response: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbedError::Api(format!("failed to parse response: {e}")))?;

        Ok(embed_response.embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_config(base_url: &str) -> OllamaConfig {
        OllamaConfig {
            base_url: base_url.to_string(),
            model: "nomic-embed-text".to_string(),
            dimensions: 768,
            max_batch_size: 512,
        }
    }

    #[tokio::test]
    async fn embed_single_text() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.1_f32; 768];
        let response_body = serde_json::json!({
            "embeddings": [embedding]
        });

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = OllamaProvider::new(test_config(&mock_server.uri()));
        let result = provider.embed(&["hello world"]).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 768);
    }

    #[tokio::test]
    async fn embed_batch_texts() {
        let mock_server = MockServer::start().await;

        let embedding = vec![0.5_f32; 768];
        let response_body = serde_json::json!({
            "embeddings": [embedding.clone(), embedding.clone(), embedding]
        });

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = OllamaProvider::new(test_config(&mock_server.uri()));
        let result = provider
            .embed(&["hello", "world", "test"])
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        for vec in &result {
            assert_eq!(vec.len(), 768);
        }
    }

    #[tokio::test]
    async fn embed_returns_unavailable_when_server_not_reachable() {
        let config = OllamaConfig {
            base_url: "http://127.0.0.1:1".to_string(),
            model: "nomic-embed-text".to_string(),
            dimensions: 768,
            max_batch_size: 512,
        };

        let provider = OllamaProvider::new(config);
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
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = OllamaProvider::new(test_config(&mock_server.uri()));
        let result = provider.embed(&["hello"]).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, EmbedError::Api(_)),
            "expected Api error, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn name_returns_configured_model() {
        let config = OllamaConfig {
            model: "mxbai-embed-large".to_string(),
            ..OllamaConfig::default()
        };
        let provider = OllamaProvider::new(config);
        assert_eq!(provider.name(), "ollama/mxbai-embed-large");
    }

    #[tokio::test]
    async fn dimensions_returns_configured_value() {
        let config = OllamaConfig {
            dimensions: 1024,
            ..OllamaConfig::default()
        };
        let provider = OllamaProvider::new(config);
        assert_eq!(provider.dimensions(), 1024);
    }

    #[tokio::test]
    async fn default_config_values() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "nomic-embed-text");
        assert_eq!(config.dimensions, 768);
        assert_eq!(config.max_batch_size, 512);
    }

    #[tokio::test]
    async fn sends_all_texts_in_single_request() {
        let mock_server = MockServer::start().await;

        let embeddings: Vec<Vec<f32>> = (0..5).map(|_| vec![0.1_f32; 768]).collect();
        let response_body = serde_json::json!({
            "embeddings": embeddings
        });

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .expect(1) // Exactly one request for all 5 texts
            .mount(&mock_server)
            .await;

        let provider = OllamaProvider::new(test_config(&mock_server.uri()));
        let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
        let result = provider.embed(&texts).await.unwrap();

        assert_eq!(result.len(), 5);
    }

    #[test]
    fn request_body_format() {
        let req = EmbedRequest {
            model: "nomic-embed-text".to_string(),
            input: vec!["hello".to_string(), "world".to_string()],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "nomic-embed-text");
        assert_eq!(json["input"], serde_json::json!(["hello", "world"]));
    }
}
