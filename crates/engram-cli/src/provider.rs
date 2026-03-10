use async_trait::async_trait;
use engram_core::{EmbedError, EmbeddingConfig, EmbeddingProvider};

/// Minimal Ollama embedding provider for the CLI.
pub struct OllamaProvider {
    model: String,
    dims: usize,
    base_url: String,
    client: reqwest::Client,
}

impl OllamaProvider {
    pub fn from_config(config: &EmbeddingConfig) -> Self {
        Self {
            model: config.model.clone(),
            dims: config.dimensions as usize,
            base_url: format!("http://{}:{}", config.ollama.host, config.ollama.port),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    fn name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn max_batch_size(&self) -> usize {
        512
    }

    async fn embed(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let resp = self
            .client
            .post(format!("{}/api/embed", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbedError::Api(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbedError::Api(format!(
                "Ollama API returned {status}: {body}"
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| EmbedError::Api(format!("failed to parse response: {e}")))?;

        let embeddings = data["embeddings"]
            .as_array()
            .ok_or_else(|| EmbedError::Api("missing 'embeddings' field in response".into()))?
            .iter()
            .map(|e| {
                e.as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect();

        Ok(embeddings)
    }
}
