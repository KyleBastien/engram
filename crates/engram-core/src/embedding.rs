use async_trait::async_trait;
use std::fmt;

/// Errors that can occur during embedding operations.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbedError {
    /// API call failed with an error message.
    Api(String),
    /// Rate limited by the embedding provider.
    RateLimited(String),
    /// Input text exceeds the provider's maximum length.
    TextTooLong(String),
    /// Embedding provider is not reachable or available.
    Unavailable(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedError::Api(msg) => write!(f, "embedding API error: {msg}"),
            EmbedError::RateLimited(msg) => write!(f, "rate limited: {msg}"),
            EmbedError::TextTooLong(msg) => write!(f, "text too long: {msg}"),
            EmbedError::Unavailable(msg) => write!(f, "provider unavailable: {msg}"),
        }
    }
}

impl std::error::Error for EmbedError {}

/// Trait for embedding providers that convert text into vector representations.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Returns the name of the embedding provider and model (e.g., "ollama/nomic-embed-text").
    fn name(&self) -> &str;

    /// Returns the dimensionality of the embedding vectors produced.
    fn dimensions(&self) -> usize;

    /// Returns the maximum number of texts that can be embedded in a single batch request.
    fn max_batch_size(&self) -> usize;

    /// Embeds a batch of texts, returning one vector per input text.
    async fn embed(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, EmbedError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_error_display() {
        assert_eq!(
            EmbedError::Api("connection refused".into()).to_string(),
            "embedding API error: connection refused"
        );
        assert_eq!(
            EmbedError::RateLimited("retry after 60s".into()).to_string(),
            "rate limited: retry after 60s"
        );
        assert_eq!(
            EmbedError::TextTooLong("exceeded 8192 tokens".into()).to_string(),
            "text too long: exceeded 8192 tokens"
        );
        assert_eq!(
            EmbedError::Unavailable("ollama not running".into()).to_string(),
            "provider unavailable: ollama not running"
        );
    }

    #[test]
    fn embed_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmbedError>();
    }

    struct MockProvider;

    #[async_trait]
    impl EmbeddingProvider for MockProvider {
        fn name(&self) -> &str {
            "mock/test-model"
        }

        fn dimensions(&self) -> usize {
            384
        }

        fn max_batch_size(&self) -> usize {
            32
        }

        async fn embed(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
            Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
        }
    }

    #[tokio::test]
    async fn mock_provider_works() {
        let provider = MockProvider;
        assert_eq!(provider.name(), "mock/test-model");
        assert_eq!(provider.dimensions(), 384);
        assert_eq!(provider.max_batch_size(), 32);

        let result = provider.embed(&["hello", "world"]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 384);
    }

    #[test]
    fn trait_is_object_safe() {
        // Verify the trait can be used as a trait object
        fn _accepts_dyn(_provider: &dyn EmbeddingProvider) {}
    }
}
