use thiserror::Error;

/// Unified error type for the Engram project.
#[derive(Error, Debug)]
pub enum EngramError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Git error: {0}")]
    Git(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Embed error: {0}")]
    Embed(String),

    #[error("Serialization error: {0}")]
    Serialize(String),

    #[error("Store error: {0}")]
    Store(String),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("Chunk parse error: {0}")]
    ChunkParse(String),
}

/// A convenience Result type alias for Engram operations.
pub type Result<T> = std::result::Result<T, EngramError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_from_std() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: EngramError = io_err.into();
        assert!(matches!(err, EngramError::Io(_)));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn string_variants_display() {
        let cases: Vec<(EngramError, &str)> = vec![
            (EngramError::Git("repo missing".into()), "Git error: repo missing"),
            (EngramError::Config("bad yaml".into()), "Config error: bad yaml"),
            (EngramError::Index("corrupt".into()), "Index error: corrupt"),
            (EngramError::Embed("rate limited".into()), "Embed error: rate limited"),
            (EngramError::Serialize("invalid json".into()), "Serialization error: invalid json"),
            (EngramError::Store("locked".into()), "Store error: locked"),
            (EngramError::Mcp("timeout".into()), "MCP error: timeout"),
            (EngramError::ChunkParse("bad syntax".into()), "Chunk parse error: bad syntax"),
        ];

        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn result_alias_works() {
        fn ok_result() -> Result<u32> {
            Ok(42)
        }

        fn err_result() -> Result<u32> {
            Err(EngramError::Store("fail".into()))
        }

        assert_eq!(ok_result().unwrap(), 42);
        assert!(err_result().is_err());
    }
}
