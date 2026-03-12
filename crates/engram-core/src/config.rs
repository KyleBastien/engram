use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Store schema metadata (version + unique store ID).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoreSchema {
    /// Semver version string for the store format.
    pub version: String,
    /// Unique identifier for this store instance.
    pub store_id: Uuid,
}

/// Top-level configuration matching `engram.config.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoreConfig {
    pub version: String,
    #[serde(default)]
    pub sources: Vec<SourceConfig>,
    #[serde(default)]
    pub store: StoreSection,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub symbol_resolution: SymbolResolutionConfig,
    #[serde(default)]
    pub chunking: ChunkingConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub context: ContextConfig,
    #[serde(default)]
    pub modes: ModesConfig,
    #[serde(default)]
    pub hooks: HooksConfig,
    #[serde(default)]
    pub watcher: WatcherConfig,
    #[serde(default)]
    pub dashboard: DashboardConfig,
    #[serde(default)]
    pub benchmark: BenchmarkConfig,
    #[serde(default)]
    pub storage: StorageConfig,
}

/// A source repository to index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceConfig {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
}

/// Store location settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoreSection {
    pub path: String,
    #[serde(default)]
    pub remote: Option<String>,
}

impl Default for StoreSection {
    fn default() -> Self {
        Self {
            path: ".engram-store".to_string(),
            remote: None,
        }
    }
}

/// Embedding provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub dimensions: u32,
    #[serde(default)]
    pub ollama: OllamaConfig,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            model: "nomic-embed-text".to_string(),
            dimensions: 768,
            ollama: OllamaConfig::default(),
        }
    }
}

/// Ollama-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OllamaConfig {
    pub host: String,
    pub port: u16,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 11434,
        }
    }
}

/// Symbol resolution settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SymbolResolutionConfig {
    pub enabled: bool,
    pub backend: String,
}

impl Default for SymbolResolutionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: "tree-sitter".to_string(),
        }
    }
}

/// Chunking strategy settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkingConfig {
    pub strategy: String,
    pub max_chunk_lines: u32,
    pub overlap_lines: u32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: "tree-sitter".to_string(),
            max_chunk_lines: 200,
            overlap_lines: 0,
        }
    }
}

/// Search configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchConfig {
    pub default_limit: u32,
    pub hybrid_weight: f32,
    pub hnsw_ef_construction: u32,
    pub hnsw_m: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 20,
            hybrid_weight: 0.7,
            hnsw_ef_construction: 200,
            hnsw_m: 16,
        }
    }
}

/// Context assembly settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextConfig {
    pub max_tokens: u32,
    pub include_signatures: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 8000,
            include_signatures: true,
        }
    }
}

/// Mode presets (e.g. default, deep-dive).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ModesConfig {
    #[serde(default)]
    pub presets: Vec<ModePreset>,
}

/// A single mode preset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModePreset {
    pub name: String,
    #[serde(default)]
    pub search_limit: Option<u32>,
    #[serde(default)]
    pub hybrid_weight: Option<f32>,
}

/// Lifecycle hook commands.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct HooksConfig {
    #[serde(default)]
    pub pre_index: Option<String>,
    #[serde(default)]
    pub post_index: Option<String>,
}

/// File watcher settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WatcherConfig {
    pub enabled: bool,
    pub debounce_ms: u32,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            debounce_ms: 500,
        }
    }
}

/// Dashboard configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DashboardConfig {
    pub enabled: bool,
    pub port: u16,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 3200,
        }
    }
}

/// Benchmark configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkConfig {
    pub enabled: bool,
    pub store_results: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            store_results: true,
        }
    }
}

/// Storage tuning configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StorageConfig {
    pub compression: bool,
    pub snapshot_interval_commits: u32,
    /// Embedding precision: "f32" (default) or "f16" for 50% storage reduction.
    #[serde(default = "default_embedding_precision")]
    pub embedding_precision: String,
}

fn default_embedding_precision() -> String {
    "f32".to_string()
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            compression: true,
            snapshot_interval_commits: 50,
            embedding_precision: "f32".to_string(),
        }
    }
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            sources: Vec::new(),
            store: StoreSection::default(),
            embedding: EmbeddingConfig::default(),
            symbol_resolution: SymbolResolutionConfig::default(),
            chunking: ChunkingConfig::default(),
            search: SearchConfig::default(),
            context: ContextConfig::default(),
            modes: ModesConfig::default(),
            hooks: HooksConfig::default(),
            watcher: WatcherConfig::default(),
            dashboard: DashboardConfig::default(),
            benchmark: BenchmarkConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_schema_roundtrip() {
        let schema = StoreSchema {
            version: "1.0.0".to_string(),
            store_id: Uuid::new_v4(),
        };
        let json = serde_json::to_string(&schema).unwrap();
        let deserialized: StoreSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, deserialized);
    }

    #[test]
    fn test_store_config_default() {
        let config = StoreConfig::default();
        assert_eq!(config.version, "1.0.0");
        assert!(config.sources.is_empty());
        assert_eq!(config.embedding.provider, "ollama");
        assert_eq!(config.embedding.model, "nomic-embed-text");
        assert_eq!(config.embedding.dimensions, 768);
        assert_eq!(config.embedding.ollama.host, "localhost");
        assert_eq!(config.embedding.ollama.port, 11434);
        assert_eq!(config.search.default_limit, 20);
        assert!((config.search.hybrid_weight - 0.7).abs() < f32::EPSILON);
        assert!(config.dashboard.enabled);
        assert_eq!(config.dashboard.port, 3200);
        assert!(!config.benchmark.enabled);
        assert!(config.storage.compression);
    }

    #[test]
    fn test_store_config_yaml_deserialization() {
        let yaml = r#"
version: "1.0.0"
sources:
  - name: my-project
    path: /home/user/project
    include:
      - "**/*.rs"
      - "**/*.ts"
    exclude:
      - "**/target/**"
store:
  path: /home/user/.engram-store
embedding:
  provider: ollama
  model: nomic-embed-text
  dimensions: 768
  ollama:
    host: localhost
    port: 11434
search:
  default_limit: 30
  hybrid_weight: 0.8
  hnsw_ef_construction: 200
  hnsw_m: 16
dashboard:
  enabled: true
  port: 9500
"#;
        let config: StoreConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.sources.len(), 1);
        assert_eq!(config.sources[0].name, "my-project");
        assert_eq!(config.sources[0].include.len(), 2);
        assert_eq!(config.sources[0].exclude.len(), 1);
        assert_eq!(config.store.path, "/home/user/.engram-store");
        assert_eq!(config.embedding.provider, "ollama");
        assert_eq!(config.embedding.dimensions, 768);
        assert_eq!(config.search.default_limit, 30);
        assert!((config.search.hybrid_weight - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.dashboard.port, 9500);
        // Fields not in YAML should get defaults
        assert!(config.watcher.enabled);
        assert_eq!(config.chunking.strategy, "tree-sitter");
    }

    #[test]
    fn test_store_config_minimal_yaml() {
        let yaml = r#"
version: "1.0.0"
"#;
        let config: StoreConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.version, "1.0.0");
        assert!(config.sources.is_empty());
        assert_eq!(config.store.path, ".engram-store");
        assert_eq!(config.embedding.provider, "ollama");
    }

    #[test]
    fn test_store_config_serialization_roundtrip() {
        let config = StoreConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: StoreConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_store_schema_uuid() {
        let id = Uuid::new_v4();
        let schema = StoreSchema {
            version: "1.0.0".to_string(),
            store_id: id,
        };
        let json = serde_json::to_string(&schema).unwrap();
        assert!(json.contains(&id.to_string()));
    }
}
