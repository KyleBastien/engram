mod chunk;
mod config;
mod embedding;
mod error;
mod knowledge;
mod manifest;
mod snapshot;
mod symbol;

pub use chunk::{ChunkKind, ChunkMetadata};
pub use config::{
    BenchmarkConfig, ChunkingConfig, ContextConfig, DashboardConfig, EmbeddingConfig, HooksConfig,
    ModePreset, ModesConfig, OllamaConfig, SearchConfig, SourceConfig, StorageConfig, StoreConfig,
    StoreSchema, StoreSection, SymbolResolutionConfig, WatcherConfig,
};
pub use embedding::{EmbedError, EmbeddingProvider};
pub use error::{EngramError, Result};
pub use knowledge::{Decision, GlossaryEntry, Lesson, Pattern};
pub use manifest::Manifest;
pub use snapshot::{Snapshot, SnapshotTier};
pub use symbol::{
    ExportedSymbol, ResolvedImport, SymbolId, SymbolReference, SymbolResolver, TypeHierarchy,
};
