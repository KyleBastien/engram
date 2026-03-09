# PRD: Engram — Git-Backed Semantic Context MCP Server

## Introduction

Engram is a git-backed semantic context server that gives AI coding agents (Claude Code, Cursor, Copilot, Codex CLI) deep, persistent awareness of codebases. Instead of agents burning tokens re-reading files and running grep chains for orientation, Engram maintains a versioned, searchable semantic index of code chunks, knowledge (architectural decisions, lessons learned, patterns), and conversation snapshots — all stored in a standard git repository.

The system is built in Rust as a single static binary exposing an MCP (Model Context Protocol) server. It uses tree-sitter for AST-aware chunking, pluggable embedding providers (Ollama by default), HNSW + BM25 hybrid search, and a write-back loop where agents contribute knowledge back to the store. The git-backed design gives history, collaboration, auditability, and sync for free.

The architecture is defined in `engram-architecture.md` at the repository root. This PRD translates that architecture into implementable user stories across all 5 rollout phases.

## Goals

- Provide AI agents with sub-second hybrid semantic + keyword search across indexed codebases
- Store all semantic data (chunk metadata, embeddings, knowledge, snapshots) in a git repository for versioning, diffing, and collaboration
- Support 500k+ chunk repositories with fast boot times (~50ms cached, ~10s cold for 200k chunks)
- Enable agents to write back architectural decisions, lessons learned, patterns, and glossary entries to the knowledge base
- Support multi-repo indexing with cross-repo symbol resolution
- Provide automated onboarding that bootstraps project knowledge from source code analysis
- Deliver a pluggable embedding architecture with 5 built-in providers (Ollama, OpenAI, Voyage, ONNX, custom HTTP)
- Include a benchmark harness that quantifies token savings and retrieval quality vs. baseline agent behavior
- Serve a live web dashboard for real-time observability of index health, search analytics, and session activity
- Adapt tool surface and behavior to client environments via a context/mode system
- Ship as a single static Rust binary with no runtime dependencies (except optional Ollama for embeddings)

## User Stories

Stories are organized by rollout phase. Each phase builds on the previous. Story IDs use the format `US-PPP-NNN` where `PPP` is the phase number.

---

### Phase 1 — Core Loop (MVP)

**Goal:** Ingest pipeline, git-backed storage, in-memory index, basic MCP search tools, CLI scaffolding. Single source repo, local-only mode, default context, explore mode.

#### Workspace & Project Scaffolding

#### US-001-001: Initialize Cargo workspace with Nx monorepo structure
**Description:** As an AI agent, I need a properly structured Cargo workspace with Nx orchestration so that all crates can be developed, built, and tested independently.

**Acceptance Criteria:**
- [ ] Root `Cargo.toml` defines a workspace with members: `crates/engram-core`, `crates/engram-ingest`, `crates/engram-query`, `crates/engram-mcp`, `crates/engram-store`, `crates/engram-cache`, `crates/engram-cli`
- [ ] Each crate has its own `Cargo.toml` with appropriate name (e.g., `engram-core`) and version `0.1.0`
- [ ] Root `nx.json` exists with workspace configuration
- [ ] Each crate has a `project.json` with `build`, `test`, and `lint` targets using `nx:run-commands` wrapping Cargo commands
- [ ] `cargo build --workspace` succeeds with empty `lib.rs`/`main.rs` files
- [ ] `nx run-many -t build` succeeds
- [ ] `.gitignore` includes `target/`, `.engram-cache/`, and `node_modules/`

#### US-001-002: Define shared error types in engram-core
**Description:** As an AI agent, I need a unified error type hierarchy so that all crates use consistent error handling.

**Acceptance Criteria:**
- [ ] `engram-core` exports an `EngramError` enum with variants: `Io`, `Git`, `Config`, `Index`, `Embed`, `Serialize`, `Store`, `Mcp`, `ChunkParse`
- [ ] Each variant wraps a source error where appropriate using `thiserror`
- [ ] A `Result<T>` type alias is exported as `type Result<T> = std::result::Result<T, EngramError>`
- [ ] `thiserror` is the only error crate dependency
- [ ] `cargo test -p engram-core` passes

#### US-001-003: Define Chunk and ChunkMetadata structs
**Description:** As an AI agent, I need the core data types for semantic chunks so that all crates share a common representation.

**Acceptance Criteria:**
- [ ] `engram-core` exports a `ChunkKind` enum with variants: `Function`, `Class`, `Method`, `Type`, `Impl`, `Module`, `DocSection`, `Readme`, `CommentBlock`, `Other`
- [ ] `engram-core` exports a `ChunkMetadata` struct with fields: `chunk_id: String`, `kind: ChunkKind`, `name: String`, `signature: Option<String>`, `start_line: u32`, `end_line: u32`, `content_hash: String`, `tags: Vec<String>`, `indexed_at: String` (ISO 8601), `source_commit: String`, `embedding_offset: u32`
- [ ] Both types derive `Serialize`, `Deserialize`, `Debug`, `Clone`, `PartialEq`
- [ ] `ChunkMetadata` can be serialized to JSON and deserialized from JSONL line format
- [ ] Unit tests verify round-trip serialization for all `ChunkKind` variants
- [ ] `cargo test -p engram-core` passes

#### US-001-004: Define EmbeddingProvider trait
**Description:** As an AI agent, I need the embedding provider trait definition so that concrete providers can be implemented against a stable interface.

**Acceptance Criteria:**
- [ ] `engram-core` exports an `EmbeddingProvider` async trait with methods: `fn name(&self) -> &str`, `fn dimensions(&self) -> usize`, `fn max_batch_size(&self) -> usize`, `async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>`
- [ ] `engram-core` exports an `EmbedError` enum with variants: `Api(String)`, `RateLimited { retry_after_ms: u64 }`, `TextTooLong { length: usize, max: usize }`, `Unavailable(String)`
- [ ] The trait uses `#[async_trait]` from the `async-trait` crate
- [ ] The trait is `Send + Sync`
- [ ] `cargo test -p engram-core` passes

#### US-001-005: Define SymbolResolver trait
**Description:** As an AI agent, I need the symbol resolver trait definition so that tree-sitter and LSP backends can be implemented against a stable interface.

**Acceptance Criteria:**
- [ ] `engram-core` exports a `SymbolResolver` async trait with methods: `async fn extract_exports(&self, file: &Path) -> Result<Vec<ExportedSymbol>>`, `async fn resolve_imports(&self, file: &Path) -> Result<Vec<ResolvedImport>>`, `async fn find_references(&self, symbol: &SymbolId) -> Result<Vec<SymbolReference>>`, `async fn type_hierarchy(&self, symbol: &SymbolId) -> Result<TypeHierarchy>`
- [ ] Supporting types `ExportedSymbol`, `ResolvedImport`, `SymbolId`, `SymbolReference`, `TypeHierarchy` are defined with appropriate fields
- [ ] The trait uses `#[async_trait]` and is `Send + Sync`
- [ ] `cargo test -p engram-core` passes

#### US-001-006: Define store schema version and store configuration types
**Description:** As an AI agent, I need the store configuration and schema types so that the store layout is well-defined and versioned.

**Acceptance Criteria:**
- [ ] `engram-core` exports a `StoreConfig` struct matching the `engram.config.yaml` schema from the architecture doc, including: `version`, `sources`, `store`, `embedding`, `symbol_resolution`, `chunking`, `search`, `context`, `modes`, `hooks`, `watcher`, `dashboard`, `benchmark`, `storage` sections
- [ ] Nested config types (e.g., `SourceConfig`, `EmbeddingConfig`, `OllamaConfig`, `SearchConfig`, `DashboardConfig`) are separate structs
- [ ] A `StoreSchema` struct exists with `version: String` (semver) and `store_id: String` (UUID)
- [ ] `StoreConfig` can be deserialized from a YAML string matching the format in the architecture doc section 16
- [ ] A default `StoreConfig` is provided via `Default` trait with sensible defaults (Ollama provider, tree-sitter backend, explore mode, etc.)
- [ ] `cargo test -p engram-core` passes

#### US-001-007: Define manifest types
**Description:** As an AI agent, I need the manifest type so that the global index manifest can be read and written.

**Acceptance Criteria:**
- [ ] `engram-core` exports a `Manifest` struct with fields: `chunk_count: u64`, `last_indexed_commit: Option<String>`, `model_name: String`, `dimensions: u16`, `source_repos: Vec<String>`, `created_at: String`, `updated_at: String`
- [ ] `Manifest` derives `Serialize`, `Deserialize`, `Debug`, `Clone`
- [ ] Can be serialized/deserialized to/from `manifest.json`
- [ ] Unit tests verify JSON round-trip
- [ ] `cargo test -p engram-core` passes

#### Embedding Providers

#### US-001-008: Implement Ollama embedding provider
**Description:** As an AI agent, I need the Ollama embedding provider so that embeddings can be generated locally at zero cost using Ollama.

**Acceptance Criteria:**
- [ ] A new crate directory `crates/engram-providers/ollama/` exists with its own `Cargo.toml`
- [ ] The provider implements the `EmbeddingProvider` trait from `engram-core`
- [ ] `name()` returns a string like `"ollama/nomic-embed-text"` (based on configured model)
- [ ] `dimensions()` returns the configured dimension count (default 768 for nomic-embed-text)
- [ ] `embed()` sends a POST request to the Ollama API (`/api/embed` endpoint) using `reqwest`
- [ ] The provider reads `base_url` and `model` from `OllamaConfig`
- [ ] If Ollama is not reachable, `embed()` returns `EmbedError::Unavailable` with a descriptive message
- [ ] Batch embedding sends all texts in a single request (Ollama supports batch)
- [ ] `cargo test -p engram-provider-ollama` passes (unit tests with mocked HTTP)

#### Git Store

#### US-001-009: Implement git store initialization (`engram init`)
**Description:** As an AI agent, I need git store initialization so that `engram init --local` creates a properly structured semantic store repository.

**Acceptance Criteria:**
- [ ] `engram-store` crate implements a `Store::init_local(path: &Path, config: &StoreConfig) -> Result<Store>` function
- [ ] Creates a new git repository at the specified path using `git2`
- [ ] Creates the directory structure: `.engram/` (with `version` and `store-id` files), `index/`, `knowledge/` (with subdirectories: `decisions/`, `lessons/`, `patterns/`, `glossary/`, `onboarding/`), `snapshots/` (with `active/`, `compressed/`, `archived/`), `metrics/` (with `baseline/`, `runs/`)
- [ ] Writes `engram.config.yaml` with the provided config
- [ ] Writes `.engram/version` with the store schema version (e.g., `"1.0.0"`)
- [ ] Writes `.engram/store-id` with a generated UUID
- [ ] Creates `.gitignore` in the store root that ignores `.engram-cache/`
- [ ] Makes an initial git commit: `"engram: initialize store"`
- [ ] `cargo test -p engram-store` passes (test creates a temp dir, inits, verifies structure)

#### US-001-010: Implement JSONL chunk metadata serialization
**Description:** As an AI agent, I need JSONL serialization for chunk metadata so that chunk metadata files can be written and read from the git store.

**Acceptance Criteria:**
- [ ] `engram-store` exports a `fn write_chunks_jsonl(path: &Path, chunks: &[ChunkMetadata]) -> Result<()>` that writes one JSON object per line
- [ ] `engram-store` exports a `fn read_chunks_jsonl(path: &Path) -> Result<Vec<ChunkMetadata>>` that reads and parses JSONL
- [ ] Empty files return an empty vec (not an error)
- [ ] Lines with trailing newlines or whitespace are handled gracefully
- [ ] Unit tests verify round-trip for multiple chunks in a single file
- [ ] `cargo test -p engram-store` passes

#### US-001-011: Implement binary embedding file format
**Description:** As an AI agent, I need the binary embedding file format so that embedding vectors can be stored compactly in the git store.

**Acceptance Criteria:**
- [ ] `engram-store` exports a `fn write_embeddings_bin(path: &Path, vectors: &[Vec<f32>], dimensions: u16) -> Result<()>`
- [ ] The binary format matches the architecture spec: 16-byte header (`magic: b"EGRM"`, `version: u16 = 1`, `dimensions: u16`, `count: u32`, `precision: u16 = 0` for f32, `reserved: u16 = 0`) followed by contiguous f32 vectors
- [ ] `engram-store` exports a `fn read_embeddings_bin(path: &Path) -> Result<EmbeddingFile>` where `EmbeddingFile` contains `dimensions`, `count`, `precision`, and the raw vector data
- [ ] Magic bytes are validated on read; mismatched magic returns an error
- [ ] Version is validated on read; unsupported versions return an error
- [ ] Unit tests verify round-trip for 0 vectors, 1 vector, and 100 vectors
- [ ] `cargo test -p engram-store` passes

#### US-001-012: Implement git commit helper for store changes
**Description:** As an AI agent, I need a git commit helper so that store changes (reindex, knowledge write-back) are committed with structured messages.

**Acceptance Criteria:**
- [ ] `engram-store` exports a `fn commit_changes(repo: &git2::Repository, message: &str) -> Result<git2::Oid>` that stages all changes in the store and commits
- [ ] The function uses `git2` to add all modified/new/deleted files to the index
- [ ] The commit author is set to `"engram"` with email `"engram@local"`
- [ ] Returns the commit OID on success
- [ ] If there are no changes to commit, returns `Ok` without creating an empty commit
- [ ] `cargo test -p engram-store` passes

#### US-001-013: Implement store file path resolution
**Description:** As an AI agent, I need file path resolution so that the store maps source repo files to their correct locations in the `index/` directory.

**Acceptance Criteria:**
- [ ] `engram-store` exports a `fn chunks_path(store_root: &Path, source_name: &str, relative_file_path: &str) -> PathBuf` that returns the path for a `.chunks.jsonl` file (e.g., `store_root/index/api/src/auth/login.ts.chunks.jsonl`)
- [ ] `engram-store` exports a `fn embeddings_path(store_root: &Path, source_name: &str, relative_file_path: &str) -> PathBuf` that returns the corresponding `.embeddings.bin` path
- [ ] Parent directories are created automatically if they don't exist
- [ ] Paths with special characters are handled safely (no path traversal)
- [ ] Unit tests verify correct path generation for nested paths
- [ ] `cargo test -p engram-store` passes

#### US-001-014: Implement manifest read/write
**Description:** As an AI agent, I need manifest read/write so that the global `manifest.json` tracks index state.

**Acceptance Criteria:**
- [ ] `engram-store` exports `fn write_manifest(store_root: &Path, manifest: &Manifest) -> Result<()>` that writes `index/manifest.json`
- [ ] `engram-store` exports `fn read_manifest(store_root: &Path) -> Result<Option<Manifest>>` that reads `index/manifest.json` (returns `None` if not found)
- [ ] JSON is pretty-printed for readability in git diffs
- [ ] Unit tests verify round-trip
- [ ] `cargo test -p engram-store` passes

#### Ingest Pipeline

#### US-001-015: Implement tree-sitter AST chunking for TypeScript
**Description:** As an AI agent, I need tree-sitter-based chunking for TypeScript so that TS/TSX files are split into semantic chunks at function, class, method, type, and module boundaries.

**Acceptance Criteria:**
- [ ] `engram-ingest` implements a `TreeSitterChunker` struct
- [ ] `TreeSitterChunker::chunk_file(path: &Path, source: &str, language: Language) -> Result<Vec<RawChunk>>` parses source using tree-sitter and extracts chunks
- [ ] For TypeScript, the following node types are extracted as chunks: `function_declaration`, `arrow_function` (when assigned), `class_declaration`, `method_definition`, `type_alias_declaration`, `interface_declaration`, `enum_declaration`, `export_statement` (when wrapping a declaration)
- [ ] Each `RawChunk` contains: `kind: ChunkKind`, `name: String`, `signature: Option<String>` (the first line / declaration line), `start_line: u32`, `end_line: u32`, `content: String` (the raw text of the chunk)
- [ ] Functions/methods include their full body
- [ ] Chunks do not overlap
- [ ] File-level code not inside any declaration is captured as `ChunkKind::Module`
- [ ] `cargo test -p engram-ingest` passes with test fixtures containing TypeScript code

#### US-001-016: Implement tree-sitter AST chunking for Rust
**Description:** As an AI agent, I need tree-sitter-based chunking for Rust so that `.rs` files are split into semantic chunks.

**Acceptance Criteria:**
- [ ] `TreeSitterChunker` supports Rust language
- [ ] For Rust, the following node types are extracted: `function_item`, `struct_item`, `enum_item`, `impl_item`, `trait_item`, `type_item`, `mod_item`, `const_item`, `static_item`, `macro_definition`
- [ ] `impl` blocks are chunked as a single unit (the impl + all methods inside)
- [ ] Each chunk has correct `kind`, `name`, `signature`, line range, and content
- [ ] `cargo test -p engram-ingest` passes with test fixtures containing Rust code

#### US-001-017: Implement tree-sitter AST chunking for Python
**Description:** As an AI agent, I need tree-sitter-based chunking for Python so that `.py` files are split into semantic chunks.

**Acceptance Criteria:**
- [ ] `TreeSitterChunker` supports Python language
- [ ] For Python, the following node types are extracted: `function_definition`, `class_definition`, `decorated_definition`
- [ ] Class bodies include all methods as part of the class chunk
- [ ] Decorators are included with the decorated function/class
- [ ] Each chunk has correct `kind`, `name`, `signature`, line range, and content
- [ ] `cargo test -p engram-ingest` passes with test fixtures containing Python code

#### US-001-018: Implement fallback sliding window chunker
**Description:** As an AI agent, I need a fallback chunker so that files in unsupported languages or non-code files are still indexed.

**Acceptance Criteria:**
- [ ] `engram-ingest` implements a `SlidingWindowChunker` struct
- [ ] `SlidingWindowChunker::chunk_file(source: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<RawChunk>` splits text into overlapping windows
- [ ] Token count is approximated by splitting on whitespace (exact tokenization not required for chunking)
- [ ] Each chunk has `kind: ChunkKind::Other`, a generated name (e.g., `"chunk_0"`, `"chunk_1"`), line range, and content
- [ ] Overlap ensures context continuity between chunks
- [ ] Files smaller than `max_tokens` produce a single chunk
- [ ] `cargo test -p engram-ingest` passes

#### US-001-019: Implement Markdown header-based chunker
**Description:** As an AI agent, I need a Markdown chunker so that documentation files are split at heading boundaries.

**Acceptance Criteria:**
- [ ] `engram-ingest` implements a `MarkdownChunker` struct
- [ ] Splits Markdown files at `#`, `##`, `###` heading boundaries
- [ ] Each chunk's `name` is the heading text
- [ ] Each chunk's `kind` is `ChunkKind::DocSection`
- [ ] Content between headings (including the heading line) is the chunk content
- [ ] Preamble before the first heading is captured as a chunk with name `"preamble"`
- [ ] `cargo test -p engram-ingest` passes

#### US-001-020: Implement language detection from file extension
**Description:** As an AI agent, I need language detection so that the correct chunker is selected for each file.

**Acceptance Criteria:**
- [ ] `engram-ingest` exports a `fn detect_language(path: &Path) -> ChunkerKind` function
- [ ] Returns `ChunkerKind::TreeSitter(Language::TypeScript)` for `.ts`, `.tsx`
- [ ] Returns `ChunkerKind::TreeSitter(Language::Rust)` for `.rs`
- [ ] Returns `ChunkerKind::TreeSitter(Language::Python)` for `.py`
- [ ] Returns `ChunkerKind::Markdown` for `.md`, `.mdx`
- [ ] Returns `ChunkerKind::SlidingWindow` for all other text file extensions
- [ ] Returns `ChunkerKind::Skip` for binary file extensions (`.png`, `.jpg`, `.wasm`, `.bin`, etc.)
- [ ] `cargo test -p engram-ingest` passes

#### US-001-021: Implement content hashing for change detection
**Description:** As an AI agent, I need content hashing so that unchanged chunks can be skipped during re-embedding.

**Acceptance Criteria:**
- [ ] `engram-ingest` exports a `fn content_hash(content: &str) -> String` that returns a hex-encoded SHA-256 hash of the chunk content
- [ ] `engram-ingest` exports a `fn has_chunk_changed(old_hash: &str, new_hash: &str) -> bool`
- [ ] Hashing is deterministic — same content always produces same hash
- [ ] `cargo test -p engram-ingest` passes

#### US-001-022: Implement incremental change detection via git diff
**Description:** As an AI agent, I need incremental change detection so that only modified files are re-chunked and re-embedded.

**Acceptance Criteria:**
- [ ] `engram-ingest` exports a `fn detect_changed_files(repo_path: &Path, since_commit: Option<&str>) -> Result<ChangedFiles>` using `git2`
- [ ] `ChangedFiles` contains `added: Vec<PathBuf>`, `modified: Vec<PathBuf>`, `deleted: Vec<PathBuf>`
- [ ] If `since_commit` is `None`, all tracked files are returned as "added" (full index)
- [ ] Respects include/exclude glob patterns from `SourceConfig`
- [ ] `cargo test -p engram-ingest` passes (test uses a temp git repo with staged changes)

#### US-001-023: Implement the full ingest pipeline
**Description:** As an AI agent, I need the full ingest pipeline orchestrator so that a source repo can be indexed end-to-end into the git store.

**Acceptance Criteria:**
- [ ] `engram-ingest` exports an `IngestPipeline` struct with method `async fn run(&self, source: &SourceConfig, store: &Store, provider: &dyn EmbeddingProvider, full: bool) -> Result<IngestReport>`
- [ ] The pipeline executes these steps in order: (1) detect changed files via git diff, (2) chunk each file with the appropriate chunker, (3) compute content hashes and skip unchanged chunks, (4) batch-embed new/modified chunks via the embedding provider, (5) write `.chunks.jsonl` and `.embeddings.bin` files to the store, (6) remove files from the store for deleted source files, (7) update `manifest.json`, (8) invalidate the compiled cache fingerprint, (9) commit changes to the store repo
- [ ] `IngestReport` contains: `files_processed: usize`, `chunks_created: usize`, `chunks_skipped: usize`, `chunks_deleted: usize`, `embed_calls: usize`
- [ ] The commit message follows the format: `"engram: reindex N files, M chunks updated"`
- [ ] `cargo test -p engram-ingest` passes (integration test with a temp source repo and temp store)

#### In-Memory Index

#### US-001-024: Implement HNSW vector index wrapper
**Description:** As an AI agent, I need an HNSW vector index so that semantic search can find the most similar chunks to a query embedding.

**Acceptance Criteria:**
- [ ] `engram-query` implements an `HnswIndex` struct wrapping the `usearch` crate
- [ ] `HnswIndex::build(chunks: &[(u64, Vec<f32>)], dimensions: usize) -> Result<Self>` constructs the index from chunk ID + vector pairs
- [ ] `HnswIndex::search(query: &[f32], top_k: usize) -> Vec<(u64, f32)>` returns (chunk_id_key, distance) sorted by ascending distance
- [ ] `HnswIndex::save(path: &Path) -> Result<()>` serializes the index to disk
- [ ] `HnswIndex::load(path: &Path) -> Result<Self>` loads a serialized index
- [ ] Performance: search returns in < 10ms for 50k vectors
- [ ] `cargo test -p engram-query` passes

#### US-001-025: Implement BM25 inverted index
**Description:** As an AI agent, I need a BM25 inverted index so that keyword search complements semantic search.

**Acceptance Criteria:**
- [ ] `engram-query` implements a `Bm25Index` struct
- [ ] `Bm25Index::build(chunks: &[(u64, &ChunkMetadata)]) -> Self` constructs the index from chunk `name`, `signature`, and `tags` fields
- [ ] `Bm25Index::search(query: &str, top_k: usize) -> Vec<(u64, f32)>` returns (chunk_id_key, score) sorted by descending relevance
- [ ] BM25 scoring uses standard parameters (k1=1.2, b=0.75)
- [ ] The index tokenizes on word boundaries and lowercases tokens
- [ ] `Bm25Index::save(path: &Path) -> Result<()>` and `Bm25Index::load(path: &Path) -> Result<()>` for cache persistence
- [ ] `cargo test -p engram-query` passes

#### US-001-026: Implement hybrid search combining HNSW and BM25
**Description:** As an AI agent, I need hybrid search so that queries combine semantic similarity and keyword relevance.

**Acceptance Criteria:**
- [ ] `engram-query` exports a `HybridSearch` struct with method `async fn search(&self, query: &str, query_embedding: &[f32], top_k: usize, alpha: f32) -> Vec<SearchResult>`
- [ ] Score is computed as: `alpha * normalized_vector_score + (1 - alpha) * normalized_bm25_score`
- [ ] Vector scores are normalized to [0, 1] range
- [ ] BM25 scores are normalized to [0, 1] range
- [ ] Results are deduplicated by chunk_id (same chunk may appear in both indexes)
- [ ] `SearchResult` contains: `chunk_id: String`, `score: f32`, `kind: ChunkKind`, `name: String`, `signature: Option<String>`, `file: String`, `repo: String`, `lines: (u32, u32)`, `stale: bool`
- [ ] Default alpha is 0.7 (favor semantic)
- [ ] `cargo test -p engram-query` passes

#### US-001-027: Implement metadata lookup indexes
**Description:** As an AI agent, I need in-memory metadata indexes so that direct lookups by file path, symbol name, chunk ID, and tag are fast.

**Acceptance Criteria:**
- [ ] `engram-query` implements a `MetadataIndex` struct with these lookup maps: `chunk_id → ChunkMetadata`, `file_path → Vec<ChunkMetadata>`, `symbol_name → Vec<ChunkMetadata>`, `tag → Vec<ChunkMetadata>`
- [ ] `MetadataIndex::build(chunks: &[ChunkMetadata]) -> Self` constructs all maps
- [ ] `MetadataIndex::lookup_by_id(id: &str) -> Option<&ChunkMetadata>`
- [ ] `MetadataIndex::lookup_by_file(path: &str) -> Vec<&ChunkMetadata>`
- [ ] `MetadataIndex::lookup_by_symbol(name: &str) -> Vec<&ChunkMetadata>`
- [ ] `MetadataIndex::lookup_by_tag(tag: &str) -> Vec<&ChunkMetadata>`
- [ ] All lookups are O(1) using `HashMap`
- [ ] `cargo test -p engram-query` passes

#### Compiled Index Cache

#### US-001-028: Implement compiled index cache write
**Description:** As an AI agent, I need to write compiled index caches so that subsequent boots are fast.

**Acceptance Criteria:**
- [ ] `engram-cache` exports a `fn write_cache(cache_dir: &Path, hnsw: &HnswIndex, bm25: &Bm25Index, metadata: &MetadataIndex, manifest_hash: &str) -> Result<()>`
- [ ] Writes to `.engram-cache/`: `fingerprint` (SHA-256 of manifest.json), `hnsw.index`, `bm25.index`, `metadata.bin`
- [ ] Creates `.engram-cache/` directory if it doesn't exist
- [ ] `cargo test -p engram-cache` passes

#### US-001-029: Implement compiled index cache read and validation
**Description:** As an AI agent, I need to read and validate the compiled index cache so that cache hits skip the slow cold boot path.

**Acceptance Criteria:**
- [ ] `engram-cache` exports a `fn try_load_cache(cache_dir: &Path, current_manifest_hash: &str) -> Result<Option<CachedIndex>>`
- [ ] If `fingerprint` file doesn't exist or doesn't match `current_manifest_hash`, returns `None` (cache miss)
- [ ] If fingerprint matches, loads `hnsw.index`, `bm25.index`, `metadata.bin` and returns `Some(CachedIndex)`
- [ ] `CachedIndex` contains the reconstructed `HnswIndex`, `Bm25Index`, and `MetadataIndex`
- [ ] If any cache file is corrupted or missing, returns `None` gracefully (no panic)
- [ ] `cargo test -p engram-cache` passes

#### US-001-030: Implement cache invalidation
**Description:** As an AI agent, I need cache invalidation so that stale caches are rebuilt after reindex.

**Acceptance Criteria:**
- [ ] `engram-cache` exports a `fn invalidate_cache(cache_dir: &Path) -> Result<()>` that deletes the `fingerprint` file
- [ ] After invalidation, `try_load_cache` returns `None` on next call
- [ ] `cargo test -p engram-cache` passes

#### Boot Sequence

#### US-001-031: Implement the full boot sequence
**Description:** As an AI agent, I need the full boot sequence so that Engram loads its index from the git store (fast path via cache, slow path via deserialization).

**Acceptance Criteria:**
- [ ] `engram-query` exports an `IndexManager` struct with `async fn boot(store_root: &Path, config: &StoreConfig) -> Result<Self>`
- [ ] Boot sequence: (1) read `manifest.json`, (2) compute manifest hash, (3) try loading compiled cache, (4) if cache hit → ready, (5) if cache miss → walk `index/` directory, deserialize all `.chunks.jsonl`, read all `.embeddings.bin` (zero-copy where possible), build HNSW index, build BM25 index, build metadata index, write compiled cache
- [ ] `IndexManager` holds the live HNSW, BM25, and metadata indexes
- [ ] Boot reports timing: `"Boot completed in Xms (cache hit)"` or `"Boot completed in Xs (cold, N chunks)"`
- [ ] `cargo test -p engram-query` passes (integration test with a populated store)

#### Staleness Detection

#### US-001-032: Implement chunk staleness detection
**Description:** As an AI agent, I need staleness detection so that search results indicate when chunks may be out of date.

**Acceptance Criteria:**
- [ ] `engram-query` exports a `fn check_staleness(chunks: &[ChunkMetadata], source_repo_path: &Path) -> Result<HashMap<String, bool>>` that compares each chunk's `source_commit` against the source repo's `HEAD`
- [ ] A chunk is stale if its `source_commit` is not an ancestor of (or equal to) `HEAD`
- [ ] Uses `git2` to check commit ancestry
- [ ] Returns a map of `chunk_id → is_stale`
- [ ] Stale chunks are still served (not removed)
- [ ] `cargo test -p engram-query` passes

#### MCP Server

#### US-001-033: Implement MCP server bootstrap with stdio transport
**Description:** As an AI agent, I need the MCP server bootstrap so that Engram can be used as an MCP server over stdio.

**Acceptance Criteria:**
- [ ] `engram-mcp` implements the MCP JSON-RPC protocol over stdin/stdout
- [ ] The server responds to `initialize` with server capabilities listing available tools
- [ ] The server responds to `tools/list` with the tool definitions for Phase 1 tools
- [ ] The server handles `tools/call` requests by dispatching to the appropriate handler
- [ ] Uses either the `rmcp` crate or a hand-rolled JSON-RPC implementation
- [ ] The server logs to stderr (not stdout, to avoid corrupting the MCP transport)
- [ ] `cargo test -p engram-mcp` passes

#### US-001-034: Implement `engram_search` MCP tool
**Description:** As an AI agent, I need the `engram_search` MCP tool so that agents can perform hybrid semantic + keyword search.

**Acceptance Criteria:**
- [ ] Tool name: `engram_search`
- [ ] Parameters: `query: String` (required), `scope: Option<String>` (one of "code", "docs", "knowledge", "all"; default "all"), `top_k: Option<u32>` (default from config), `compact: Option<bool>` (default from config)
- [ ] The tool embeds the query using the configured embedding provider, then runs hybrid search
- [ ] In Phase 1, `scope` only supports "code", "docs", and "all" (no knowledge partition yet)
- [ ] Returns JSON matching the `code_results` format from architecture doc section 20.2 (without `knowledge_results` in Phase 1)
- [ ] Compact mode omits `signature` and truncates long names
- [ ] Response includes `meta` with `code_count`, `search_time_ms`, and `compact`
- [ ] `cargo test -p engram-mcp` passes

#### US-001-035: Implement `engram_lookup` MCP tool
**Description:** As an AI agent, I need the `engram_lookup` MCP tool so that agents can directly look up chunks by file path, symbol name, or chunk ID.

**Acceptance Criteria:**
- [ ] Tool name: `engram_lookup`
- [ ] Parameters: `identifier: String` (required), `include_content: Option<bool>` (default false)
- [ ] The tool detects the identifier type: chunk ID (contains `#`), file path (contains `/` or `.`), or symbol name (everything else)
- [ ] Dispatches to the appropriate `MetadataIndex` lookup
- [ ] Returns matching chunks as JSON array with fields: `chunk_id`, `kind`, `name`, `signature`, `file`, `repo`, `lines`, `stale`
- [ ] If `include_content` is true, includes the `content` field (raw source text — read from the source repo, NOT stored in the semantic store)
- [ ] Returns empty array for no matches (not an error)
- [ ] `cargo test -p engram-mcp` passes

#### US-001-036: Implement `engram_status` MCP tool
**Description:** As an AI agent, I need the `engram_status` MCP tool so that agents can check index health and staleness.

**Acceptance Criteria:**
- [ ] Tool name: `engram_status`
- [ ] No required parameters
- [ ] Returns JSON with: `total_chunks: u64`, `source_repos: Vec<{name, chunk_count, last_indexed_commit, stale_chunks}>`, `index_age` (time since last reindex), `cache_status` ("hit" or "miss" from last boot), `boot_time_ms`, `embedding_provider: String`, `store_path: String`
- [ ] `cargo test -p engram-mcp` passes

#### CLI

#### US-001-037: Implement `engram init` CLI command
**Description:** As an AI agent, I need the `engram init` CLI command so that users can create a new semantic store.

**Acceptance Criteria:**
- [ ] `engram-cli` binary crate uses `clap` for argument parsing
- [ ] `engram init --local [--path <dir>]` creates a new local store (defaults to `./engram-store`)
- [ ] Calls `Store::init_local` from `engram-store`
- [ ] Prints a success message with the store path
- [ ] If the store already exists, prints an error and exits with code 1
- [ ] `cargo test -p engram-cli` passes

#### US-001-038: Implement `engram reindex` CLI command
**Description:** As an AI agent, I need the `engram reindex` CLI command so that users can trigger indexing of source repos.

**Acceptance Criteria:**
- [ ] `engram reindex [--full | --incremental] [--paths <glob>] [--repo <name>]`
- [ ] `--full` forces a complete re-chunk and re-embed of all files
- [ ] `--incremental` (default) only processes files changed since last indexed commit
- [ ] `--paths` scopes to specific file globs within the source repo
- [ ] `--repo` scopes to a specific source repo by name (from config)
- [ ] Reads `engram.config.yaml` to find source repos and embedding config
- [ ] Runs the `IngestPipeline` and prints the `IngestReport`
- [ ] `cargo test -p engram-cli` passes

#### US-001-039: Implement `engram serve` CLI command
**Description:** As an AI agent, I need the `engram serve` CLI command so that the MCP server can be started.

**Acceptance Criteria:**
- [ ] `engram serve [--transport stdio | --transport sse --port <port>] [--context <ctx>]`
- [ ] `--transport stdio` (default) starts the MCP server reading from stdin and writing to stdout
- [ ] `--context` sets the server context (default: "default") — stored but only used for tool surface filtering in later phases
- [ ] On start, runs the boot sequence to load the index
- [ ] Logs boot timing and index stats to stderr
- [ ] In Phase 1, only stdio transport is required (SSE is a later phase)
- [ ] `cargo test -p engram-cli` passes

#### US-001-040: Implement `engram status` CLI command
**Description:** As an AI agent, I need the `engram status` CLI command so that users can check store health from the terminal.

**Acceptance Criteria:**
- [ ] `engram status` reads the store and prints: chunk count, source repos, last indexed commit per repo, staleness summary, cache status, embedding provider
- [ ] Output is human-readable (not JSON)
- [ ] If no store is found, prints an error suggesting `engram init`
- [ ] `cargo test -p engram-cli` passes

---

### Phase 2 — Write-Back, Knowledge & Onboarding

**Goal:** Agents contribute knowledge back to the store. Automated onboarding bootstraps project knowledge. Snapshots enable session resumability. SmartRouter sidecar surfaces knowledge alongside code search results. Self-regulation tools help agents assess their own context-gathering behavior.

#### Knowledge Data Types

#### US-002-001: Define knowledge item types (Decision, Lesson, Pattern, Glossary)
**Description:** As an AI agent, I need the knowledge item data types so that all knowledge write-back tools share a common schema.

**Acceptance Criteria:**
- [ ] `engram-core` exports a `Decision` struct with fields: `id: String`, `title: String`, `status: String` (accepted/proposed/superseded/deprecated), `context: String`, `decision: String`, `consequences: Vec<String>`, `related_files: Vec<String>`, `contributed_by: String`, `created_at: String`, `embedding_ref: Option<String>`
- [ ] `engram-core` exports a `Lesson` struct with fields: `id: String`, `title: String`, `description: String`, `trigger: String`, `resolution: Option<String>`, `related_files: Vec<String>`, `contributed_by: String`, `created_at: String`, `embedding_ref: Option<String>`
- [ ] `engram-core` exports a `Pattern` struct with fields: `id: String`, `name: String`, `description: String`, `examples: Vec<String>`, `anti_patterns: Vec<String>`, `contributed_by: String`, `created_at: String`, `embedding_ref: Option<String>`
- [ ] `engram-core` exports a `GlossaryEntry` struct with fields: `term: String`, `definition: String`, `context: Option<String>`, `contributed_by: String`, `created_at: String`
- [ ] All types derive `Serialize`, `Deserialize`, `Debug`, `Clone`
- [ ] All types can be serialized to and deserialized from YAML format
- [ ] `cargo test -p engram-core` passes

#### US-002-002: Define snapshot types with tiered storage
**Description:** As an AI agent, I need snapshot data types so that conversation context can be saved and restored.

**Acceptance Criteria:**
- [ ] `engram-core` exports a `Snapshot` struct with fields: `session_id: String`, `summary: String`, `key_context: Vec<String>`, `full_transcript: String`, `created_at: String`, `tier: SnapshotTier`
- [ ] `SnapshotTier` enum has variants: `Active`, `Compressed`, `Archived`
- [ ] All types derive `Serialize`, `Deserialize`, `Debug`, `Clone`
- [ ] Active snapshots serialize to YAML
- [ ] `cargo test -p engram-core` passes

#### US-002-003: Define onboarding knowledge types
**Description:** As an AI agent, I need onboarding knowledge types so that automated bootstrapping produces well-structured knowledge files.

**Acceptance Criteria:**
- [ ] `engram-core` exports `ProjectOverview` struct with fields: `name`, `language`, `framework`, `package_manager`, `repo_type`, `description`, `entry_points: Vec<String>`
- [ ] `engram-core` exports `BuildTestCommands` struct with fields: `build: CommandInfo`, `test: CommandInfo`, `lint: CommandInfo`, `run: CommandInfo` where `CommandInfo` has `command`, `output_dir`, `framework`, `config`, `port` (all optional except `command`)
- [ ] `engram-core` exports `ArchitectureMap` struct with `directories: Vec<DirectoryInfo>` where `DirectoryInfo` has `path`, `purpose`, `key_files`
- [ ] `engram-core` exports `KeyAbstractions` struct with `abstractions: Vec<Abstraction>`, `patterns: Vec<PatternInfo>` where `Abstraction` has `name`, `kind`, `file`, `description` and `PatternInfo` has `name`, `description`, `examples`
- [ ] All types serialize to YAML matching the format in architecture doc section 8.1
- [ ] `cargo test -p engram-core` passes

#### Knowledge Write-Back to Store

#### US-002-004: Implement knowledge YAML write to git store
**Description:** As an AI agent, I need to write knowledge items as YAML files to the git store so that decisions, lessons, and patterns are versioned.

**Acceptance Criteria:**
- [ ] `engram-store` exports `fn write_decision(store_root: &Path, decision: &Decision) -> Result<PathBuf>` that writes to `knowledge/decisions/{date}_{slug}.yaml`
- [ ] `engram-store` exports `fn write_lesson(store_root: &Path, lesson: &Lesson) -> Result<PathBuf>` that writes to `knowledge/lessons/{date}_{slug}.yaml`
- [ ] `engram-store` exports `fn write_pattern(store_root: &Path, pattern: &Pattern) -> Result<PathBuf>` that writes to `knowledge/patterns/{slug}.yaml`
- [ ] `engram-store` exports `fn write_glossary_entry(store_root: &Path, entry: &GlossaryEntry) -> Result<PathBuf>` that writes to `knowledge/glossary/terms.yaml` (appends/updates the entry in the terms file)
- [ ] Slug is generated from the title/name by lowercasing and replacing spaces with hyphens
- [ ] YAML output is human-readable with multiline strings where appropriate
- [ ] `cargo test -p engram-store` passes

#### US-002-005: Implement knowledge item embedding and indexing
**Description:** As an AI agent, I need knowledge items to be embedded and indexed so that they are searchable alongside code chunks.

**Acceptance Criteria:**
- [ ] When a knowledge item is written, its text content (title + context/description + decision/resolution) is embedded using the configured provider
- [ ] The embedding is stored in a small binary file alongside the YAML (e.g., `knowledge/decisions/2026-03-08_auth-strategy.embedding.bin`)
- [ ] The `embedding_ref` field in the YAML points to this binary file
- [ ] Knowledge embeddings use the same binary format as code chunk embeddings (EGRM header)
- [ ] On boot, knowledge embeddings are loaded into the HNSW index in a separate partition (partition 2 per architecture doc section 20.3)
- [ ] `cargo test -p engram-store` passes

#### US-002-006: Implement snapshot write with tiered storage
**Description:** As an AI agent, I need snapshot storage so that conversation contexts can be saved and later retrieved or searched.

**Acceptance Criteria:**
- [ ] `engram-store` exports `fn write_snapshot(store_root: &Path, snapshot: &Snapshot) -> Result<PathBuf>` that writes to `snapshots/active/{timestamp}_{session_id}.yaml`
- [ ] `engram-store` exports `fn read_snapshot(store_root: &Path, session_id: &str) -> Result<Option<Snapshot>>` that searches all tiers for the snapshot
- [ ] Compressed snapshots (`.yaml.zst`) are transparently decompressed on read using the `zstd` crate
- [ ] Snapshot summary and key_context are embedded and stored in partition 3 of the HNSW index
- [ ] `cargo test -p engram-store` passes

#### US-002-007: Implement snapshot tier promotion (`engram compact`)
**Description:** As an AI agent, I need snapshot compaction so that old snapshots are compressed to manage repo size.

**Acceptance Criteria:**
- [ ] `engram-store` exports `fn compact_snapshots(store_root: &Path) -> Result<CompactReport>`
- [ ] Snapshots in `active/` older than 7 days are zstd-compressed and moved to `compressed/`
- [ ] Snapshots in `compressed/` older than 90 days are moved to `archived/`
- [ ] Original uncompressed files are deleted after successful compression
- [ ] `CompactReport` contains counts of snapshots promoted per tier
- [ ] Changes are committed: `"engram: compact N snapshots"`
- [ ] `cargo test -p engram-store` passes

#### Write-Back MCP Tools

#### US-002-008: Implement `engram_record_decision` MCP tool
**Description:** As an AI agent, I need the `engram_record_decision` MCP tool so that agents can commit architectural decisions to the knowledge base.

**Acceptance Criteria:**
- [ ] Tool name: `engram_record_decision`
- [ ] Required parameters: `title: String`, `context: String`, `decision: String`
- [ ] Optional parameters: `consequences: Vec<String>`, `related_files: Vec<String>`, `status: String` (default "accepted")
- [ ] Auto-generates `id` (format: `decision-{date}-{seq}`), `contributed_by` (from MCP client info), `created_at` (ISO 8601 now)
- [ ] Writes the decision YAML, embeds it, and commits to the store
- [ ] Returns JSON with `id`, `path` (store-relative path), and `committed: true`
- [ ] `cargo test -p engram-mcp` passes

#### US-002-009: Implement `engram_record_lesson` MCP tool
**Description:** As an AI agent, I need the `engram_record_lesson` MCP tool so that agents can commit lessons learned to the knowledge base.

**Acceptance Criteria:**
- [ ] Tool name: `engram_record_lesson`
- [ ] Required parameters: `title: String`, `description: String`, `trigger: String`
- [ ] Optional parameters: `resolution: Option<String>`, `related_files: Vec<String>`
- [ ] Auto-generates `id`, `contributed_by`, `created_at`
- [ ] Writes the lesson YAML, embeds it, and commits to the store
- [ ] Returns JSON with `id`, `path`, and `committed: true`
- [ ] `cargo test -p engram-mcp` passes

#### US-002-010: Implement `engram_record_pattern` MCP tool
**Description:** As an AI agent, I need the `engram_record_pattern` MCP tool so that agents can document discovered code patterns.

**Acceptance Criteria:**
- [ ] Tool name: `engram_record_pattern`
- [ ] Required parameters: `name: String`, `description: String`
- [ ] Optional parameters: `examples: Vec<String>`, `anti_patterns: Vec<String>`
- [ ] Auto-generates `id`, `contributed_by`, `created_at`
- [ ] Writes the pattern YAML, embeds it, and commits to the store
- [ ] Returns JSON with `id`, `path`, and `committed: true`
- [ ] `cargo test -p engram-mcp` passes

#### US-002-011: Implement `engram_record_glossary` MCP tool
**Description:** As an AI agent, I need the `engram_record_glossary` MCP tool so that agents can add or update project-specific terminology.

**Acceptance Criteria:**
- [ ] Tool name: `engram_record_glossary`
- [ ] Required parameters: `term: String`, `definition: String`
- [ ] Optional parameters: `context: Option<String>`
- [ ] If the term already exists in `terms.yaml`, updates it in place
- [ ] If the term is new, appends it
- [ ] Commits to the store
- [ ] Returns JSON with `term`, `action` ("created" or "updated"), and `committed: true`
- [ ] `cargo test -p engram-mcp` passes

#### US-002-012: Implement `engram_snapshot` MCP tool
**Description:** As an AI agent, I need the `engram_snapshot` MCP tool so that agents can save full conversation context for session resumability.

**Acceptance Criteria:**
- [ ] Tool name: `engram_snapshot`
- [ ] Required parameters: `session_id: String`, `summary: String`, `key_context: Vec<String>`, `full_transcript: String`
- [ ] Writes the snapshot to `snapshots/active/` with timestamp prefix
- [ ] Embeds the summary + key_context for searchability
- [ ] Commits to the store: `"engram: snapshot session {session_id}"`
- [ ] Returns JSON with `session_id`, `path`, `tier: "active"`, and `committed: true`
- [ ] `cargo test -p engram-mcp` passes

#### SmartRouter Sidecar

#### US-002-013: Implement index partitioning for knowledge items
**Description:** As an AI agent, I need the HNSW index to be partitioned so that code chunks and knowledge items can be queried separately in parallel.

**Acceptance Criteria:**
- [ ] `engram-query` `IndexManager` partitions the HNSW index into logical partitions: partition 0 (code chunks), partition 1 (doc chunks), partition 2 (knowledge items), partition 3 (snapshots)
- [ ] Partition is determined by `ChunkKind`: functions/classes/methods/types/impls/modules → 0, doc_sections/readmes/comment_blocks → 1, decisions/lessons/patterns/glossary/onboarding → 2, snapshots → 3
- [ ] Search can be filtered to specific partitions
- [ ] `cargo test -p engram-query` passes

#### US-002-014: Implement sidecar knowledge search in `engram_search`
**Description:** As an AI agent, I need `engram_search` to run parallel sidecar queries against the knowledge partition so that agents see relevant institutional knowledge alongside code results.

**Acceptance Criteria:**
- [ ] `engram_search` now runs two parallel queries: (1) code/doc query against partitions 0+1, (2) knowledge query against partition 2
- [ ] Response includes both `code_results` and `knowledge_results` sections matching architecture doc section 20.2
- [ ] `knowledge_results` include: `kind` (decision/lesson/pattern/glossary/onboarding), `id`, `title`, `relevance_score`, `related_files`, `created_at`, and kind-specific fields (`status` for decisions, `trigger` for lessons)
- [ ] Knowledge results below `min_relevance` threshold (default 0.6) are dropped
- [ ] Recency boost (default 1.1x) applied to knowledge items less than 30 days old
- [ ] `scope` parameter controls sidecar behavior: "all" → both, "code" → code only, "docs" → docs only, "knowledge" → knowledge only
- [ ] `meta` includes `knowledge_count` field
- [ ] `cargo test -p engram-mcp` passes

#### Onboarding

#### US-002-015: Implement project metadata detection
**Description:** As an AI agent, I need project metadata detection so that onboarding can identify the language, framework, and package manager of a project.

**Acceptance Criteria:**
- [ ] `engram-ingest` exports `fn detect_project_metadata(repo_path: &Path) -> Result<ProjectOverview>`
- [ ] Detects language from file extensions (majority wins) and config files (`Cargo.toml` → Rust, `package.json` → TypeScript/JavaScript, `pyproject.toml`/`setup.py` → Python, `go.mod` → Go)
- [ ] Detects framework from dependency files (e.g., `@nestjs/core` → NestJS, `react` → React, `actix-web` → Actix, `django` → Django)
- [ ] Detects package manager from lock files (`pnpm-lock.yaml` → pnpm, `yarn.lock` → yarn, `package-lock.json` → npm, `Cargo.lock` → cargo)
- [ ] Detects repo type: monorepo (multiple `package.json` or workspace members) vs single project
- [ ] Finds entry points by convention (`src/main.ts`, `src/lib.rs`, `src/index.ts`, `main.py`, `app.py`)
- [ ] `cargo test -p engram-ingest` passes

#### US-002-016: Implement build/test command extraction
**Description:** As an AI agent, I need build/test command extraction so that onboarding discovers how to build, test, lint, and run a project.

**Acceptance Criteria:**
- [ ] `engram-ingest` exports `fn extract_build_commands(repo_path: &Path) -> Result<BuildTestCommands>`
- [ ] For Node.js projects: reads `package.json` `scripts` field for `build`, `test`, `lint`, `start`/`dev`/`start:dev`
- [ ] For Rust projects: defaults to `cargo build`, `cargo test`, `cargo clippy`, `cargo run`
- [ ] For Python projects: checks for `pytest`, `tox`, `flake8`/`ruff`, `python -m`
- [ ] Detects test framework from config files (e.g., `jest.config.ts` → Jest, `pytest.ini` → pytest)
- [ ] `cargo test -p engram-ingest` passes

#### US-002-017: Implement directory structure analysis
**Description:** As an AI agent, I need directory structure analysis so that onboarding maps key directories and their purposes.

**Acceptance Criteria:**
- [ ] `engram-ingest` exports `fn analyze_directory_structure(repo_path: &Path, source_config: &SourceConfig) -> Result<ArchitectureMap>`
- [ ] Identifies key directories by naming conventions: `src/auth/` → authentication, `src/api/` → API routes, `src/common/` or `src/utils/` → shared utilities, `src/models/` → data models, `tests/` → tests, `docs/` → documentation
- [ ] Lists the top 5 most important files per directory (by size, import count, or naming convention)
- [ ] Respects include/exclude globs from source config
- [ ] `cargo test -p engram-ingest` passes

#### US-002-018: Implement key abstractions extraction
**Description:** As an AI agent, I need key abstractions extraction so that onboarding identifies the most important types, classes, and patterns.

**Acceptance Criteria:**
- [ ] `engram-ingest` exports `fn extract_key_abstractions(repo_path: &Path, language: &str) -> Result<KeyAbstractions>`
- [ ] Uses tree-sitter to find exported types, interfaces, classes, and traits
- [ ] Prioritizes by: (1) items exported from entry point files, (2) items referenced in many files, (3) items in high-level directories
- [ ] Extracts a brief description from adjacent doc comments if available
- [ ] Limits to top ~50 abstractions to avoid information overload
- [ ] `cargo test -p engram-ingest` passes

#### US-002-019: Implement full onboarding pipeline (`engram onboard`)
**Description:** As an AI agent, I need the full onboarding pipeline so that `engram onboard` runs end-to-end and produces knowledge files.

**Acceptance Criteria:**
- [ ] `engram-ingest` exports `async fn run_onboarding(repo_path: &Path, store: &Store, config: &StoreConfig, depth: OnboardingDepth) -> Result<OnboardingReport>`
- [ ] `OnboardingDepth` enum: `Quick`, `Standard`, `Deep`
- [ ] `Quick`: runs project metadata + build commands + directory overview only (~5s)
- [ ] `Standard` (default): Quick + key abstractions + architecture map (~30s)
- [ ] `Deep`: Standard + full symbol export analysis + cross-file dependency mapping (~2-5min) — tree-sitter only in Phase 2, LSP in Phase 5
- [ ] Writes YAML files to `knowledge/onboarding/`: `project-overview.yaml`, `build-test-commands.yaml`, `architecture-map.yaml`, `key-abstractions.yaml`
- [ ] Embeds all onboarding knowledge files for searchability
- [ ] Commits to the store: `"engram: onboard {project_name} ({N} knowledge files, {M} key abstractions identified)"`
- [ ] `OnboardingReport` contains: `depth`, `files_created`, `abstractions_found`, `duration_ms`
- [ ] Onboarding is idempotent — re-running overwrites existing files
- [ ] `cargo test -p engram-ingest` passes

#### US-002-020: Implement `engram_onboard` MCP tool
**Description:** As an AI agent, I need the `engram_onboard` MCP tool so that agents can trigger onboarding from within a session.

**Acceptance Criteria:**
- [ ] Tool name: `engram_onboard`
- [ ] Optional parameters: `repo: Option<String>` (source repo name), `depth: Option<String>` (quick/standard/deep, default standard)
- [ ] Calls the onboarding pipeline
- [ ] Returns JSON with the `OnboardingReport` fields
- [ ] `cargo test -p engram-mcp` passes

#### Self-Regulation Tools

#### US-002-021: Implement `engram_assess_context` MCP tool
**Description:** As an AI agent, I need the `engram_assess_context` tool so that agents can self-evaluate whether they have gathered enough context before proceeding.

**Acceptance Criteria:**
- [ ] Tool name: `engram_assess_context`
- [ ] Optional parameters: `task_description: Option<String>`
- [ ] Returns a structured prompt containing: (1) summary of searches performed so far in the session (count, queries), (2) total tokens consumed by search results, (3) count and types of knowledge items surfaced, (4) a structured question asking the agent to evaluate sufficiency
- [ ] Tracks session search history in-memory (list of queries and result counts)
- [ ] Does NOT perform any search itself — it's a reflection tool
- [ ] `cargo test -p engram-mcp` passes

#### US-002-022: Implement `engram_check_staleness` MCP tool
**Description:** As an AI agent, I need the `engram_check_staleness` tool so that agents can assess whether their retrieved context is still current.

**Acceptance Criteria:**
- [ ] Tool name: `engram_check_staleness`
- [ ] No required parameters
- [ ] Returns staleness information for all chunks retrieved so far in the session: chunk_id, file, last indexed commit, current HEAD of source repo, is_stale boolean
- [ ] Includes a structured prompt asking the agent to decide if stale context is acceptable or requires a reindex
- [ ] `cargo test -p engram-mcp` passes

#### CLI Additions

#### US-002-023: Implement `engram onboard` CLI command
**Description:** As an AI agent, I need the `engram onboard` CLI command so that users can trigger onboarding from the terminal.

**Acceptance Criteria:**
- [ ] `engram onboard [--depth quick|standard|deep] [--repo <name>]`
- [ ] Calls the onboarding pipeline
- [ ] Prints progress indicators for each step
- [ ] Prints the `OnboardingReport` summary on completion
- [ ] `cargo test -p engram-cli` passes

#### US-002-024: Implement `engram compact` CLI command
**Description:** As an AI agent, I need the `engram compact` CLI command so that users can trigger snapshot tier promotion.

**Acceptance Criteria:**
- [ ] `engram compact` runs snapshot compaction
- [ ] Prints the `CompactReport` (counts of snapshots promoted per tier)
- [ ] `cargo test -p engram-cli` passes

---

### Phase 3 — Multi-Repo, Sync & Contexts

**Goal:** Index multiple source repos into a single store with cross-repo symbol resolution. Remote store push/pull. GitHub Agentic Workflow for CI reindexing. Full context/mode system with custom YAML support.

#### Multi-Repo Indexing

#### US-003-001: Extend ingest pipeline for multi-source-repo support
**Description:** As an AI agent, I need the ingest pipeline to support multiple source repos so that a single Engram store can index code from several repositories.

**Acceptance Criteria:**
- [ ] `IngestPipeline::run` accepts a list of `SourceConfig` entries and processes each independently
- [ ] Each source repo's chunks are stored under `index/{source_name}/` in the store
- [ ] `manifest.json` `source_repos` field lists all indexed source repo names
- [ ] Incremental indexing tracks `last_indexed_commit` per source repo independently
- [ ] Source repos can have independent include/exclude patterns
- [ ] `cargo test -p engram-ingest` passes (test with two temp source repos)

#### US-003-002: Add `source_repo` filter to metadata index
**Description:** As an AI agent, I need source repo filtering in the metadata index so that searches can be scoped to a specific repo.

**Acceptance Criteria:**
- [ ] `MetadataIndex` adds a `source_repo → Vec<ChunkMetadata>` map
- [ ] `MetadataIndex::lookup_by_repo(repo: &str) -> Vec<&ChunkMetadata>` returns all chunks for a repo
- [ ] `engram_search` gains a `repo: Option<String>` parameter to scope search to a specific source repo
- [ ] When `repo` is specified, only chunks from that repo are included in results
- [ ] `cargo test -p engram-query` passes

#### Cross-Repo Symbol Resolution

#### US-003-003: Implement tree-sitter export extraction
**Description:** As an AI agent, I need tree-sitter-based export extraction so that cross-repo references can be resolved by matching imports against known exports.

**Acceptance Criteria:**
- [ ] `engram-ingest` implements the `SymbolResolver` trait's `extract_exports` method using tree-sitter
- [ ] For TypeScript: extracts `export` declarations (named exports, default exports, re-exports)
- [ ] For Rust: extracts `pub` items from module boundaries
- [ ] For Python: extracts items in `__all__` or top-level function/class definitions
- [ ] `ExportedSymbol` includes: `name`, `kind` (type/function/class/const), `file`, `chunk_id`
- [ ] Results are written to `index/_xrefs/exports.jsonl`
- [ ] `cargo test -p engram-ingest` passes

#### US-003-004: Implement tree-sitter import resolution
**Description:** As an AI agent, I need tree-sitter-based import resolution so that cross-repo references can link importing files to their source exports.

**Acceptance Criteria:**
- [ ] `engram-ingest` implements the `SymbolResolver` trait's `resolve_imports` method using tree-sitter
- [ ] For TypeScript: parses `import` statements and resolves paths using `tsconfig.json` path aliases and `package.json` dependency mappings
- [ ] For Rust: parses `use` statements and resolves against Cargo workspace members
- [ ] For Python: parses `import`/`from...import` and resolves against known packages
- [ ] Cross-repo resolution uses heuristic matching: import paths matched against known exports using package dependency mappings
- [ ] `ResolvedImport` includes: `symbol`, `importing_repo`, `importing_file`, `source_repo`, `source_file`, `resolved_chunk`, `resolution: "heuristic"`
- [ ] Results are written to `index/_xrefs/imports.jsonl`
- [ ] `cargo test -p engram-ingest` passes

#### US-003-005: Build cross-repo symbol graph
**Description:** As an AI agent, I need the cross-repo symbol graph so that `engram_graph` can traverse dependencies across repository boundaries.

**Acceptance Criteria:**
- [ ] `engram-query` builds an in-memory directed graph from `_xrefs/exports.jsonl` and `_xrefs/imports.jsonl`
- [ ] Graph edges represent: "repo A file X imports symbol Y from repo B file Z"
- [ ] Graph supports traversal in both directions (callers/callees)
- [ ] Graph is built during boot alongside HNSW and BM25 indexes
- [ ] Graph is included in the compiled cache for fast boot
- [ ] `cargo test -p engram-query` passes

#### US-003-006: Implement `engram_graph` MCP tool
**Description:** As an AI agent, I need the `engram_graph` MCP tool so that agents can explore dependency/reference graphs for symbols across repos.

**Acceptance Criteria:**
- [ ] Tool name: `engram_graph`
- [ ] Required parameters: `symbol: String`
- [ ] Optional parameters: `direction: Option<String>` (callers/callees/both, default both), `depth: Option<u32>` (default 2)
- [ ] Returns a JSON graph structure with nodes (chunk_id, name, file, repo) and edges (source, target, relationship type)
- [ ] Cross-repo edges are clearly labeled with source and target repo names
- [ ] `cargo test -p engram-mcp` passes

#### US-003-007: Implement `engram_related` MCP tool
**Description:** As an AI agent, I need the `engram_related` MCP tool so that agents can find chunks semantically related to a given chunk or symbol.

**Acceptance Criteria:**
- [ ] Tool name: `engram_related`
- [ ] Parameters: `chunk_id: Option<String>`, `symbol: Option<String>` (one required), `top_k: Option<u32>` (default 10)
- [ ] If `chunk_id` is provided, looks up the chunk's embedding and finds nearest neighbors
- [ ] If `symbol` is provided, looks up all chunks for that symbol, averages embeddings, and finds nearest neighbors
- [ ] Results exclude the input chunk(s) themselves
- [ ] Returns same format as `engram_search` code results
- [ ] `cargo test -p engram-mcp` passes

#### Remote Store & Sync

#### US-003-008: Implement remote store initialization
**Description:** As an AI agent, I need remote store initialization so that `engram init --remote <url>` clones or creates a remote-backed store.

**Acceptance Criteria:**
- [ ] `engram-store` implements `Store::init_remote(url: &str, local_path: &Path, config: &StoreConfig) -> Result<Store>`
- [ ] If the remote repo exists, clones it to `local_path`
- [ ] If the remote repo is empty, initializes local store and pushes the initial commit
- [ ] Stores the remote URL in `engram.config.yaml` `store.remote` field
- [ ] `cargo test -p engram-store` passes (test with a local bare repo as "remote")

#### US-003-009: Implement `engram_sync` pull/push
**Description:** As an AI agent, I need sync functionality so that local and remote stores stay in sync.

**Acceptance Criteria:**
- [ ] `engram-store` exports `fn sync_pull(store: &Store) -> Result<SyncReport>` that runs `git pull` on the store repo
- [ ] `engram-store` exports `fn sync_push(store: &Store) -> Result<SyncReport>` that runs `git push` on the store repo
- [ ] `SyncReport` includes: `direction`, `commits_transferred`, `conflicts: Vec<String>`
- [ ] Conflicts on individual chunk files are resolved via last-write-wins (the file with the newer `indexed_at` timestamp wins)
- [ ] Conflicts on `manifest.json` are resolved by merging: take the higher chunk count and the more recent `updated_at`
- [ ] Uses `git2` for all git operations (no shell-out)
- [ ] `cargo test -p engram-store` passes

#### US-003-010: Implement `engram_sync` MCP tool
**Description:** As an AI agent, I need the `engram_sync` MCP tool so that agents can trigger store sync within a session.

**Acceptance Criteria:**
- [ ] Tool name: `engram_sync`
- [ ] Optional parameters: `direction: Option<String>` (pull/push/both, default both)
- [ ] Calls the sync functions from `engram-store`
- [ ] Returns JSON with the `SyncReport` fields
- [ ] `cargo test -p engram-mcp` passes

#### US-003-011: Implement `engram sync` CLI command
**Description:** As an AI agent, I need the `engram sync` CLI command so that users can sync from the terminal.

**Acceptance Criteria:**
- [ ] `engram sync [--pull | --push | --both]`
- [ ] Default is `--both`
- [ ] Prints the `SyncReport` summary
- [ ] If no remote is configured, prints an error suggesting `engram init --remote`
- [ ] `cargo test -p engram-cli` passes

#### US-003-012: Implement `engram init --remote` CLI command
**Description:** As an AI agent, I need the remote init CLI variant so that users can create a remote-backed store.

**Acceptance Criteria:**
- [ ] `engram init --remote <url> [--path <dir>]`
- [ ] Calls `Store::init_remote`
- [ ] Prints success message with store path and remote URL
- [ ] `cargo test -p engram-cli` passes

#### GitHub Agentic Workflow

#### US-003-013: Create GitHub Agentic Workflow definition for remote reindexing
**Description:** As an AI agent, I need the GitHub Agentic Workflow definition so that remote stores can be reindexed automatically on push to main.

**Acceptance Criteria:**
- [ ] File `gh-aw/engram-reindex.md` contains the workflow definition in the format expected by GitHub Agentic Workflows
- [ ] Workflow triggers on: push to `main`, and on daily schedule (2am UTC)
- [ ] Workflow steps: (1) clone source repo(s) and engram store repo, (2) run `engram reindex --full`, (3) run `engram analyze` to update knowledge graph, (4) flag stale knowledge items referencing deleted files, (5) commit and push changes to the store, (6) if >50 chunks updated, open a summary PR on the store repo
- [ ] Workflow definition is clear enough for GitHub's agentic system to execute

#### Context & Mode System

#### US-003-014: Implement context system with tool surface control
**Description:** As an AI agent, I need the context system so that Engram adapts its exposed MCP tool set based on the client environment.

**Acceptance Criteria:**
- [ ] `engram-mcp` reads the `--context` flag (or `context` config value) on startup
- [ ] Built-in contexts: `default` (all tools), `claude-code` (disable duplicate tools like `engram_lookup` for simple file reads), `cursor` (full tools + forced compact mode), `ci` (disable dashboard, snapshot, self-regulation tools; enable benchmark auto-logging), `ide-assistant` (reduced tool set, knowledge sidecar disabled by default)
- [ ] `tools/list` MCP response only includes tools allowed by the active context
- [ ] Context is fixed for the session duration (cannot be changed)
- [ ] `cargo test -p engram-mcp` passes

#### US-003-015: Implement mode system with dynamic switching
**Description:** As an AI agent, I need the mode system so that search behavior adapts to the current task type.

**Acceptance Criteria:**
- [ ] `engram-mcp` tracks active modes (default: `["explore"]`)
- [ ] Built-in modes: `explore` (standard behavior), `edit` (boost lessons/patterns, raise min_relevance), `plan` (boost decisions/patterns, increase knowledge top_k, auto-suggest `engram_assess_context`), `onboard` (run onboarding if not done, surface onboarding knowledge prominently), `benchmark` (enable auto-logging, add token counting)
- [ ] Multiple modes can be active simultaneously
- [ ] Mode affects: knowledge sidecar `top_k`, `min_relevance`, boost factors for specific knowledge kinds
- [ ] `cargo test -p engram-mcp` passes

#### US-003-016: Implement `engram_switch_mode` MCP tool
**Description:** As an AI agent, I need the `engram_switch_mode` tool so that agents can change modes during a session.

**Acceptance Criteria:**
- [ ] Tool name: `engram_switch_mode`
- [ ] Required parameters: `modes: Vec<String>` (list of mode names to activate)
- [ ] Validates mode names against built-in + custom modes
- [ ] Returns JSON with `active_modes: Vec<String>` and `behavior_changes: String` (human-readable summary of what changed)
- [ ] `cargo test -p engram-mcp` passes

#### US-003-017: Implement `engram_get_config` MCP tool
**Description:** As an AI agent, I need the `engram_get_config` tool so that agents can inspect the current context, modes, and tool surface.

**Acceptance Criteria:**
- [ ] Tool name: `engram_get_config`
- [ ] No required parameters
- [ ] Returns JSON with: `context: String`, `active_modes: Vec<String>`, `available_tools: Vec<String>`, `search_config: {hybrid_alpha, default_top_k, knowledge: {top_k, min_relevance}}`, `embedding_provider: String`
- [ ] `cargo test -p engram-mcp` passes

#### US-003-018: Implement custom context and mode YAML definitions
**Description:** As an AI agent, I need custom context and mode support so that users can define their own contexts and modes via YAML files.

**Acceptance Criteria:**
- [ ] On startup, `engram-mcp` scans `.engram/contexts/` and `.engram/modes/` in the store for YAML files
- [ ] Custom context YAML format: `name`, `description`, `tools.exclude: Vec<String>`, `search.compact_by_default: bool`, `search.knowledge_sidecar: bool`
- [ ] Custom mode YAML format: `name`, `description`, `search.knowledge.top_k`, `search.knowledge.min_relevance`, `search.knowledge.boost_decisions`, `search.knowledge.boost_patterns`
- [ ] Custom contexts and modes are available alongside built-in ones
- [ ] If a custom name conflicts with a built-in name, the custom definition wins
- [ ] `cargo test -p engram-mcp` passes

#### SSE Transport

#### US-003-019: Implement SSE transport for MCP server
**Description:** As an AI agent, I need SSE transport so that Engram can serve as an MCP server over HTTP with Server-Sent Events.

**Acceptance Criteria:**
- [ ] `engram serve --transport sse --port <port>` starts an HTTP server using `axum`
- [ ] The SSE endpoint follows the MCP SSE transport specification
- [ ] Multiple concurrent MCP clients can connect via SSE
- [ ] The server handles JSON-RPC requests and sends responses/notifications via SSE
- [ ] `cargo test -p engram-mcp` passes

---

### Phase 4 — Benchmark, Dashboard & Proof

**Goal:** Full benchmark harness comparing baseline vs. Engram-assisted agent performance. Live web dashboard for operational observability. Metrics committed to the store for trend analysis.

#### Benchmark Data Types

#### US-004-001: Define benchmark session and metric types
**Description:** As an AI agent, I need benchmark data types so that benchmark sessions, events, and reports share a common schema.

**Acceptance Criteria:**
- [ ] `engram-core` exports `BenchmarkSession` struct with fields: `session_id: String`, `mode: BenchmarkMode` (Baseline/Assisted), `task_description: String`, `started_at: String`, `ended_at: Option<String>`, `task_outcome: Option<String>`, `notes: Option<String>`
- [ ] `engram-core` exports `BenchmarkEvent` struct with fields: `timestamp: String`, `event_type: BenchmarkEventType` (Search/FileRead/ToolCall), `query: Option<String>`, `tokens_used: u64`, `files_read: Vec<String>`, `chunks_returned: Option<u32>`, `hit: Option<bool>`
- [ ] `engram-core` exports `BenchmarkReport` struct with fields: `session_id: String`, `mode: BenchmarkMode`, `token_efficiency: u64` (total tokens), `retrieval_precision: f64`, `retrieval_recall: f64`, `time_to_first_edit_ms: Option<u64>`, `file_read_count: u32`, `search_to_read_ratio: f64`, `context_waste_tokens: u64`
- [ ] All types derive `Serialize`, `Deserialize`, `Debug`, `Clone`
- [ ] `cargo test -p engram-core` passes

#### Benchmark Harness

#### US-004-002: Implement benchmark session lifecycle in engram-bench
**Description:** As an AI agent, I need the benchmark session lifecycle so that sessions can be started, events logged, and metrics computed on end.

**Acceptance Criteria:**
- [ ] `engram-bench` exports `BenchmarkHarness` struct with methods: `start(mode: BenchmarkMode, task_description: &str) -> Result<String>` (returns session_id), `log_event(session_id: &str, event: BenchmarkEvent) -> Result<()>`, `end(session_id: &str, task_outcome: &str, notes: Option<&str>) -> Result<BenchmarkReport>`
- [ ] `start` creates a session and begins tracking
- [ ] `log_event` appends to an in-memory event log for the session
- [ ] `end` computes all metrics from the event log, generates a `BenchmarkReport`, and writes it to `metrics/runs/{timestamp}_{session_id}.jsonl` in the store
- [ ] Multiple sessions can be active simultaneously
- [ ] `cargo test -p engram-bench` passes

#### US-004-003: Implement metric computation from event logs
**Description:** As an AI agent, I need metric computation so that raw benchmark events produce meaningful metrics.

**Acceptance Criteria:**
- [ ] `engram-bench` exports `fn compute_metrics(events: &[BenchmarkEvent], session: &BenchmarkSession) -> BenchmarkReport`
- [ ] `token_efficiency`: sum of all `tokens_used` across events
- [ ] `retrieval_precision`: count of events where `hit == true` / total search events
- [ ] `retrieval_recall`: requires ground truth (computed when comparison data is available, otherwise `NaN`)
- [ ] `time_to_first_edit_ms`: time from session start to first `FileRead` or `ToolCall` event that looks like an edit
- [ ] `file_read_count`: count of `FileRead` events
- [ ] `search_to_read_ratio`: count of `Search` events / count of `FileRead` events
- [ ] `context_waste_tokens`: sum of `tokens_used` for events where `hit == false`
- [ ] `cargo test -p engram-bench` passes

#### US-004-004: Implement baseline vs. assisted comparison report generation
**Description:** As an AI agent, I need comparison report generation so that baseline and assisted benchmark sessions can be compared side-by-side.

**Acceptance Criteria:**
- [ ] `engram-bench` exports `fn compare(baseline: &BenchmarkReport, assisted: &BenchmarkReport) -> ComparisonReport`
- [ ] `ComparisonReport` includes: metric-by-metric comparison with absolute and percentage differences
- [ ] Positive differences (assisted better than baseline) are clearly marked
- [ ] Report is serialized as YAML and committed to `metrics/` in the store
- [ ] `cargo test -p engram-bench` passes

#### Benchmark MCP Tools

#### US-004-005: Implement `engram_benchmark_start` MCP tool
**Description:** As an AI agent, I need the `engram_benchmark_start` tool so that agents can begin a benchmarked session.

**Acceptance Criteria:**
- [ ] Tool name: `engram_benchmark_start`
- [ ] Required parameters: `mode: String` (baseline/assisted), `task_description: String`
- [ ] Creates a new benchmark session via `BenchmarkHarness`
- [ ] In `benchmark` mode or when auto-logging is enabled, all subsequent `engram_search` calls auto-log events
- [ ] Returns JSON with `session_id` and `mode`
- [ ] `cargo test -p engram-mcp` passes

#### US-004-006: Implement `engram_benchmark_log` MCP tool
**Description:** As an AI agent, I need the `engram_benchmark_log` tool so that agents can explicitly log retrieval events during a benchmark.

**Acceptance Criteria:**
- [ ] Tool name: `engram_benchmark_log`
- [ ] Required parameters: `query: String`, `tokens_used: u64`, `files_read: Vec<String>`
- [ ] Optional parameters: `hit: Option<bool>`
- [ ] Logs the event to the active benchmark session
- [ ] Returns JSON with `logged: true` and `event_count` (total events so far)
- [ ] If no active session, returns an error
- [ ] `cargo test -p engram-mcp` passes

#### US-004-007: Implement `engram_benchmark_end` MCP tool
**Description:** As an AI agent, I need the `engram_benchmark_end` tool so that agents can end a benchmark session and see results.

**Acceptance Criteria:**
- [ ] Tool name: `engram_benchmark_end`
- [ ] Required parameters: `task_outcome: String` (success/failure/partial)
- [ ] Optional parameters: `notes: Option<String>`
- [ ] Ends the session, computes metrics, commits report to store
- [ ] Returns the full `BenchmarkReport` as JSON
- [ ] `cargo test -p engram-mcp` passes

#### US-004-008: Implement automatic search event logging in benchmark mode
**Description:** As an AI agent, I need automatic logging so that all `engram_search` calls are automatically captured as benchmark events when a session is active.

**Acceptance Criteria:**
- [ ] When a benchmark session is active and `mode` is `assisted`, every `engram_search` call auto-logs: query text, tokens in returned context (estimated from result size), chunks returned count
- [ ] When `mode` is `baseline`, `engram_search` is disabled (returns an error telling the agent to use native tools)
- [ ] Auto-logging does not affect the search results returned to the agent
- [ ] `cargo test -p engram-mcp` passes

#### Live Dashboard

#### US-004-009: Implement dashboard HTTP server in engram-dashboard
**Description:** As an AI agent, I need the dashboard HTTP server so that the web UI is served by the Engram process.

**Acceptance Criteria:**
- [ ] `engram-dashboard` crate implements an `axum` HTTP server
- [ ] Server is started alongside the MCP server when `dashboard.enabled` is true
- [ ] Serves on configurable port (default 3200)
- [ ] Serves static HTML/JS/CSS files bundled into the binary using `include_dir` or `rust-embed`
- [ ] Dashboard accessible at `http://localhost:{port}/dashboard`
- [ ] If `dashboard.enabled` is false or context is `ci`, the server is not started
- [ ] `cargo test -p engram-dashboard` passes

#### US-004-010: Implement WebSocket endpoint for real-time dashboard updates
**Description:** As an AI agent, I need a WebSocket endpoint so that the dashboard can receive live updates.

**Acceptance Criteria:**
- [ ] `engram-dashboard` exposes a WebSocket endpoint at `/ws`
- [ ] Events are broadcast to all connected WebSocket clients: search queries (query text, result count, time), knowledge write-backs, reindex progress, session connect/disconnect
- [ ] Events are JSON-serialized with a `type` field for client-side routing
- [ ] WebSocket handles multiple concurrent clients
- [ ] `cargo test -p engram-dashboard` passes

#### US-004-011: Build dashboard frontend — Live Session View
**Description:** As an AI agent, I need the Live Session View page so that users can see active MCP connections and real-time search activity.

**Acceptance Criteria:**
- [ ] Static HTML/JS page that connects to the WebSocket endpoint
- [ ] Shows active MCP connections (client name, connected since, request count)
- [ ] Shows real-time stream of search queries with: query text, result count, search time, knowledge sidecar hit/miss
- [ ] Shows current active context and modes
- [ ] Shows session token consumption (estimated from search result sizes)
- [ ] Auto-scrolls the query stream
- [ ] No build step required (vanilla JS or bundled at compile time)

#### US-004-012: Build dashboard frontend — Index Health View
**Description:** As an AI agent, I need the Index Health View page so that users can see the state of the semantic index.

**Acceptance Criteria:**
- [ ] Shows total chunks indexed per source repo (bar chart or table)
- [ ] Shows staleness map: list of files with stale chunks, sorted by staleness
- [ ] Shows last reindex time and duration per source repo
- [ ] Shows embedding provider status (name, model, reachable yes/no)
- [ ] Shows cache status (hit/miss on last boot, cache size on disk)
- [ ] Data loaded via REST API endpoints from `engram-dashboard`

#### US-004-013: Build dashboard frontend — Knowledge Activity View
**Description:** As an AI agent, I need the Knowledge Activity View so that users can see recent knowledge write-backs and knowledge item counts.

**Acceptance Criteria:**
- [ ] Shows recent write-backs (decisions, lessons, patterns) with timestamps and titles
- [ ] Shows knowledge item count by category (decisions, lessons, patterns, glossary, onboarding)
- [ ] Shows onboarding status per source repo (onboarded yes/no, depth, last run)
- [ ] Data loaded via REST API endpoints

#### US-004-014: Build dashboard frontend — Search Analytics View
**Description:** As an AI agent, I need the Search Analytics View so that users can understand search patterns and quality.

**Acceptance Criteria:**
- [ ] Shows query frequency over time (line chart)
- [ ] Shows average result relevance scores over time
- [ ] Shows cache hit rate (compiled index cache)
- [ ] Shows SmartRouter sidecar hit rate (how often knowledge results are above min_relevance)
- [ ] Data aggregated from WebSocket events and persisted in-memory during the session

#### US-004-015: Build dashboard frontend — Benchmark Dashboard View
**Description:** As an AI agent, I need the Benchmark Dashboard View so that users can see benchmark results and trends.

**Acceptance Criteria:**
- [ ] Shows active benchmark sessions (if any) with live event counts
- [ ] Shows historical comparison trends: token savings over time (line chart from committed metrics)
- [ ] Shows per-task metrics drill-down (clickable table of past sessions)
- [ ] Reads data from `metrics/runs/` in the store via REST API endpoints

#### US-004-016: Implement REST API endpoints for dashboard data
**Description:** As an AI agent, I need REST API endpoints so that the dashboard frontend can fetch data.

**Acceptance Criteria:**
- [ ] `GET /api/status` — returns index health data (chunk counts, staleness, provider status)
- [ ] `GET /api/knowledge` — returns knowledge item counts and recent write-backs
- [ ] `GET /api/benchmarks` — returns list of benchmark runs with summary metrics
- [ ] `GET /api/benchmarks/{session_id}` — returns detailed metrics for a specific run
- [ ] `GET /api/sessions` — returns active MCP session info
- [ ] All endpoints return JSON
- [ ] `cargo test -p engram-dashboard` passes

#### CLI Additions

#### US-004-017: Implement `engram benchmark run` CLI command
**Description:** As an AI agent, I need the benchmark CLI so that users can run baseline vs. assisted comparisons from the terminal.

**Acceptance Criteria:**
- [ ] `engram benchmark run --task "description" --baseline --assisted`
- [ ] Orchestrates: (1) start baseline session, (2) run task with engram_search disabled, (3) end baseline, (4) start assisted session, (5) run task with engram_search enabled, (6) end assisted, (7) generate comparison report
- [ ] Prints the comparison report summary
- [ ] `cargo test -p engram-cli` passes

#### US-004-018: Implement `engram benchmark report` CLI command
**Description:** As an AI agent, I need the benchmark report CLI so that users can view past benchmark results.

**Acceptance Criteria:**
- [ ] `engram benchmark report [--last | --all | --compare <id1> <id2>]`
- [ ] `--last`: shows the most recent benchmark run
- [ ] `--all`: lists all runs with summary metrics
- [ ] `--compare`: generates a side-by-side comparison of two specific runs
- [ ] Output is a human-readable table
- [ ] `cargo test -p engram-cli` passes

---

### Phase 5 — Advanced Providers, LSP & Ecosystem

**Goal:** Additional embedding providers (OpenAI, Voyage, ONNX). Optional LSP backend for exact symbol resolution. Git hooks auto-install. File watcher mode. f16 embedding precision support. Cost estimation for paid providers.

#### Additional Embedding Providers

#### US-005-001: Implement OpenAI embedding provider
**Description:** As an AI agent, I need the OpenAI embedding provider so that users can use OpenAI's text-embedding models.

**Acceptance Criteria:**
- [ ] New crate `crates/engram-providers/openai/` with its own `Cargo.toml`
- [ ] Implements `EmbeddingProvider` trait from `engram-core`
- [ ] `name()` returns `"openai/{model_name}"` (e.g., `"openai/text-embedding-3-small"`)
- [ ] `embed()` sends POST requests to `https://api.openai.com/v1/embeddings` (or custom `base_url` for Azure/proxies)
- [ ] Reads API key from environment variable specified in `openai.api_key_env` config, with fallback to `openai.api_key_file`
- [ ] Supports `dimensions` override for `text-embedding-3-large` (dimension reduction)
- [ ] Handles rate limiting: respects `Retry-After` header, returns `EmbedError::RateLimited` with `retry_after_ms`
- [ ] Supports configurable `request_timeout_ms` and `max_retries`
- [ ] Batches texts up to the provider's max batch size
- [ ] `cargo test -p engram-provider-openai` passes (unit tests with mocked HTTP)

#### US-005-002: Implement Voyage embedding provider
**Description:** As an AI agent, I need the Voyage embedding provider so that users can use Voyage AI's code-optimized models.

**Acceptance Criteria:**
- [ ] New crate `crates/engram-providers/voyage/` with its own `Cargo.toml`
- [ ] Implements `EmbeddingProvider` trait
- [ ] `name()` returns `"voyage/{model_name}"` (e.g., `"voyage/voyage-code-3"`)
- [ ] `embed()` sends POST requests to Voyage API (`https://api.voyageai.com/v1/embeddings` or custom base_url)
- [ ] Reads API key from `voyage.api_key_env` config
- [ ] Handles rate limiting and retries
- [ ] `cargo test -p engram-provider-voyage` passes (unit tests with mocked HTTP)

#### US-005-003: Implement ONNX in-process embedding provider
**Description:** As an AI agent, I need the ONNX embedding provider so that users can run embeddings locally without Ollama, using ONNX Runtime.

**Acceptance Criteria:**
- [ ] New crate `crates/engram-providers/onnx/` with its own `Cargo.toml`
- [ ] Implements `EmbeddingProvider` trait
- [ ] Uses the `ort` crate (ONNX Runtime Rust bindings)
- [ ] `name()` returns `"onnx/{model_name}"` (e.g., `"onnx/BAAI/bge-small-en-v1.5"`)
- [ ] On first use, downloads the model to `~/.engram/models/` (configurable cache dir) if not already cached
- [ ] Supports `quantized: true` config for INT8 quantized models
- [ ] Supports `device: "cpu"` and `device: "cuda"` config
- [ ] `embed()` runs inference in-process (no external service)
- [ ] Handles tokenization using the model's tokenizer (bundled with the ONNX model)
- [ ] `cargo test -p engram-provider-onnx` passes

#### US-005-004: Implement custom HTTP embedding provider
**Description:** As an AI agent, I need the custom HTTP provider so that users can point Engram at any embedding API.

**Acceptance Criteria:**
- [ ] New crate `crates/engram-providers/custom/` with its own `Cargo.toml`
- [ ] Implements `EmbeddingProvider` trait
- [ ] `name()` returns `"custom/{endpoint_host}"`
- [ ] `embed()` sends POST `{"texts": [...]}` to the configured `endpoint` URL and expects `{"embeddings": [[...]]}`
- [ ] Reads `dimensions`, `max_batch_size` from config
- [ ] Supports custom headers (e.g., `Authorization` with env var substitution via `${ENV_VAR}` syntax)
- [ ] `cargo test -p engram-provider-custom` passes (unit tests with mocked HTTP)

#### US-005-005: Implement provider switching detection and reindex guard
**Description:** As an AI agent, I need provider switching detection so that mismatched embeddings are caught and blocked until a full reindex is performed.

**Acceptance Criteria:**
- [ ] On boot, Engram compares the configured provider/model against `manifest.json`'s `model_name` and `dimensions`
- [ ] If they don't match, Engram logs a warning and blocks all search operations (returning an error: "Embedding model mismatch. Run `engram reindex --full` to reindex with the new model.")
- [ ] `engram_status` reports the mismatch
- [ ] After `engram reindex --full`, the manifest is updated and search is unblocked
- [ ] `cargo test -p engram-query` passes

#### US-005-006: Implement cost estimation for paid providers (`engram reindex --estimate`)
**Description:** As an AI agent, I need cost estimation so that users can preview the cost of a full reindex before committing.

**Acceptance Criteria:**
- [ ] `engram reindex --estimate` counts all chunks that would be re-embedded, estimates total tokens, and computes cost per provider
- [ ] Cost tables: OpenAI text-embedding-3-small ($0.02/1M tokens), OpenAI text-embedding-3-large ($0.13/1M tokens), Voyage voyage-code-3 (check current pricing)
- [ ] Output shows: estimated chunks, estimated tokens, estimated cost, estimated time
- [ ] Does NOT perform any embedding calls
- [ ] `cargo test -p engram-cli` passes

#### f16 Embedding Precision

#### US-005-007: Implement f16 embedding read/write support
**Description:** As an AI agent, I need f16 embedding support so that embedding storage can be reduced by 50% with minimal recall loss.

**Acceptance Criteria:**
- [ ] `write_embeddings_bin` supports `precision: 1` (f16) via the `half` crate
- [ ] When `storage.embedding_precision` config is `"f16"`, embeddings are converted from f32 to f16 before writing
- [ ] `read_embeddings_bin` detects precision from the header and converts f16 back to f32 for the HNSW index
- [ ] Precision flag in the binary header is correctly set (0 = f32, 1 = f16)
- [ ] Unit tests verify round-trip accuracy within acceptable tolerance (< 0.1% relative error)
- [ ] `cargo test -p engram-store` passes

#### LSP Symbol Resolution

#### US-005-008: Implement LSP client for symbol resolution
**Description:** As an AI agent, I need the LSP client so that exact symbol resolution is available as an alternative to tree-sitter heuristics.

**Acceptance Criteria:**
- [ ] New crate `crates/engram-lsp/` with its own `Cargo.toml`
- [ ] Implements the `SymbolResolver` trait from `engram-core` using LSP
- [ ] Starts language servers on-demand during indexing (TypeScript, Rust, Python, Go)
- [ ] Reads server commands from `symbol_resolution.lsp.servers` config
- [ ] Uses `lsp-types` crate for LSP protocol types
- [ ] Calls `textDocument/documentSymbol` for export extraction
- [ ] Calls `textDocument/references` for cross-file reference finding
- [ ] Calls `textDocument/typeHierarchy` for type hierarchy (where supported)
- [ ] Shuts down language servers after indexing completes (no persistent processes)
- [ ] Respects `startup_timeout_ms` config
- [ ] Cross-repo edges marked as `resolution: "exact"` (vs tree-sitter's `"heuristic"`)
- [ ] `cargo test -p engram-lsp` passes

#### US-005-009: Implement auto-install for language servers
**Description:** As an AI agent, I need auto-install so that users don't have to manually install language servers.

**Acceptance Criteria:**
- [ ] When `symbol_resolution.lsp.auto_install` is true and a configured language server is not found on PATH, Engram attempts to install it
- [ ] For TypeScript: runs `npm install -g typescript-language-server typescript`
- [ ] For Rust: checks for `rust-analyzer` (usually installed with rustup)
- [ ] For Python: runs `pip install python-lsp-server`
- [ ] For Go: runs `go install golang.org/x/tools/gopls@latest`
- [ ] If install fails, falls back to tree-sitter for that language with a warning
- [ ] `cargo test -p engram-lsp` passes

#### US-005-010: Integrate LSP with deep onboarding depth
**Description:** As an AI agent, I need LSP integration in deep onboarding so that deep onboarding produces exact cross-file dependency maps.

**Acceptance Criteria:**
- [ ] When `OnboardingDepth::Deep` is used and `symbol_resolution.backend` is `"lsp"`, the onboarding pipeline uses LSP for full symbol export analysis
- [ ] Deep onboarding produces richer `key-abstractions.yaml` with exact type hierarchy information
- [ ] If LSP is not configured, deep onboarding falls back to tree-sitter with a warning
- [ ] `cargo test -p engram-ingest` passes

#### Git Hooks Auto-Install

#### US-005-011: Implement `engram hooks install` CLI command
**Description:** As an AI agent, I need git hooks auto-install so that source repos automatically trigger incremental reindexing.

**Acceptance Criteria:**
- [ ] `engram hooks install` installs hooks into each configured source repo
- [ ] Installs `post-commit` hook: `engram reindex --incremental --commit HEAD`
- [ ] Installs `post-merge` hook: `engram reindex --incremental --commit HEAD`
- [ ] Installs `post-checkout` hook: `engram reindex --incremental --commit HEAD`
- [ ] If hooks already exist, appends Engram's command (does not overwrite)
- [ ] Prints which hooks were installed in which repos
- [ ] `cargo test -p engram-cli` passes

#### US-005-012: Implement auto-hook installation during `engram init`
**Description:** As an AI agent, I need hooks to be optionally installed during init so that new stores are immediately set up for auto-update.

**Acceptance Criteria:**
- [ ] `engram init --hooks` installs git hooks in all configured source repos after store creation
- [ ] When `hooks.install` config is true, hooks are installed automatically during init
- [ ] `cargo test -p engram-cli` passes

#### File Watcher

#### US-005-013: Implement file watcher for real-time indexing
**Description:** As an AI agent, I need the file watcher so that the index stays up-to-date in real-time during active development.

**Acceptance Criteria:**
- [ ] `engram-watch` crate uses the `notify` crate for cross-platform file watching
- [ ] Watches all configured source repo directories
- [ ] Debounces file change events (configurable, default 2000ms)
- [ ] On change: re-chunks and re-embeds changed files, updates the store in-memory (but does NOT auto-commit)
- [ ] Respects `watcher.ignore` patterns from config
- [ ] Respects source include/exclude patterns
- [ ] `cargo test -p engram-watch` passes

#### US-005-014: Implement `engram watch` CLI command
**Description:** As an AI agent, I need the watch CLI command so that users can start the file watcher standalone.

**Acceptance Criteria:**
- [ ] `engram watch` starts the file watcher in the foreground
- [ ] Prints file change events as they are detected and processed
- [ ] Can be stopped with Ctrl+C (graceful shutdown)
- [ ] If `watcher.enabled` is true in config, the watcher is also started alongside `engram serve`
- [ ] `cargo test -p engram-cli` passes

#### Additional Language Support

#### US-005-015: Implement tree-sitter chunking for Go
**Description:** As an AI agent, I need Go chunking so that `.go` files are properly indexed.

**Acceptance Criteria:**
- [ ] `TreeSitterChunker` supports Go language
- [ ] Extracts: `function_declaration`, `method_declaration`, `type_declaration` (struct, interface), `const_declaration`, `var_declaration`
- [ ] Method receivers are included in the chunk
- [ ] `cargo test -p engram-ingest` passes with Go test fixtures

#### US-005-016: Implement tree-sitter chunking for Java
**Description:** As an AI agent, I need Java chunking so that `.java` files are properly indexed.

**Acceptance Criteria:**
- [ ] `TreeSitterChunker` supports Java language
- [ ] Extracts: `class_declaration`, `method_declaration`, `interface_declaration`, `enum_declaration`, `constructor_declaration`
- [ ] Inner classes are chunked as part of the outer class
- [ ] Annotations are included with the annotated declaration
- [ ] `cargo test -p engram-ingest` passes with Java test fixtures

#### US-005-017: Implement tree-sitter chunking for C/C++
**Description:** As an AI agent, I need C/C++ chunking so that `.c`, `.cpp`, `.h`, `.hpp` files are properly indexed.

**Acceptance Criteria:**
- [ ] `TreeSitterChunker` supports C and C++ languages
- [ ] Extracts: `function_definition`, `struct_specifier`, `class_specifier`, `enum_specifier`, `namespace_definition`, `template_declaration`
- [ ] Header files (`.h`, `.hpp`) have their declarations chunked
- [ ] `cargo test -p engram-ingest` passes with C/C++ test fixtures

#### US-005-018: Update language detection for new languages
**Description:** As an AI agent, I need language detection updated so that Go, Java, C, and C++ files use tree-sitter chunking.

**Acceptance Criteria:**
- [ ] `detect_language` returns `ChunkerKind::TreeSitter(Language::Go)` for `.go`
- [ ] Returns `ChunkerKind::TreeSitter(Language::Java)` for `.java`
- [ ] Returns `ChunkerKind::TreeSitter(Language::C)` for `.c`, `.h`
- [ ] Returns `ChunkerKind::TreeSitter(Language::Cpp)` for `.cpp`, `.cc`, `.cxx`, `.hpp`
- [ ] `cargo test -p engram-ingest` passes

---

## Functional Requirements

### Core System
- FR-001: The system must be a single static Rust binary (`engram`) with no runtime dependencies except optional Ollama
- FR-002: The system must use a git repository as the sole persistent storage for all semantic data
- FR-003: The system must expose an MCP server over stdio and SSE transports
- FR-004: The system must support boot times of ~50ms (cached) and ~10s (cold, 200k chunks)
- FR-005: The system must handle repos with 500k+ chunks with sub-second search

### Ingest Pipeline
- FR-006: The system must use tree-sitter for AST-aware chunking of TypeScript, Rust, Python, Go, Java, C, and C++
- FR-007: The system must fall back to sliding window chunking for unsupported languages
- FR-008: The system must split Markdown files at heading boundaries
- FR-009: The system must support incremental indexing via `git diff` change detection
- FR-010: The system must skip re-embedding chunks whose content hash has not changed
- FR-011: The system must commit all index changes to the git store with structured commit messages

### Storage Format
- FR-012: The system must store chunk metadata as JSONL files (`.chunks.jsonl`) for human-readable diffs
- FR-013: The system must store embedding vectors as compact binary files (`.embeddings.bin`) with a 16-byte header (magic `EGRM`, version, dimensions, count, precision)
- FR-014: The system must support f32 and f16 embedding precision
- FR-015: The system must store knowledge items as YAML files for readability in git diffs

### Search
- FR-016: The system must perform hybrid search combining HNSW vector similarity (semantic) and BM25 (keyword) with configurable alpha weighting
- FR-017: The system must support direct lookup by chunk ID, file path, symbol name, and tag
- FR-018: The system must run parallel sidecar queries against the knowledge partition on every `engram_search` call
- FR-019: The system must drop knowledge results below a configurable minimum relevance threshold
- FR-020: The system must apply a configurable recency boost to recent knowledge items

### Knowledge Write-Back
- FR-021: The system must support recording architectural decisions, lessons learned, patterns, and glossary entries via MCP tools
- FR-022: All knowledge items must be embedded and indexed for searchability
- FR-023: The system must support saving and retrieving full conversation snapshots with tiered compression (active → compressed → archived)

### Multi-Repo
- FR-024: The system must support indexing multiple source repositories into a single semantic store
- FR-025: The system must build cross-repo symbol graphs with export/import resolution
- FR-026: The system must support tree-sitter (heuristic) and LSP (exact) symbol resolution backends

### Sync
- FR-027: The system must support local-only and remote-backed store modes
- FR-028: The system must support push/pull sync with conflict resolution (last-write-wins on chunk files)

### Context & Modes
- FR-029: The system must support fixed contexts (set at startup) that control the exposed MCP tool surface
- FR-030: The system must support dynamic modes (switchable during session) that adjust search behavior
- FR-031: The system must support custom context and mode definitions via YAML files

### Benchmark & Dashboard
- FR-032: The system must include a benchmark harness that measures token efficiency, retrieval precision/recall, time to first edit, file read count, and context waste
- FR-033: The system must commit benchmark metrics to the store for trend analysis
- FR-034: The system must serve a live web dashboard embedded in the binary for real-time observability

### Auto-Update
- FR-035: The system must support git hooks in source repos for automatic incremental reindexing
- FR-036: The system must support opt-in file watching with debounce for real-time updates
- FR-037: The system must support GitHub Agentic Workflows for remote CI-level reindexing

### Security
- FR-038: The ingest pipeline must strip content matching common secret patterns before embedding
- FR-039: Raw source code must NOT be stored in the semantic store — only chunk metadata and embeddings
- FR-040: Cloud embedding providers must read API keys from environment variables or key files, never from config
- FR-041: LSP servers must only run during indexing and shut down immediately after

## Non-Goals (Out of Scope)

- Engram is NOT an IDE — it does not provide code editing, debugging, or live code intelligence
- Engram does NOT store raw source code — agents read source from the source repo, Engram provides metadata and embeddings only
- Engram does NOT replace git — it uses git as its database but does not modify the source repos (only installs hooks)
- Engram does NOT provide real-time code completion or inline suggestions
- Engram does NOT manage or deploy AI agents — it provides context to agents, not orchestration
- Engram does NOT provide authentication or user management — access is controlled by git repo permissions
- Engram does NOT support non-git version control systems (SVN, Mercurial, etc.)
- Engram does NOT provide a general-purpose vector database API — it is purpose-built for code semantic search
- Engram does NOT run embedding models itself (except via the ONNX provider) — it delegates to Ollama or cloud APIs
- Engram does NOT provide a SaaS or hosted offering — it is a local/self-hosted tool
- Engram does NOT index build artifacts, node_modules, or other generated files
- Engram does NOT provide code review automation or PR commenting (it provides context; the agent decides what to do)

## Design Considerations

- **Nx + Cargo workspace:** Each Rust crate is an Nx project. Nx handles caching, affected detection, and task orchestration. Cargo handles compilation. Integration via `nx:run-commands` wrapping Cargo commands.
- **Crate boundaries:** Core types in `engram-core` (no heavy dependencies). Ingest pipeline in `engram-ingest`. Search/index in `engram-query`. MCP server in `engram-mcp`. Git operations in `engram-store`. Each crate has a clear single responsibility.
- **Binary format:** The EGRM binary embedding format is simple and forward-compatible. The version field and precision flag enable future evolution without breaking existing stores.
- **YAML for knowledge:** YAML was chosen over JSON for knowledge files because it produces cleaner git diffs, supports multiline strings naturally, and is more readable for humans reviewing PRs.
- **Partition-based search:** The HNSW index uses logical partitions (label-based filtering) rather than separate indexes to keep memory usage low and enable cross-partition queries if needed.

## Technical Considerations

- **Rust crate dependencies:** Key crates: `usearch` (HNSW), `tree-sitter` + language grammars, `git2` (libgit2), `reqwest` (HTTP), `axum` (web server), `serde`/`serde_json`/`serde_yaml`, `clap` (CLI), `notify` (file watcher), `zstd` (compression), `half` (f16), `ort` (ONNX Runtime), `thiserror`, `async-trait`, `tokio`
- **Cross-compilation:** The binary should compile on Linux (x86_64, aarch64), macOS (x86_64, aarch64), and Windows (x86_64). The ONNX provider may require platform-specific ONNX Runtime binaries.
- **Memory usage:** The in-memory HNSW index is the primary memory consumer. For 200k chunks with 768-dim f32 vectors: ~600MB. f16 reduces this to ~300MB. The compiled cache avoids rebuilding on restart.
- **Git repo size:** Binary embedding files compress well in git packfiles. A 50k-chunk repo with 768-dim f32 embeddings produces ~150MB of `.embeddings.bin` files. f16 halves this. git-lfs is available as an escape hatch for very large repos.
- **Ollama dependency:** Ollama is the default but not required. Users can switch to any provider. The ONNX provider provides a fully self-contained option with no external service.
- **LSP lifecycle:** Language servers are started on-demand and shut down after indexing. They are NOT kept running persistently, avoiding zombie processes and resource leaks.
- **MCP protocol:** The MCP protocol is JSON-RPC 2.0 over stdio or SSE. The `rmcp` crate provides Rust bindings. Fallback is a hand-rolled JSON-RPC implementation (~500 lines).

## Success Metrics

- Agents using Engram consume 50%+ fewer tokens on orientation/context-gathering compared to baseline
- Boot time with cached index is < 100ms for repos up to 200k chunks
- Hybrid search returns relevant results in < 50ms for repos up to 500k chunks
- Knowledge sidecar adds < 300 tokens per search query in compact mode
- Incremental reindex processes 10 changed files in < 5 seconds
- Full reindex of a 50k-chunk repo completes in < 5 minutes with Ollama
- Onboarding (standard depth) completes in < 60 seconds for a typical project
- The binary size is < 50MB (without ONNX runtime)
- Store repo size stays < 500MB for a 100k-chunk project with f32 precision

## Open Questions

1. **Quantized embeddings (u8):** The binary format reserves precision flag `2` for quantized u8 vectors. Should this be implemented? It would further reduce storage by 4x but may significantly impact recall quality.
2. **Git LFS integration:** At what store size should we recommend or auto-enable git-lfs for embedding files? 500MB? 1GB?
3. **Concurrent write safety:** If multiple agents write knowledge items simultaneously, how do we handle conflicts beyond last-write-wins? Should we use git branches for concurrent sessions and merge?
4. **Snapshot search relevance:** Should snapshot content be returned in regular `engram_search` results, or only when explicitly scoped with `scope: "snapshots"`? Including snapshots by default may add noise.
5. **Embedding model recommendations:** Should Engram ship with a recommended model comparison matrix (quality vs. speed vs. cost) for common use cases?
6. **Telemetry:** Should Engram support opt-in anonymous usage telemetry (search frequency, index sizes, provider usage) to inform development priorities?
7. **Plugin system:** Should Engram support custom chunking strategies via a plugin/extension mechanism, or is tree-sitter + sliding window sufficient?
8. **Token counting accuracy:** The current approach approximates tokens by splitting on whitespace. Should we integrate a proper tokenizer (tiktoken for OpenAI, sentencepiece for others) for accurate cost estimation?

