# Engram — Git-Backed Semantic Context MCP Server

> *"Memory that diffs."*

**Working name:** `engram` / `@engram-mcp`
**Status:** Architecture Draft v0.3
**Last updated:** 2026-03-09

---

## 1. Problem Statement

AI coding agents (Claude Code, Cursor, Copilot, Codex CLI) operate with limited contextual awareness of a codebase. They compensate by reading files one-by-one, running grep chains, and burning tokens on orientation work before they can do real work. Existing solutions (ContextStream, claude-context, code-memory) solve this via proprietary cloud backends or local vector databases that are:

- **Opaque** — no audit trail of what the agent "knows" or when knowledge changed
- **Ephemeral** — indexes are rebuilt from scratch, no history of how understanding evolved
- **Non-collaborative** — can't be shared across machines/team members without a SaaS dependency
- **Costly** — ContextStream charges per-seat; vector DB solutions require infrastructure

Engram solves this by using **a git repository as the semantic store**. Your agent's knowledge about your codebase is itself versioned, diffable, reviewable, and sharable — using infrastructure you already have for free.

---

## 2. Core Design Principles

1. **Git is the database.** All semantic data lives in a git repo. Queries run against an in-memory index rebuilt from committed data on boot (with a compiled cache for fast restarts). Git provides history, sync, collaboration, and auditability for free.

2. **Performance is non-negotiable.** Rust runtime, zero-copy deserialization, compiled index cache. Engram must handle 500k+ chunk repos with sub-second search and fast boot times. A single static binary with no runtime dependencies.

3. **Write-back loop.** Agents don't just read context — they contribute lessons, architectural decisions, and discovered patterns back to the store, committed with clear messages.

4. **Pluggable everything.** Embedding provider, symbol resolution backend, transport, chunk strategy — all swappable. Sensible local-first defaults so it works with zero config.

5. **Provably better.** A built-in benchmark harness quantifies token savings and retrieval quality vs. baseline agent behavior.

6. **Context-aware serving.** Engram adapts its tool surface and behavior to the client environment and task type via a context/mode system.

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Clients                          │
│  Claude Code │ Cursor │ Copilot │ Any MCP Client │ etc. │
└──────────┬──────────────────────────────────────────────┘
           │ MCP (stdio or SSE)
           ▼
┌─────────────────────────────────────────────────────────┐
│                  Engram MCP Server (Rust)                │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐  │
│  │  Query   │ │  Ingest  │ │ Benchmark │ │Dashboard │  │
│  │  Engine  │ │  Pipeline│ │ Harness   │ │ (Web UI) │  │
│  └────┬─────┘ └────┬─────┘ └───────────┘ └──────────┘  │
│       │             │                                    │
│  ┌────▼─────────────▼───────┐  ┌────────────────────┐   │
│  │   In-Memory Index        │  │ Context/Mode Engine │   │
│  │ (HNSW + BM25 inverted)  │  │ (tool surface ctrl) │   │
│  └──────────┬───┬───────────┘  └────────────────────┘   │
│     boot:   │   │  fast path:                            │
│  deserialize│   │  load compiled cache                   │
│  from git   │   │  (.engram-cache/, gitignored)          │
│  ┌──────────▼───▼───────────┐                            │
│  │     Git Semantic Store   │  ◄── local clone           │
│  │     (the git repo)       │  ◄── or remote origin      │
│  └──────────────────────────┘                            │
│                                                          │
│  ┌──────────────────────────┐  ┌────────────────────┐   │
│  │   Embedding Provider     │  │ Symbol Resolution   │   │
│  │ (Ollama | OpenAI | etc.) │  │ (tree-sitter | LSP) │   │
│  └──────────────────────────┘  └────────────────────┘   │
└──────────────────────────────────────────────────────────┘

          ┌─────────────────────────────┐
          │  Auto-Update Layer          │
          │                             │
          │  • git hooks (default)      │
          │  • file watcher (opt-in)    │
          │  • GH Agentic Workflows     │
          │  • manual `reindex` tool    │
          └─────────────────────────────┘
```

---

## 4. The Git Semantic Store — Repository Layout

The semantic store is a standard git repository with a well-defined directory structure. Metadata files are human-readable and produce meaningful diffs. Embeddings are stored as compact binary files to minimize repo size.

```
engram-store/
├── engram.config.yaml          # Store configuration
├── .engram/
│   ├── version                 # Store schema version (semver)
│   └── store-id                # UUID identifying this store instance
│
├── index/                      # Semantic chunks + embeddings
│   ├── manifest.json           # Global manifest: chunk count, last indexed commit,
│   │                           #   model name, dimensions, source repos list
│   │
│   ├── api/                    # Per-source-repo, mirrors structure
│   │   └── src/
│   │       ├── auth/
│   │       │   ├── login.ts.chunks.jsonl       # Chunk metadata (no embeddings)
│   │       │   ├── login.ts.embeddings.bin     # Binary embedding vectors
│   │       │   ├── middleware.ts.chunks.jsonl
│   │       │   └── middleware.ts.embeddings.bin
│   │       └── api/
│   │           ├── routes.ts.chunks.jsonl
│   │           └── routes.ts.embeddings.bin
│   │
│   ├── _xrefs/                 # Cross-repo reference index
│   │   ├── exports.jsonl
│   │   └── imports.jsonl
│   │
│   └── docs/
│       ├── README.md.chunks.jsonl
│       └── README.md.embeddings.bin
│
├── knowledge/                  # Agent-contributed knowledge (write-back)
│   ├── decisions/              # Architectural decisions (ADR-like)
│   │   ├── 2026-03-08_auth-strategy.yaml
│   │   └── 2026-03-07_db-migration-approach.yaml
│   ├── lessons/                # Learned lessons from failures/successes
│   │   ├── 2026-03-08_recursive-import-fix.yaml
│   │   └── 2026-03-07_test-isolation-pattern.yaml
│   ├── patterns/               # Discovered code patterns & conventions
│   │   └── error-handling.yaml
│   ├── glossary/               # Project-specific terminology
│   │   └── terms.yaml
│   └── onboarding/             # Auto-generated onboarding knowledge
│       ├── project-overview.yaml
│       ├── architecture-map.yaml
│       ├── build-test-commands.yaml
│       └── key-abstractions.yaml
│
├── snapshots/                  # Full conversation context snapshots
│   ├── active/                 # Recent (< 7 days) — plain YAML
│   │   └── 2026-03-08T14-30-00_session-abc.yaml
│   ├── compressed/             # Older (7-90 days) — zstd compressed
│   │   └── 2026-03-01T09-00_session-xyz.yaml.zst
│   └── archived/               # Old (90+ days) — zstd, git-lfs eligible
│       └── 2026-01-15T16-45_session-def.yaml.zst
│
└── metrics/                    # Benchmark & usage telemetry
    ├── baseline/
    └── runs/
        └── 2026-03-08T14-30-00_session-abc.jsonl
```

### 4.1 Split Storage Format

**Chunk metadata** (`.chunks.jsonl`) — human-readable, produces meaningful diffs:

```jsonl
{"chunk_id":"api/src/auth/login.ts#0","kind":"function","name":"authenticateUser","signature":"async function authenticateUser(req: Request): Promise<AuthResult>","start_line":14,"end_line":47,"content_hash":"a1b2c3d4","tags":["auth","middleware"],"indexed_at":"2026-03-08T10:00:00Z","source_commit":"abc1234","embedding_offset":0}
{"chunk_id":"api/src/auth/login.ts#1","kind":"class","name":"SessionManager","signature":"class SessionManager","start_line":49,"end_line":120,"content_hash":"e5f6g7h8","tags":["auth","session"],"indexed_at":"2026-03-08T10:00:00Z","source_commit":"abc1234","embedding_offset":1}
```

**Embedding vectors** (`.embeddings.bin`) — compact binary, one vector per chunk in order:

```
File format:
  Header (16 bytes):
    magic:      [u8; 4]    = b"EGRM"
    version:    u16         = 1
    dimensions: u16         = 768
    count:      u32         = number of vectors
    precision:  u16         = 0 (f32) | 1 (f16) | 2 (u8/quantized)
    reserved:   u16         = 0
  Body:
    vectors:    [f32; dimensions * count]  (or f16/u8 per precision flag)
```

The `embedding_offset` field in the JSONL maps each chunk to its position in the binary file.

**Why split storage:**
- Metadata diffs show exactly what changed: "function renamed", "new chunk added", "chunk deleted"
- Binary embedding files keep repo size ~50% smaller than JSON-encoded floats
- Git handles binary files fine (and they compress well in packfiles)
- Future option: git-lfs for embedding files if repos get very large
- `f16` precision flag enables further 50% size reduction with minimal recall loss

### 4.2 Knowledge Write-Back Format

Decisions and lessons use YAML for readability in diffs and PRs:

```yaml
# knowledge/decisions/2026-03-08_auth-strategy.yaml
id: decision-2026-03-08-001
title: "JWT with refresh token rotation for API auth"
status: accepted
context: >
  Evaluated session-based auth vs JWT. API is consumed by both
  web frontend and mobile clients. Need stateless verification.
decision: >
  Use short-lived JWTs (15min) with rotating refresh tokens stored
  in httpOnly cookies. Refresh token reuse triggers full revocation.
consequences:
  - Need token blacklist for logout-before-expiry edge case
  - Redis dependency for blacklist (TTL-based, low overhead)
related_files:
  - src/auth/jwt.ts
  - src/auth/refresh.ts
  - src/middleware/auth-guard.ts
contributed_by: claude-code-session-xyz
created_at: "2026-03-08T14:30:00Z"
embedding_ref: "knowledge/decisions/2026-03-08_auth-strategy.embedding.bin"
```

Knowledge items get their own small embedding binary files so they're searchable alongside code chunks.

---

## 5. MCP Tool Surface

Engram exposes the following MCP tools, grouped by intent. The active tool set varies based on the configured context and mode (see §9).

### 5.1 Query Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `engram_search` | Hybrid semantic + keyword search across all indexed content (with sidecar knowledge results, see §18) | `query`, `scope?` (code\|docs\|knowledge\|all), `top_k?`, `compact?` |
| `engram_lookup` | Direct lookup by file path, symbol name, or chunk ID | `identifier`, `include_content?` |
| `engram_related` | Find chunks semantically related to a given chunk or symbol | `chunk_id` or `symbol`, `top_k?` |
| `engram_graph` | Dependency/reference graph for a symbol (cross-repo aware) | `symbol`, `direction?` (callers\|callees\|both), `depth?` |
| `engram_status` | Index health: staleness, coverage, last sync time | — |

### 5.2 Write-Back Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `engram_record_decision` | Commit an architectural decision to the knowledge base | `title`, `context`, `decision`, `consequences?`, `related_files?` |
| `engram_record_lesson` | Commit a lesson learned (from failures, debugging, etc.) | `title`, `description`, `trigger`, `resolution?`, `related_files?` |
| `engram_record_pattern` | Document a discovered code pattern or convention | `name`, `description`, `examples?`, `anti_patterns?` |
| `engram_record_glossary` | Add/update project-specific terminology | `term`, `definition`, `context?` |
| `engram_snapshot` | Save a full conversation context snapshot for resumability | `session_id`, `summary`, `key_context`, `full_transcript` |

#### Snapshot Tiered Storage

Full conversation transcripts are preserved to maintain 100% functionality, with a tiered compression strategy to manage repo size:

- **Active** (< 7 days): plain YAML in `snapshots/active/`
- **Compressed** (7–90 days): zstd compressed in `snapshots/compressed/`
- **Archived** (90+ days): zstd compressed in `snapshots/archived/`, git-lfs eligible

Tier promotion happens during `engram reindex` or via `engram compact`. All tiers store full transcripts — the only difference is compression. The `engram_snapshot` MCP tool transparently decompresses on read. Embedding vectors for snapshot content are stored in the normal binary format alongside code embeddings so snapshots are always searchable regardless of tier.

### 5.3 Index Management Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `engram_reindex` | Trigger incremental or full reindex of source repo(s) | `scope?` (incremental\|full), `paths?`, `repo?` |
| `engram_sync` | Pull/push the semantic store to/from remote | `direction?` (pull\|push\|both) |
| `engram_onboard` | Run automated knowledge bootstrapping for a project (see §8) | `repo?`, `depth?` (quick\|standard\|deep) |

### 5.4 Benchmark Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `engram_benchmark_start` | Begin a benchmarked session (baseline or engram-assisted) | `mode` (baseline\|assisted), `task_description` |
| `engram_benchmark_log` | Log a retrieval event during benchmarked session | `query`, `tokens_used`, `files_read`, `hit?` |
| `engram_benchmark_end` | End session, compute and commit metrics | `task_outcome`, `notes?` |

### 5.5 Self-Regulation Tools

Tools that help agents reflect on their own context-gathering behavior. These tools don't perform search — they prompt structured reasoning about whether the agent has enough context to proceed. Inspired by Serena's thinking tools, adapted for Engram's retrieval-focused role.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `engram_assess_context` | Returns a structured prompt asking the agent to evaluate whether it has gathered enough context for the current task, or whether it should search more. Includes a summary of searches performed so far in this session, tokens consumed, and knowledge items surfaced. | `task_description?` |
| `engram_check_staleness` | Returns staleness information for all chunks the agent has retrieved so far in the session. Prompts the agent to decide whether stale context is acceptable or requires a reindex. | — |

**Rationale:** Agents tend to either over-fetch (burning tokens reading files they don't need) or under-fetch (making changes without understanding the full picture). These tools give agents a structured checkpoint to self-correct. They're especially valuable in agentic pipelines where multiple steps build on each other — a quick `engram_assess_context` between planning and execution can prevent wasted work.

---

## 6. In-Memory Index Architecture

### 6.1 Boot Sequence

```
1. Check .engram-cache/ for compiled index
   ├── Cache exists AND cache fingerprint matches manifest.json hash?
   │   └── YES: mmap the compiled cache → ready in ~50ms (fast path)
   └── NO: Walk index/ directory
           ├── Deserialize all .chunks.jsonl files
           ├── mmap all .embeddings.bin files (zero-copy)
           ├── Construct HNSW index from embeddings
           ├── Construct BM25 inverted index from chunk metadata
           ├── Write compiled cache to .engram-cache/
           └── Ready (cold path: ~2s for 50k chunks, ~10s for 200k)
```

### 6.2 Compiled Index Cache

The `.engram-cache/` directory is gitignored and contains pre-built index structures:

```
.engram-cache/
├── fingerprint           # SHA-256 of manifest.json — invalidation key
├── hnsw.index            # Serialized HNSW graph (usearch format)
├── bm25.index            # Serialized BM25 inverted index
└── metadata.bin          # Serialized chunk metadata maps
```

Cache is rebuilt automatically when the fingerprint doesn't match (i.e., after `git pull` brings new index data). This gives near-instant boot on repeat runs while keeping git as the source of truth.

### 6.3 HNSW Vector Index
- Crate: `usearch` (Rust bindings) — fast, supports f16/f32, memory-mappable
- Rebuilt from `.embeddings.bin` files on cold boot
- Typical cold rebuild: ~2s for 50k chunks, ~8s for 200k chunks
- Cached rebuild: ~50ms (mmap the serialized graph)

### 6.4 BM25 Inverted Index
- Built from chunk `name`, `signature`, and `tags` fields
- Used in hybrid retrieval: `score = α * vector_score + (1 - α) * bm25_score`
- α configurable, default 0.7 (favor semantic over keyword)

### 6.5 Metadata Index
- In-memory maps for fast lookup:
  - `chunk_id → chunk` (direct access)
  - `file_path → chunk[]` (file-level retrieval)
  - `symbol_name → chunk[]` (symbol lookup)
  - `tag → chunk[]` (tag-based filtering)
  - `source_repo → chunk[]` (multi-repo filtering)
- Knowledge items (decisions, lessons, patterns, onboarding) also indexed in both HNSW and BM25

### 6.6 Cross-Repo Symbol Graph

For multi-repo stores, Engram builds a unified symbol graph that resolves references across repository boundaries. This uses a pluggable symbol resolution backend.

#### Symbol Resolution Backends

**Tree-sitter (default):** Available everywhere, no external dependencies. Extracts definitions and imports via AST parsing. Cross-repo resolution uses heuristic matching: import paths are matched against known exports using package.json/Cargo.toml dependency mappings and path aliases (tsconfig paths, Cargo workspace members). For ambiguous paths, falls back to symbol name + type signature similarity.

**LSP (optional, advanced):** Connects to running language servers (rust-analyzer, typescript-language-server, pylsp, gopls, etc.) for exact symbol resolution. The LSP backend provides precise "go to definition," reference finding, and type hierarchy information — no heuristics needed. Cross-repo edges become exact rather than probabilistic.

```rust
#[async_trait]
pub trait SymbolResolver: Send + Sync {
    /// Resolve all exports from a file
    async fn extract_exports(&self, file: &Path) -> Result<Vec<ExportedSymbol>>;

    /// Resolve all imports in a file to their source locations
    async fn resolve_imports(&self, file: &Path) -> Result<Vec<ResolvedImport>>;

    /// Find all references to a symbol across the indexed repos
    async fn find_references(&self, symbol: &SymbolId) -> Result<Vec<SymbolReference>>;

    /// Get type hierarchy (supertypes/subtypes) for a symbol
    async fn type_hierarchy(&self, symbol: &SymbolId) -> Result<TypeHierarchy>;
}
```

**Configuration:**

```yaml
# engram.config.yaml
symbol_resolution:
  backend: "tree-sitter"           # tree-sitter | lsp

  # LSP-specific config (only when backend: lsp)
  lsp:
    servers:
      typescript: "typescript-language-server"
      rust: "rust-analyzer"
      python: "pylsp"
      go: "gopls"
    auto_install: true             # Auto-install language servers if missing
    startup_timeout_ms: 10000
    # LSP servers are started on-demand during indexing and shut down after
```

**LSP lifecycle management:** Language servers are started on-demand during indexing and cross-repo graph construction, then shut down. They are NOT kept running between reindex operations — Engram is a retrieval system, not an IDE. The LSP is used as a batch analysis tool during ingest, not a live service. This avoids the zombie process and asyncio deadlock issues that Serena's team documented in their lessons learned.

**Cross-repo reference index:**

```jsonl
// index/_xrefs/exports.jsonl
{"symbol":"AuthResult","kind":"type","repo":"shared","file":"packages/types/auth.ts","chunk_id":"shared/packages/types/auth.ts#2"}

// index/_xrefs/imports.jsonl
{"symbol":"AuthResult","importing_repo":"api","importing_file":"src/auth/login.ts","source_repo":"shared","source_file":"packages/types/auth.ts","resolved_chunk":"shared/packages/types/auth.ts#2","resolution":"exact"}
```

The `resolution` field indicates `"exact"` (from LSP) or `"heuristic"` (from tree-sitter), so consumers know the confidence level.

**Limitations:**
- Tree-sitter: Dynamic imports and re-exports can't be statically resolved
- LSP: Adds ~30% to ingest time; requires language servers to be installed; some languages have slow or incomplete LSP implementations
- First-time cross-repo graph build is slower than incremental updates

### 6.7 Staleness Detection
- Each chunk stores `source_commit` — the commit hash of the source repo when indexed
- On boot, compare `HEAD` of each source repo vs. `source_commit` on chunks
- Mark stale chunks but still serve them (with staleness flag) until reindex completes
- Enables instant boot with eventual consistency

---

## 7. Ingest Pipeline

```
Source Repo(s)                 Engram Pipeline
──────────────                 ──────────────
                               ┌──────────────────┐
  file change  ──────────────► │  Change Detection │
  (git diff)                   │  (incremental)    │
                               └────────┬─────────┘
                                        │ changed files only
                               ┌────────▼─────────┐
                               │   AST Chunker     │
                               │   (tree-sitter)   │
                               └────────┬─────────┘
                                        │ semantic chunks
                               ┌────────▼─────────┐
                               │  Content Hasher   │
                               │  (skip unchanged) │
                               └────────┬─────────┘
                                        │ new/modified chunks
                               ┌────────▼──────────┐
                               │  Embedding Provider│
                               │  (pluggable)       │
                               └────────┬──────────┘
                                        │ chunks + embeddings
                               ┌────────▼──────────┐
                               │  Symbol Resolver   │
                               │  (tree-sitter|LSP) │
                               └────────┬──────────┘
                                        │ cross-repo refs
                               ┌────────▼──────────┐
                               │  Split Serializer  │
                               │  .chunks.jsonl +   │
                               │  .embeddings.bin   │
                               └────────┬──────────┘
                                        │
                               ┌────────▼──────────┐
                               │  Cache Invalidate  │
                               │  (delete .engram-  │
                               │   cache/fingerprint)│
                               └────────┬──────────┘
                                        │
                               ┌────────▼──────────┐
                               │  Git Commit        │
                               │  "engram: reindex  │
                               │   3 files, 12      │
                               │   chunks updated"  │
                               └────────────────────┘
```

### 7.1 AST Chunking Strategy
- **Primary:** tree-sitter (native Rust crate) for language-aware structural chunking — functions, classes, methods, type definitions, impl blocks
- **Fallback:** Sliding window with overlap for unsupported languages or non-code files
- **Documentation:** Markdown header-based splitting (h1/h2/h3 boundaries)
- Chunk metadata includes: kind, name, signature, line range, file path, source repo

### 7.2 Incremental Indexing
- Run `git diff <last_indexed_commit>..HEAD` on each source repo
- Only re-chunk and re-embed files that changed
- Content hash per chunk — if a function's body didn't change, skip re-embedding even if the file did
- Deleted files → remove corresponding `.chunks.jsonl` + `.embeddings.bin`, commit removal

### 7.3 Multi-Repo Indexing

Engram supports indexing multiple source repositories into a single semantic store.

```yaml
# engram.config.yaml
sources:
  - name: "api"
    repo: "../api-service"
    include: ["src/**", "docs/**"]
    exclude: ["**/*.test.ts", "dist/**"]
  - name: "web"
    repo: "../web-frontend"
    include: ["src/**", "docs/**"]
  - name: "shared"
    repo: "../shared-libs"
    include: ["packages/**"]
```

Cross-repo search is the default. The `repo?` parameter on `engram_search` allows scoping to a specific source repo when needed.

---

## 8. Onboarding — Automated Knowledge Bootstrapping

When Engram connects to a new project (or when explicitly triggered), it can run a structured onboarding pass that systematically analyzes the codebase and generates initial knowledge entries. Unlike ad-hoc write-back from agents, onboarding is a deliberate, repeatable process that produces a baseline understanding of the project.

### 8.1 What Onboarding Produces

Onboarding generates YAML files in `knowledge/onboarding/` covering:

**`project-overview.yaml`** — High-level project description:
```yaml
name: "api-service"
language: "TypeScript"
framework: "NestJS"
package_manager: "pnpm"
repo_type: "monorepo-member"
description: >
  REST API service handling authentication, user management,
  and billing integration. Part of a multi-repo architecture
  with shared types from @shared/types.
entry_points:
  - src/main.ts
  - src/app.module.ts
```

**`build-test-commands.yaml`** — How to build, test, lint, and run:
```yaml
build:
  command: "pnpm build"
  output_dir: "dist/"
test:
  command: "pnpm test"
  framework: "jest"
  config: "jest.config.ts"
lint:
  command: "pnpm lint"
  tool: "eslint"
run:
  command: "pnpm start:dev"
  port: 3000
```

**`architecture-map.yaml`** — Key directories and their purposes:
```yaml
directories:
  - path: "src/auth/"
    purpose: "Authentication and authorization (JWT, guards, strategies)"
    key_files: ["jwt.strategy.ts", "auth.guard.ts", "auth.service.ts"]
  - path: "src/billing/"
    purpose: "Stripe integration and subscription management"
    key_files: ["stripe.service.ts", "webhook.controller.ts"]
  - path: "src/common/"
    purpose: "Shared utilities, decorators, pipes, interceptors"
```

**`key-abstractions.yaml`** — Important types, interfaces, and patterns:
```yaml
abstractions:
  - name: "AuthResult"
    kind: "type"
    file: "src/auth/types.ts"
    description: "Union type returned by all auth operations"
  - name: "BaseController"
    kind: "class"
    file: "src/common/base.controller.ts"
    description: "Abstract controller with standard error handling and logging"
patterns:
  - name: "Guard pattern"
    description: "NestJS guards used for route-level auth checks"
    examples: ["src/auth/auth.guard.ts", "src/auth/roles.guard.ts"]
```

### 8.2 How Onboarding Works

```
engram onboard
     │
     ▼
┌─────────────────────┐
│  1. Detect Project   │  Read package.json, Cargo.toml, pyproject.toml, etc.
│     Metadata         │  Identify language, framework, package manager
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  2. Scan Directory   │  Walk file tree, identify key directories
│     Structure        │  by naming conventions and file count/type
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  3. Analyze Key      │  Read entry points, config files, READMEs
│     Files            │  Extract build/test/lint commands
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  4. Extract Symbol   │  Use tree-sitter (or LSP) to find exported
│     Overview         │  types, key classes, public APIs
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  5. Generate         │  Write YAML files to knowledge/onboarding/
│     Knowledge Files  │  Embed and index them
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  6. Git Commit       │  "engram: onboard api-service (4 knowledge
│                      │   files, 23 key abstractions identified)"
└─────────────────────┘
```

### 8.3 Onboarding Depth Levels

| Depth | What It Does | Speed |
|-------|-------------|-------|
| `quick` | Project metadata + build commands + directory overview. No symbol analysis. | ~5s |
| `standard` (default) | Quick + key abstractions + architecture map. Tree-sitter analysis of entry points and high-traffic files. | ~30s |
| `deep` | Standard + full symbol export analysis + cross-file dependency mapping + pattern detection. Uses LSP if available. | ~2-5min |

### 8.4 Re-Onboarding

Onboarding is idempotent. Running `engram onboard` again overwrites existing onboarding files with fresh analysis. The git diff shows exactly what changed in the project's structure since last onboarding, which is useful for catching drift (new directories, removed abstractions, changed build commands).

Onboarding can also be triggered via the `engram_onboard` MCP tool, allowing agents to self-onboard when they detect they're working in an unfamiliar area of the codebase.

---

## 9. Context & Mode System

Engram adapts its behavior based on the client environment (context) and the current task type (mode). This controls which tools are exposed, how verbose search results are, and how aggressively knowledge is surfaced.

### 9.1 Contexts (Set at Startup, Fixed for Session)

A context defines the client environment. It's set once when the MCP server starts and cannot change during a session.

| Context | Description | Tool Adjustments |
|---------|-------------|------------------|
| `default` | Full tool surface, suitable for any MCP client | All tools enabled |
| `claude-code` | Optimized for Claude Code sessions. Claude Code has its own file read/write, grep, and shell execution. | Disables tools that duplicate Claude Code builtins (e.g., `engram_lookup` for simple file reads). Keeps search, knowledge, and benchmark tools. |
| `cursor` | Optimized for Cursor IDE. Similar to `claude-code` but adjusted for Cursor's MCP integration patterns. | Full search + knowledge tools. Compact mode forced on (Cursor's context window is constrained). |
| `ci` | For headless/CI environments (GitHub Actions, danbot pipelines). No interactive tools. | Disables dashboard, snapshot, and self-regulation tools. Enables benchmark auto-logging. |
| `ide-assistant` | Generic context for IDE integrations (VSCode, Windsurf, etc.) | Reduced tool set focused on search and lookup. Write-back tools available but knowledge sidecar disabled by default to reduce noise. |

**Configuration:**

```bash
# Set context at startup
engram serve --context claude-code
engram serve --context ci
```

Or in the MCP client config:

```json
{
  "command": "engram",
  "args": ["serve", "--transport", "stdio", "--context", "claude-code"]
}
```

### 9.2 Modes (Dynamic, Switchable During Session)

Modes refine behavior for specific task types. Multiple modes can be active simultaneously. Modes can be switched during a session via the `engram_switch_mode` MCP tool.

| Mode | Description | Behavior Changes |
|------|-------------|-----------------|
| `explore` (default) | General exploration and understanding | Full sidecar knowledge, standard top_k |
| `edit` | Focused code modification | Boosts relevance of lessons and patterns in sidecar. Increases `min_relevance` threshold to reduce noise during focused work. |
| `plan` | Architecture and design planning | Boosts decisions and patterns. Increases knowledge top_k. Enables `engram_assess_context` prompts automatically. |
| `onboard` | First encounter with a codebase | Runs onboarding if not already done. Surfaces onboarding knowledge prominently. |
| `benchmark` | Performance measurement mode | Enables auto-logging of all search/read events. Adds token counting to responses. |

```yaml
# Default modes in config
modes:
  default: ["explore"]
```

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `engram_switch_mode` | Activate one or more modes for the current session | `modes` (list of mode names) |
| `engram_get_config` | Return current context, active modes, and tool surface | — |

### 9.3 Custom Contexts and Modes

Users can define custom contexts and modes via YAML files:

```yaml
# .engram/contexts/my-pipeline.yaml
name: "my-pipeline"
description: "Custom context for nightly analysis pipeline"
tools:
  exclude: ["engram_snapshot", "engram_switch_mode", "engram_assess_context"]
search:
  compact_by_default: true
  knowledge_sidecar: false
```

```yaml
# .engram/modes/review.yaml
name: "review"
description: "Code review mode — surfaces decisions and patterns heavily"
search:
  knowledge:
    top_k: 10
    min_relevance: 0.5
    boost_decisions: 1.5
    boost_patterns: 1.3
```

---

## 10. Pluggable Embedding Architecture

Engram treats embedding as a pure I/O boundary — text goes in, vectors come out. The provider trait is intentionally minimal so adding a new provider is trivial.

### 10.1 Provider Trait

```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Human-readable provider name (e.g., "openai/text-embedding-3-small")
    fn name(&self) -> &str;

    /// Dimensionality of output vectors
    fn dimensions(&self) -> usize;

    /// Maximum texts per batch call
    fn max_batch_size(&self) -> usize;

    /// Embed a batch of texts. Returns one vector per input text.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("Provider API error: {0}")]
    Api(String),
    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },
    #[error("Text too long: {length} tokens exceeds max {max}")]
    TextTooLong { length: usize, max: usize },
    #[error("Provider not available: {0}")]
    Unavailable(String),
}
```

### 10.2 Built-In Providers

#### Ollama (Default — Free, Local)

```yaml
embedding:
  provider: "ollama"
  model: "nomic-embed-text"    # or mxbai-embed-large, all-minilm, etc.
  ollama:
    base_url: "http://localhost:11434"
    keep_alive: "5m"           # Keep model loaded between batches
```

- Zero cost, zero data egress
- Requires Ollama installed and running
- `nomic-embed-text` (768 dims) is the default: strong code understanding, fast inference
- Engram auto-detects if Ollama is available on boot and warns if not

#### OpenAI

```yaml
embedding:
  provider: "openai"
  model: "text-embedding-3-small"   # or text-embedding-3-large
  openai:
    api_key_env: "OPENAI_API_KEY"   # Read from environment variable
    api_key_file: "~/.engram/openai.key"  # Or read from file (fallback)
    base_url: null                  # Override for Azure OpenAI or proxies
    dimensions: 1536                # Optional: reduce dims for 3-large
    request_timeout_ms: 30000
    max_retries: 3
    rate_limit:
      requests_per_minute: 3000
      tokens_per_minute: 1000000
```

#### Anthropic (Voyage Integration)

```yaml
embedding:
  provider: "voyage"
  model: "voyage-code-3"           # Code-optimized model
  voyage:
    api_key_env: "VOYAGE_API_KEY"
    base_url: "https://api.voyageai.com/v1"
    request_timeout_ms: 30000
    max_retries: 3
```

#### In-Process ONNX (Local, No External Dependencies)

```yaml
embedding:
  provider: "onnx"
  model: "BAAI/bge-small-en-v1.5"  # Downloaded and cached locally
  onnx:
    model_cache_dir: "~/.engram/models"
    device: "cpu"                   # or "cuda" if available
    quantized: true                 # Use INT8 quantized model
```

#### Custom Providers

```yaml
embedding:
  provider: "custom"
  custom:
    # HTTP adapter: Engram POSTs {"texts": [...]} and expects {"embeddings": [[...]]}
    endpoint: "http://localhost:8080/embed"
    dimensions: 512
    max_batch_size: 64
    headers:
      Authorization: "Bearer ${CUSTOM_EMBED_KEY}"
```

### 10.3 Provider Switching & Reindex

**Critical invariant:** Changing the embedding provider or model invalidates the entire index.

On boot, Engram compares the configured provider/model against `manifest.json`. If they don't match, Engram warns and blocks search until `engram reindex --full` is run.

### 10.4 Cost Estimation

Before a full reindex with a paid provider, `engram reindex --estimate` calculates estimated token count, cost per provider, and time.

---

## 11. Auto-Update Layer

Four tiers of update automation, from lightest to richest:

### 11.1 Git Hooks (Default, Local, Instant)

Installed into each **source repo** (not the semantic store):

**`post-commit` hook:**
```bash
#!/bin/bash
engram reindex --incremental --commit HEAD
```

**`post-merge` / `post-checkout` hook:**
```bash
#!/bin/bash
engram reindex --incremental --commit HEAD
```

Hooks are installed via `engram init --hooks` and are lightweight — they shell out to the Engram CLI.

### 11.2 File Watcher (Opt-In, Local, Real-Time)

For developers who want live updates during active development without waiting for commits:

```yaml
watcher:
  enabled: true
  debounce_ms: 2000
  ignore: ["*.lock", "*.log"]
```

- Uses `notify` crate (Rust cross-platform file watcher)
- Debounced: waits for edits to settle, then batches changed files
- Does NOT auto-commit — stages changes for the next explicit commit or hook-triggered commit

### 11.3 GitHub Agentic Workflows (Remote, Rich)

For repos with a remote semantic store, a GitHub Agentic Workflow handles CI-level reindexing.

**Workflow definition** (`engram-reindex.md`):

```markdown
# Engram Reindex

## Trigger
On push to `main`, or on schedule (daily at 2am UTC).

## Task
1. Clone the source repo(s) and the engram semantic store repo.
2. Run `engram reindex --full` against all configured source repos.
3. Run `engram analyze` to update knowledge graph relationships.
4. If any decisions or lessons reference files that no longer exist,
   flag them as potentially stale.
5. Commit changes to the semantic store and push.
6. If significant changes detected (>50 chunks updated), open a
   summary PR on the semantic store for visibility.
```

### 11.4 Manual Reindex Tool

The `engram_reindex` MCP tool (§5.3) serves as the escape hatch. Supports path-scoped and repo-scoped reindex.

---

## 12. Benchmark Harness

### 12.1 What We Measure

| Metric | Description |
|--------|-------------|
| **Token Efficiency** | Total tokens consumed (input + output) to complete a task |
| **Retrieval Precision** | % of retrieved chunks that were actually relevant to the task |
| **Retrieval Recall** | % of relevant chunks that were successfully retrieved |
| **Time to First Edit** | Wall time from task start to first meaningful code change |
| **File Read Count** | Number of raw file reads the agent performed |
| **Search-to-Read Ratio** | Engram searches vs. raw file reads (higher = more efficient) |
| **Context Waste** | Tokens spent on files/content that didn't contribute to the outcome |

### 12.2 How It Works

```
┌──────────────────────────────────────────────────────┐
│                  Benchmark Session                    │
│                                                      │
│  1. Agent starts task                                │
│  2. engram_benchmark_start(mode: "assisted")         │
│  3. Every engram_search call auto-logs:              │
│     - query text                                     │
│     - tokens in returned context                     │
│     - chunks returned                                │
│  4. Every raw file read (detected via MCP) logs:     │
│     - file path                                      │
│     - tokens consumed                                │
│  5. engram_benchmark_end(task_outcome: "success")    │
│  6. Metrics computed, committed to metrics/runs/     │
│                                                      │
│  Baseline mode: same task, engram_search disabled,   │
│  agent uses only native file/grep tools.             │
│  Comparison report generated automatically.          │
└──────────────────────────────────────────────────────┘
```

### 12.3 Comparison Reports

Committed to `metrics/` as YAML, tracked in git history for trend analysis.

---

## 13. Live Dashboard

Engram includes an embedded web dashboard for real-time observability. The dashboard is served by the MCP server process on a configurable port.

### 13.1 What the Dashboard Shows

**Live Session View:**
- Active MCP connections (which clients are connected)
- Real-time stream of search queries and results
- Current active context and modes
- Token consumption per session

**Index Health:**
- Total chunks indexed per source repo
- Staleness map: which files have stale chunks, how stale
- Last reindex time and duration
- Embedding provider status (Ollama running? API key valid?)

**Knowledge Activity:**
- Recent write-backs (decisions, lessons, patterns)
- Knowledge item count and category breakdown
- Onboarding status per source repo

**Search Analytics:**
- Query frequency heatmap (which areas of the codebase are searched most)
- Average result relevance scores
- Cache hit rate (compiled index cache)
- SmartRouter sidecar hit rate (how often knowledge results are above min_relevance)

**Benchmark Dashboard:**
- Active benchmark sessions
- Historical comparison trends (token savings over time)
- Per-task metrics drill-down

### 13.2 Implementation

```yaml
# engram.config.yaml
dashboard:
  enabled: true
  port: 3200
  open_on_start: false            # Auto-open browser on engram serve
```

- Served by the Engram MCP server process (no separate process)
- Built with a lightweight embedded HTTP server (`axum` or `warp`)
- Frontend: static HTML + JS bundled into the binary (no build step for users)
- WebSocket connection for real-time updates
- Disabled in `ci` context by default
- Accessible at `http://localhost:3200/dashboard`

### 13.3 Dashboard vs. Benchmark Harness

The dashboard shows live, ephemeral operational data (current session, real-time metrics). The benchmark harness produces persistent, committed data (comparison reports in `metrics/`). They're complementary — the dashboard helps you understand what's happening now, the benchmark harness proves that Engram is working over time.

---

## 14. Sync Model — Local and Remote Stores

### 14.1 Local-Only Mode
- Semantic store is a local git repo (not pushed anywhere)
- `engram init --local` creates the store adjacent to the source repo
- No remote needed; git history is still useful for local auditability

### 14.2 Remote Mode
- Semantic store is a GitHub/GitLab repo
- `engram init --remote git@github.com:team/project-engram.git`
- Cloned locally on first run
- `engram_sync` tool handles pull/push
- Conflicts resolved via last-write-wins on individual chunk files
- GH Agentic Workflows keep the remote up to date

### 14.3 Multi-Machine / Team Sync

```
Developer A (laptop)          GitHub             Developer B (laptop)
┌──────────────┐         ┌─────────────┐        ┌──────────────┐
│ Engram MCP   │──push──►│ engram-store │◄──pull─│ Engram MCP   │
│ (local)      │◄──pull──│ (remote)     │──push─►│ (local)      │
└──────────────┘         └──────┬──────┘        └──────────────┘
                                │
                         ┌──────▼──────┐
                         │ GH Agentic  │
                         │ Workflow     │
                         │ (reindex on  │
                         │  push to     │
                         │  source repo)│
                         └─────────────┘
```

---

## 15. Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | Rust | Performance-critical hot paths, single static binary, native tree-sitter |
| **MCP Protocol** | `rmcp` crate or direct JSON-RPC impl | Rust MCP ecosystem maturing; protocol is simple JSON-RPC |
| **AST Parsing** | `tree-sitter` (native crate) | Best-in-class multi-language structural parsing |
| **LSP Client** | `tower-lsp` or `lsp-types` + custom | For optional LSP symbol resolution backend |
| **Vector Index** | `usearch` (Rust bindings) | Fast HNSW, f16 support, memory-mappable |
| **BM25** | Custom (inverted index with `fst` crate) | Lightweight, in-process |
| **Embeddings (default)** | `nomic-embed-text` via Ollama | Free, local, good code performance |
| **Embeddings (in-process)** | `ort` (ONNX Runtime bindings) | Fully self-contained/air-gapped |
| **Git Operations** | `git2` (libgit2 bindings) | Native Rust git, no shell-out |
| **File Watching** | `notify` crate | Cross-platform |
| **HTTP Client** | `reqwest` | For cloud embedding APIs |
| **Web Server** | `axum` | Dashboard + SSE transport |
| **Config** | `serde` + YAML | Human-readable, good diffs |
| **CLI** | `clap` | Standard Rust CLI |
| **Compression** | `zstd` crate | Snapshot tiered compression |
| **Serialization** | `serde_json` (JSONL) + custom binary | Split format |

### 15.1 Nx Monorepo Structure

Engram uses an Nx monorepo to manage the workspace. Each Rust crate is an Nx project with build, test, and lint targets orchestrated by Nx via `nx:run-commands` wrapping Cargo commands. Nx provides caching, affected detection, and task orchestration; Cargo handles actual compilation.

```
engram/
├── nx.json                     # Nx workspace configuration
├── Cargo.toml                  # Cargo workspace root
├── crates/
│   ├── engram-core/            # Chunk types, provider trait, store schema, config,
│   │   ├── Cargo.toml          #   context/mode definitions
│   │   ├── project.json        # Nx project config
│   │   └── src/
│   ├── engram-ingest/          # AST chunking, change detection, embedding pipeline,
│   │                           #   onboarding logic
│   ├── engram-query/           # HNSW + BM25 hybrid search, graph traversal,
│   │                           #   cross-repo symbol resolution, SmartRouter sidecar
│   ├── engram-mcp/             # MCP tool definitions, server bootstrap (stdio + SSE),
│   │                           #   context/mode engine
│   ├── engram-store/           # Git read/write, sync, commit formatting, binary format
│   ├── engram-cache/           # Compiled index cache (build, load, invalidate)
│   ├── engram-bench/           # Benchmark harness, metrics computation, reports
│   ├── engram-watch/           # File watcher integration
│   ├── engram-dashboard/       # Web dashboard (axum server, static frontend)
│   ├── engram-lsp/             # Optional LSP client for symbol resolution
│   ├── engram-cli/             # `engram` CLI binary (clap)
│   └── engram-providers/
│       ├── ollama/
│       ├── openai/
│       ├── voyage/
│       ├── onnx/
│       └── custom/
├── gh-aw/                      # GitHub Agentic Workflow definitions
│   └── engram-reindex.md
└── tools/                      # Nx custom executors if needed
```

---

## 16. Configuration

```yaml
# engram.config.yaml
version: "1.0"

sources:
  - name: "api"
    repo: "../api-service"
    include: ["src/**", "docs/**"]
    exclude: ["**/*.test.ts", "dist/**"]
  - name: "web"
    repo: "../web-frontend"
    include: ["src/**", "docs/**"]
  - name: "shared"
    repo: "../shared-libs"
    include: ["packages/**"]

store:
  path: "./engram-store"
  remote: "git@github.com:team/my-project-engram.git"
  auto_sync: true

embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  dimensions: 768
  ollama:
    base_url: "http://localhost:11434"
    keep_alive: "5m"
  openai:
    api_key_env: "OPENAI_API_KEY"
    model: "text-embedding-3-small"
    dimensions: 1536
  voyage:
    api_key_env: "VOYAGE_API_KEY"
    model: "voyage-code-3"
  onnx:
    model: "BAAI/bge-small-en-v1.5"
    model_cache_dir: "~/.engram/models"
    device: "cpu"
    quantized: true
  custom:
    endpoint: "http://localhost:8080/embed"
    dimensions: 512
    max_batch_size: 64

symbol_resolution:
  backend: "tree-sitter"
  lsp:
    servers:
      typescript: "typescript-language-server"
      rust: "rust-analyzer"
      python: "pylsp"
      go: "gopls"
    auto_install: true
    startup_timeout_ms: 10000

chunking:
  strategy: "tree-sitter"
  max_chunk_tokens: 512
  overlap_tokens: 64

search:
  hybrid_alpha: 0.7
  default_top_k: 10
  compact_by_default: true
  knowledge:
    enabled: true
    top_k: 5
    min_relevance: 0.6
    boost_recent: true
    recent_boost_factor: 1.1

context: "default"              # default | claude-code | cursor | ci | ide-assistant

modes:
  default: ["explore"]

hooks:
  install: true
  post_commit: true
  post_merge: true

watcher:
  enabled: false
  debounce_ms: 2000
  ignore: ["*.lock", "*.log"]

dashboard:
  enabled: true
  port: 3200
  open_on_start: false

benchmark:
  enabled: true
  auto_log: true

storage:
  embedding_precision: "f32"
```

---

## 17. CLI Commands

```bash
# Initialize a new Engram store
engram init [--local | --remote <url>]

# Automated knowledge bootstrapping
engram onboard [--depth quick|standard|deep] [--repo <name>]

# Install git hooks in source repo(s)
engram hooks install

# Index/reindex source repo(s)
engram reindex [--full | --incremental] [--paths <glob>] [--repo <name>]

# Estimate cost of a full reindex (for paid providers)
engram reindex --estimate

# Compact snapshots (promote to compressed/archived tiers)
engram compact

# Sync with remote store
engram sync [--pull | --push | --both]

# Start the MCP server
engram serve [--transport stdio | --transport sse --port 3100] [--context <ctx>]

# Check store health
engram status

# Run a benchmark comparison
engram benchmark run --task "description" --baseline --assisted

# View benchmark results
engram benchmark report [--last | --all | --compare <id1> <id2>]

# Start file watcher (if not using watcher.enabled in config)
engram watch
```

---

## 18. Security Considerations

- **No secrets in the store.** The ingest pipeline strips content matching common secret patterns before embedding. Configurable via `exclude_patterns`.
- **Embedding-only storage.** Raw source code is NOT stored in the semantic store — only chunk metadata and embeddings. The agent reads actual source from the source repo.
- **Git-level access control.** The semantic store repo inherits your git access model.
- **Local embeddings by default.** No data leaves your machine unless you opt into a cloud embedding provider.
- **API keys never in config.** Cloud providers read keys from environment variables or key files.
- **LSP servers are sandboxed to indexing.** When the LSP backend is used, language servers are started only during reindex operations and shut down immediately after. They never run persistently.

---

## 19. Rollout Phases

### Phase 1 — Core Loop (MVP)
- Ingest pipeline (tree-sitter chunking, Ollama embeddings)
- Split serialization (JSONL metadata + binary embeddings) to git store
- In-memory HNSW + BM25 index with compiled cache
- `engram_search`, `engram_lookup`, `engram_status` MCP tools
- `engram init`, `engram reindex`, `engram serve` CLI
- Local-only mode, single source repo
- Default context, explore mode

### Phase 2 — Write-Back, Knowledge & Onboarding
- `engram_record_decision`, `engram_record_lesson`, `engram_record_pattern` tools
- `engram_snapshot` with tiered compression
- `engram onboard` — automated knowledge bootstrapping (quick + standard depth)
- Knowledge items indexed alongside code chunks
- SmartRouter sidecar search (§20)
- Self-regulation tools (`engram_assess_context`, `engram_check_staleness`)

### Phase 3 — Multi-Repo, Sync & Contexts
- Multi-source-repo indexing
- Cross-repo symbol resolution (tree-sitter backend)
- Remote store support (push/pull)
- `engram_sync` MCP tool
- GH Agentic Workflow for remote reindexing
- Context/mode system (all built-in contexts and modes)
- `engram_switch_mode`, `engram_get_config` tools
- Custom context/mode YAML support

### Phase 4 — Benchmark, Dashboard & Proof
- Full benchmark harness
- Baseline vs. assisted comparison tooling
- Metrics committed to store
- Live web dashboard
- Report generation

### Phase 5 — Advanced Providers, LSP & Ecosystem
- OpenAI, Voyage, ONNX embedding providers
- Cost estimation for paid providers
- Optional LSP backend for symbol resolution (deep onboarding depth)
- `engram_graph` (dependency/reference graph, cross-repo aware)
- `engram_related` (semantic similarity)
- Git hooks auto-install
- File watcher mode (opt-in)
- `f16` embedding precision support

---

## 20. SmartRouter Design — Sidecar Knowledge Surfacing

### 20.1 Design Decision

**Approach: Parallel Knowledge Query ("Sidecar")**

Every `engram_search` call executes two queries in parallel against separate index partitions: one against code/doc chunks, one against knowledge items (decisions, lessons, patterns, glossary, onboarding). Results are returned in two clearly separated sections so the agent always has visibility into both relevant code and institutional knowledge.

**Evolution path:** If token cost or relevancy becomes a problem at scale, the sidecar can be enhanced with signal-triggered filtering — adding query analysis that suppresses the knowledge sidecar for narrow lookups. This is a strictly additive change that doesn't alter the response format.

### 20.2 Response Format

`engram_search` returns a structured response with two top-level sections:

```json
{
  "query": "authentication middleware",
  "code_results": [
    {
      "chunk_id": "api/src/auth/middleware.ts#0",
      "kind": "function",
      "name": "authGuard",
      "signature": "export function authGuard(req, res, next): void",
      "file": "src/auth/middleware.ts",
      "repo": "api",
      "lines": [14, 47],
      "score": 0.92,
      "stale": false
    }
  ],
  "knowledge_results": [
    {
      "kind": "decision",
      "id": "decision-2026-03-08-001",
      "title": "JWT with refresh token rotation for API auth",
      "status": "accepted",
      "relevance_score": 0.84,
      "related_files": ["src/auth/jwt.ts", "src/auth/refresh.ts"],
      "created_at": "2026-03-08T14:30:00Z"
    },
    {
      "kind": "lesson",
      "id": "lesson-2026-03-06-003",
      "title": "Auth middleware must check token blacklist before expiry",
      "trigger": "Race condition where revoked tokens passed expiry check",
      "relevance_score": 0.79,
      "related_files": ["src/auth/middleware.ts"],
      "created_at": "2026-03-06T09:15:00Z"
    }
  ],
  "meta": {
    "code_count": 1,
    "knowledge_count": 2,
    "search_time_ms": 12,
    "compact": true
  }
}
```

### 20.3 Index Partitioning

The HNSW index is logically partitioned using a label/filter mechanism:

```
Partition 0: code chunks       (kind = function | class | method | type | impl | module)
Partition 1: doc chunks        (kind = doc_section | readme | comment_block)
Partition 2: knowledge items   (kind = decision | lesson | pattern | glossary | onboarding)
Partition 3: snapshots         (kind = snapshot)
```

The sidecar runs two filtered HNSW searches in parallel:
- Code query: partitions 0 + 1, top_k from config (default 10)
- Knowledge query: partition 2, top_k = 5

### 20.4 Knowledge Relevance Tuning

**Minimum relevance threshold:** Knowledge results below a configurable score floor (default 0.6) are dropped. This prevents tangentially related lessons from cluttering results.

**Recency boost:** A configurable boost factor (default 1.1x) is applied to knowledge items less than 30 days old. Multiplicative on the relevance score, so it only promotes items that are already reasonably relevant.

**Mode-aware tuning:** The active mode adjusts knowledge behavior (see §9.2). `plan` mode increases knowledge top_k and lowers min_relevance to surface more context. `edit` mode raises min_relevance to reduce noise.

### 20.5 Scope Interaction

| `scope` value | Code/doc query | Knowledge sidecar |
|---------------|----------------|-------------------|
| `"all"` (default) | Yes | Yes |
| `"code"` | Code only | No |
| `"docs"` | Docs only | No |
| `"knowledge"` | No | Knowledge only |

### 20.6 Token Budget Estimation

Worst-case per search (compact mode, all defaults):
- 10 code results × ~40 tokens = ~400 tokens
- 5 knowledge results × ~60 tokens = ~300 tokens
- Response metadata = ~30 tokens
- **Total: ~730 tokens per search**

Without sidecar: ~430 tokens. The sidecar adds ~300 tokens (~70% overhead in compact mode). Still a massive net reduction vs. agents without Engram reading 10-20 files at 500-2000 tokens each.

---

## 21. Resolved Design Decisions

1. **MCP transport in Rust.** Primary: `rmcp` crate for stdio + SSE. Fallback: hand-rolled JSON-RPC over stdio (~500 lines).

2. **SmartRouter design.** Sidecar approach with `scope` parameter for agent opt-out and `min_relevance` threshold. Full design in §20.

3. **Nx + Cargo integration.** Use `nx:run-commands` targets wrapping Cargo commands. Simplest approach with fewest moving parts.

4. **Storage format.** Split storage: JSONL metadata for diffability, binary embeddings for compactness.

5. **Snapshot storage.** Full transcripts preserved with tiered zstd compression (active → compressed → archived).

6. **Cross-repo symbol resolution.** Tree-sitter as default, LSP as optional upgrade. Resolution confidence tagged as `"exact"` or `"heuristic"`.

7. **Auto-update.** Layered: git hooks (default) + file watcher (opt-in) + GH Agentic Workflows (remote) + manual reindex.

8. **Embedding provider architecture.** Pluggable trait with 5 built-in providers. Model change = full reindex.

9. **Context/mode system.** Contexts fixed at session start, modes switchable during session. Both control tool surface and search behavior.

10. **Onboarding.** Automated, repeatable, idempotent. Three depth levels. Produces versioned YAML knowledge files committed to the store.

11. **Dashboard.** Embedded web UI served by the MCP server process. Live operational data, not a replacement for committed benchmark metrics.
