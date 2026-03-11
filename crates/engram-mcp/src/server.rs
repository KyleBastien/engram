use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use engram_core::{ChunkKind, Decision, EmbeddingProvider, GlossaryEntry, Lesson, OnboardingDepth, Partition, Pattern, Snapshot, SnapshotTier, SourceConfig};
use engram_query::{ChunkEntry, Direction, HybridSearch, SearchResult, SymbolGraph, DEFAULT_ALPHA};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse, METHOD_NOT_FOUND, PARSE_ERROR};
use crate::tools::{assessment_tool_definitions, graph_tool_definitions, knowledge_tool_definitions, onboarding_tool_definitions, phase1_tool_definitions, related_tool_definitions};

const SERVER_NAME: &str = "engram";
const SERVER_VERSION: &str = "0.1.0";
const PROTOCOL_VERSION: &str = "2024-11-05";

/// Records a single search event for session tracking.
#[derive(Debug, Clone)]
struct SearchEvent {
    query: String,
    scope: String,
    code_result_count: usize,
    knowledge_result_count: usize,
    search_time_ms: u64,
    timestamp: String,
}

/// Records a chunk retrieved during the session for staleness tracking.
#[derive(Debug, Clone)]
struct RetrievedChunk {
    chunk_id: String,
    file: String,
    stale: bool,
    indexed_at: String,
}

/// Thread-safe session tracker for accumulating search history.
#[derive(Debug, Default)]
struct SessionTracker {
    events: Mutex<Vec<SearchEvent>>,
    retrieved_chunks: Mutex<Vec<RetrievedChunk>>,
}

impl SessionTracker {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            retrieved_chunks: Mutex::new(Vec::new()),
        }
    }

    fn record(&self, event: SearchEvent) {
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }

    fn record_chunks(&self, chunks: Vec<RetrievedChunk>) {
        if let Ok(mut retrieved) = self.retrieved_chunks.lock() {
            retrieved.extend(chunks);
        }
    }

    fn snapshot(&self) -> Vec<SearchEvent> {
        self.events
            .lock()
            .map(|events| events.clone())
            .unwrap_or_default()
    }

    fn retrieved_chunks_snapshot(&self) -> Vec<RetrievedChunk> {
        self.retrieved_chunks
            .lock()
            .map(|chunks| chunks.clone())
            .unwrap_or_default()
    }
}

/// Holds the search engine and embedding provider needed for tool execution.
pub struct EngineState {
    pub search: HybridSearch,
    pub provider: Box<dyn EmbeddingProvider>,
    /// Map from repo name to local filesystem path, used to read source content for engram_lookup.
    pub source_roots: HashMap<String, PathBuf>,
    /// Time in milliseconds it took to boot/load the index.
    pub boot_time_ms: u64,
    /// Filesystem path to the engram store.
    pub store_path: String,
    /// Whether cache was used during boot ("hit", "miss", or "none").
    pub cache_status: String,
    /// In-memory session tracker for search history (used by engram_assess_context).
    session: SessionTracker,
    /// Cross-repo symbol dependency graph (used by engram_graph).
    pub graph: SymbolGraph,
}

/// MCP server that communicates over stdio using JSON-RPC 2.0.
pub struct McpServer {
    state: Option<EngineState>,
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

impl McpServer {
    /// Create a new MCP server without a search engine (tools that require search will return errors).
    pub fn new() -> Self {
        Self { state: None }
    }

    /// Create a new MCP server with a search engine and embedding provider.
    pub fn with_engine(search: HybridSearch, provider: Box<dyn EmbeddingProvider>) -> Self {
        Self {
            state: Some(EngineState {
                search,
                provider,
                source_roots: HashMap::new(),
                boot_time_ms: 0,
                store_path: String::new(),
                cache_status: "none".to_string(),
                session: SessionTracker::new(),
                graph: SymbolGraph::default(),
            }),
        }
    }

    /// Create a new MCP server with search engine, embedding provider, and source roots for content reading.
    pub fn with_engine_and_sources(
        search: HybridSearch,
        provider: Box<dyn EmbeddingProvider>,
        source_roots: HashMap<String, PathBuf>,
    ) -> Self {
        Self {
            state: Some(EngineState {
                search,
                provider,
                source_roots,
                boot_time_ms: 0,
                store_path: String::new(),
                cache_status: "none".to_string(),
                session: SessionTracker::new(),
                graph: SymbolGraph::default(),
            }),
        }
    }

    /// Set the cross-repo symbol graph on the engine state.
    pub fn set_graph(&mut self, graph: SymbolGraph) {
        if let Some(ref mut state) = self.state {
            state.graph = graph;
        }
    }

    /// Set boot info on the engine state (boot timing, store path, cache status).
    pub fn set_boot_info(&mut self, boot_time_ms: u64, store_path: String, cache_status: String) {
        if let Some(ref mut state) = self.state {
            state.boot_time_ms = boot_time_ms;
            state.store_path = store_path;
            state.cache_status = cache_status;
        }
    }

    /// Run the MCP server, reading JSON-RPC requests from the provided reader
    /// and writing responses to the provided writer. Logs go to stderr.
    pub async fn run<R, W>(&self, reader: R, mut writer: W) -> engram_core::Result<()>
    where
        R: tokio::io::AsyncRead + Unpin,
        W: tokio::io::AsyncWrite + Unpin,
    {
        let buf_reader = BufReader::new(reader);
        let mut lines = buf_reader.lines();

        while let Some(line) = lines.next_line().await.map_err(|e| {
            engram_core::EngramError::Mcp(format!("Failed to read from stdin: {e}"))
        })? {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            let request = match serde_json::from_str::<JsonRpcRequest>(&line) {
                Ok(req) => req,
                Err(e) => {
                    eprintln!("engram-mcp: parse error: {e}");
                    let resp =
                        JsonRpcResponse::error(None, PARSE_ERROR, format!("Parse error: {e}"));
                    write_response(&mut writer, &resp).await?;
                    continue;
                }
            };

            // Notifications (no id) don't get responses
            if request.id.is_none() {
                eprintln!("engram-mcp: notification: {}", request.method);
                continue;
            }

            let response = handle_request(&request, self.state.as_ref()).await;

            eprintln!(
                "engram-mcp: {} -> {}",
                request.method,
                if response.error.is_some() {
                    "error"
                } else {
                    "ok"
                }
            );

            write_response(&mut writer, &response).await?;
        }

        Ok(())
    }
}

async fn handle_request(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    match request.method.as_str() {
        "initialize" => handle_initialize(request),
        "tools/list" => handle_tools_list(request),
        "tools/call" => handle_tools_call(request, state).await,
        _ => JsonRpcResponse::error(
            request.id.clone(),
            METHOD_NOT_FOUND,
            format!("Method not found: {}", request.method),
        ),
    }
}

fn handle_initialize(request: &JsonRpcRequest) -> JsonRpcResponse {
    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            }
        }),
    )
}

fn handle_tools_list(request: &JsonRpcRequest) -> JsonRpcResponse {
    let mut tools = phase1_tool_definitions();
    tools.extend(knowledge_tool_definitions());
    tools.extend(onboarding_tool_definitions());
    tools.extend(assessment_tool_definitions());
    tools.extend(graph_tool_definitions());
    tools.extend(related_tool_definitions());
    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "tools": tools
        }),
    )
}

async fn handle_tools_call(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let params = request.params.as_ref();
    let tool_name = params
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("");

    match tool_name {
        "engram_search" => handle_engram_search(request, state).await,
        "engram_lookup" => handle_engram_lookup(request, state).await,
        "engram_status" => handle_engram_status(request, state).await,
        "engram_record_decision" => handle_engram_record_decision(request, state).await,
        "engram_record_lesson" => handle_engram_record_lesson(request, state).await,
        "engram_record_pattern" => handle_engram_record_pattern(request, state).await,
        "engram_record_glossary" => handle_engram_record_glossary(request, state).await,
        "engram_snapshot" => handle_engram_snapshot(request, state).await,
        "engram_onboard" => handle_engram_onboard(request, state).await,
        "engram_assess_context" => handle_engram_assess_context(request, state).await,
        "engram_check_staleness" => handle_engram_check_staleness(request, state).await,
        "engram_graph" => handle_engram_graph(request, state).await,
        "engram_related" => handle_engram_related(request, state).await,
        _ => JsonRpcResponse::success(
            request.id.clone(),
            json!({
                "content": [{
                    "type": "text",
                    "text": format!("Unknown tool: {}", tool_name)
                }],
                "isError": true
            }),
        ),
    }
}

/// Default minimum relevance score for knowledge results.
const DEFAULT_MIN_RELEVANCE: f64 = 0.6;

/// Recency boost multiplier for knowledge items less than 30 days old.
const RECENCY_BOOST: f64 = 1.1;

/// Number of days within which knowledge items receive a recency boost.
const RECENCY_DAYS: i64 = 30;

async fn handle_engram_search(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Search engine not initialized"),
    };

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"));

    let query = match args.and_then(|a| a.get("query")).and_then(|q| q.as_str()) {
        Some(q) if !q.is_empty() => q,
        _ => return tool_error_response(request, "query parameter is required"),
    };

    let scope = args
        .and_then(|a| a.get("scope"))
        .and_then(|s| s.as_str())
        .unwrap_or("all");
    let top_k = args
        .and_then(|a| a.get("top_k"))
        .and_then(|k| k.as_u64())
        .unwrap_or(10) as usize;
    let compact = args
        .and_then(|a| a.get("compact"))
        .and_then(|c| c.as_bool())
        .unwrap_or(false);
    let repo_filter = args
        .and_then(|a| a.get("repo"))
        .and_then(|r| r.as_str());

    // Validate scope
    if !matches!(scope, "code" | "docs" | "all" | "knowledge") {
        return tool_error_response(
            request,
            &format!("Invalid scope: {scope}. Must be code, docs, knowledge, or all"),
        );
    }

    let start = Instant::now();

    // Embed the query
    let embedding = match state.provider.embed(&[query]).await {
        Ok(mut vecs) if !vecs.is_empty() => vecs.remove(0),
        Ok(_) => return tool_error_response(request, "Embedding returned empty result"),
        Err(e) => return tool_error_response(request, &format!("Embedding failed: {e}")),
    };

    // Determine which partitions to search based on scope
    let search_code_docs = matches!(scope, "code" | "docs" | "all");
    let search_knowledge = matches!(scope, "knowledge" | "all");

    // Run parallel queries: code/docs and knowledge sidecar
    let code_doc_results = if search_code_docs {
        let partitions = match scope {
            "code" => vec![Partition::Code],
            "docs" => vec![Partition::Docs],
            _ => vec![Partition::Code, Partition::Docs],
        };
        match state
            .search
            .search(query, &embedding, top_k, DEFAULT_ALPHA, Some(&partitions))
            .await
        {
            Ok(r) => r,
            Err(e) => return tool_error_response(request, &format!("Search failed: {e}")),
        }
    } else {
        Vec::new()
    };

    let knowledge_results = if search_knowledge {
        match state
            .search
            .search(query, &embedding, top_k, DEFAULT_ALPHA, Some(&[Partition::Knowledge]))
            .await
        {
            Ok(r) => r,
            Err(e) => return tool_error_response(request, &format!("Knowledge search failed: {e}")),
        }
    } else {
        Vec::new()
    };

    // Apply repo filter if specified
    let code_doc_results: Vec<SearchResult> = if let Some(repo) = repo_filter {
        code_doc_results
            .into_iter()
            .filter(|r| r.repo == repo)
            .collect()
    } else {
        code_doc_results
    };
    let knowledge_results: Vec<SearchResult> = if let Some(repo) = repo_filter {
        knowledge_results
            .into_iter()
            .filter(|r| r.repo == repo)
            .collect()
    } else {
        knowledge_results
    };

    let now = chrono::Utc::now();

    // Apply recency boost and min_relevance filter to knowledge results
    let knowledge_results: Vec<SearchResult> = knowledge_results
        .into_iter()
        .map(|mut r| {
            // Apply recency boost if item is less than RECENCY_DAYS old
            if let Ok(created) = chrono::DateTime::parse_from_rfc3339(&r.indexed_at) {
                let age_days = (now - created.with_timezone(&chrono::Utc)).num_days();
                if age_days < RECENCY_DAYS {
                    r.score *= RECENCY_BOOST;
                }
            }
            r
        })
        .filter(|r| r.score >= DEFAULT_MIN_RELEVANCE)
        .collect();

    let code_count = code_doc_results.iter().filter(|r| is_code_kind(&r.kind)).count();
    let search_time_ms = start.elapsed().as_millis() as u64;

    // Record search event for session tracking
    state.session.record(SearchEvent {
        query: query.to_string(),
        scope: scope.to_string(),
        code_result_count: code_doc_results.len(),
        knowledge_result_count: knowledge_results.len(),
        search_time_ms,
        timestamp: chrono::Utc::now().to_rfc3339(),
    });

    // Record retrieved chunks for staleness tracking
    let mut retrieved: Vec<RetrievedChunk> = code_doc_results
        .iter()
        .map(|r| RetrievedChunk {
            chunk_id: r.chunk_id.clone(),
            file: r.file.clone(),
            stale: r.stale,
            indexed_at: r.indexed_at.clone(),
        })
        .collect();
    retrieved.extend(knowledge_results.iter().map(|r| RetrievedChunk {
        chunk_id: r.chunk_id.clone(),
        file: r.file.clone(),
        stale: r.stale,
        indexed_at: r.indexed_at.clone(),
    }));
    state.session.record_chunks(retrieved);

    // Format code/doc results
    let formatted_code: Vec<serde_json::Value> = code_doc_results
        .iter()
        .map(|r| format_code_result(r, compact))
        .collect();

    // Format knowledge results
    let formatted_knowledge: Vec<serde_json::Value> = knowledge_results
        .iter()
        .map(format_knowledge_result)
        .collect();

    let response_data = json!({
        "code_results": formatted_code,
        "knowledge_results": formatted_knowledge,
        "meta": {
            "code_count": code_count,
            "knowledge_count": formatted_knowledge.len(),
            "search_time_ms": search_time_ms,
            "compact": compact,
        }
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

fn format_code_result(r: &SearchResult, compact: bool) -> serde_json::Value {
    let name = if compact {
        truncate_name(&r.name, 50)
    } else {
        r.name.clone()
    };
    let mut obj = json!({
        "chunk_id": r.chunk_id,
        "score": r.score,
        "kind": serde_json::to_value(&r.kind).unwrap_or_default(),
        "name": name,
        "file": r.file,
        "repo": r.repo,
        "lines": [r.lines.0, r.lines.1],
        "stale": r.stale,
    });
    if !compact {
        if let Some(sig) = &r.signature {
            obj.as_object_mut()
                .unwrap()
                .insert("signature".to_string(), json!(sig));
        }
    }
    obj
}

fn format_knowledge_result(r: &SearchResult) -> serde_json::Value {
    // Extract kind and id from chunk_id format: "knowledge#{kind}:{id}"
    let (kind, id) = if let Some(rest) = r.chunk_id.strip_prefix("knowledge#") {
        if let Some((k, i)) = rest.split_once(':') {
            (k.to_string(), i.to_string())
        } else {
            (r.tags.first().cloned().unwrap_or_default(), rest.to_string())
        }
    } else {
        (r.tags.first().cloned().unwrap_or_default(), r.chunk_id.clone())
    };

    json!({
        "kind": kind,
        "id": id,
        "title": r.name,
        "relevance_score": r.score,
        "created_at": r.indexed_at,
    })
}

async fn handle_engram_lookup(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Search engine not initialized"),
    };

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"));

    let identifier = match args.and_then(|a| a.get("identifier")).and_then(|i| i.as_str()) {
        Some(id) if !id.is_empty() => id,
        _ => return tool_error_response(request, "identifier parameter is required"),
    };

    let include_content = args
        .and_then(|a| a.get("include_content"))
        .and_then(|c| c.as_bool())
        .unwrap_or(false);

    // Detect identifier type and dispatch to appropriate lookup
    let entries: Vec<&ChunkEntry> = if identifier.contains('#') {
        // Chunk ID lookup
        state
            .search
            .lookup_by_chunk_id(identifier)
            .into_iter()
            .collect()
    } else if identifier.contains('/') || identifier.contains('.') {
        // File path lookup
        state.search.lookup_by_file(identifier)
    } else {
        // Symbol name lookup
        state.search.lookup_by_symbol(identifier)
    };

    // Format results
    let formatted: Vec<serde_json::Value> = entries
        .iter()
        .map(|entry| {
            let mut obj = json!({
                "chunk_id": entry.chunk_id,
                "kind": serde_json::to_value(&entry.kind).unwrap_or_default(),
                "name": entry.name,
                "file": entry.file,
                "repo": entry.repo,
                "lines": [entry.start_line, entry.end_line],
                "stale": entry.stale,
            });
            if let Some(sig) = &entry.signature {
                obj.as_object_mut()
                    .unwrap()
                    .insert("signature".to_string(), json!(sig));
            }
            if include_content {
                let content = read_source_content(state, &entry.repo, &entry.file);
                obj.as_object_mut()
                    .unwrap()
                    .insert("content".to_string(), json!(content));
            }
            obj
        })
        .collect();

    // Record retrieved chunks for staleness tracking
    let retrieved: Vec<RetrievedChunk> = entries
        .iter()
        .map(|entry| RetrievedChunk {
            chunk_id: entry.chunk_id.clone(),
            file: entry.file.clone(),
            stale: entry.stale,
            indexed_at: entry.indexed_at.clone(),
        })
        .collect();
    state.session.record_chunks(retrieved);

    let response_data = json!({
        "results": formatted,
        "meta": {
            "count": formatted.len(),
        }
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

async fn handle_engram_status(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    let total_chunks = state.search.chunk_count();

    let source_repos: Vec<serde_json::Value> = state
        .search
        .repo_stats()
        .into_iter()
        .map(|(name, chunk_count, stale_chunks)| {
            json!({
                "name": name,
                "chunk_count": chunk_count,
                "stale_chunks": stale_chunks
            })
        })
        .collect();

    let response_data = json!({
        "total_chunks": total_chunks,
        "source_repos": source_repos,
        "cache_status": state.cache_status,
        "boot_time_ms": state.boot_time_ms,
        "embedding_provider": state.provider.name(),
        "store_path": state.store_path,
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

async fn handle_engram_record_decision(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    if state.store_path.is_empty() {
        return tool_error_response(request, "Store path not configured");
    }

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"));

    let title = match args.and_then(|a| a.get("title")).and_then(|v| v.as_str()) {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => return tool_error_response(request, "title parameter is required"),
    };

    let context = match args.and_then(|a| a.get("context")).and_then(|v| v.as_str()) {
        Some(c) if !c.is_empty() => c.to_string(),
        _ => return tool_error_response(request, "context parameter is required"),
    };

    let decision_text = match args.and_then(|a| a.get("decision")).and_then(|v| v.as_str()) {
        Some(d) if !d.is_empty() => d.to_string(),
        _ => return tool_error_response(request, "decision parameter is required"),
    };

    let consequences: Vec<String> = args
        .and_then(|a| a.get("consequences"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let related_files: Vec<String> = args
        .and_then(|a| a.get("related_files"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let status = args
        .and_then(|a| a.get("status"))
        .and_then(|v| v.as_str())
        .unwrap_or("accepted")
        .to_string();

    let decision_id = format!("DEC-{}", uuid::Uuid::new_v4().as_simple());
    let created_at = chrono::Utc::now().to_rfc3339();

    let decision = Decision {
        id: decision_id.clone(),
        title,
        status,
        context,
        decision: decision_text,
        consequences,
        related_files,
        contributed_by: "mcp-agent".to_string(),
        created_at,
        embedding_ref: None,
    };

    // Get embed text and embed it
    let embed_text = engram_store::decision_embed_text(&decision);
    let embedding = match state.provider.embed(&[&embed_text]).await {
        Ok(mut vecs) if !vecs.is_empty() => vecs.remove(0),
        Ok(_) => return tool_error_response(request, "Embedding returned empty result"),
        Err(e) => return tool_error_response(request, &format!("Embedding failed: {e}")),
    };

    let store_path = std::path::Path::new(&state.store_path);
    let dimensions = state.provider.dimensions();

    // Write decision with embedding
    let yaml_path = match engram_store::write_decision_with_embedding(
        store_path,
        &decision,
        &embedding,
        dimensions,
    ) {
        Ok(p) => p,
        Err(e) => {
            return tool_error_response(request, &format!("Failed to write decision: {e}"))
        }
    };

    let rel_path = yaml_path
        .strip_prefix(store_path)
        .unwrap_or(&yaml_path)
        .to_string_lossy()
        .to_string();

    // Commit to git
    let committed = match git2::Repository::open(store_path) {
        Ok(repo) => {
            match engram_store::commit_changes(&repo, &format!("knowledge: record decision {decision_id}")) {
                Ok(Some(_)) => true,
                Ok(None) => false,
                Err(e) => {
                    return tool_error_response(
                        request,
                        &format!("Failed to commit: {e}"),
                    )
                }
            }
        }
        Err(e) => {
            return tool_error_response(request, &format!("Failed to open repository: {e}"))
        }
    };

    let response_data = json!({
        "id": decision_id,
        "path": rel_path,
        "committed": committed,
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

async fn handle_engram_record_lesson(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    if state.store_path.is_empty() {
        return tool_error_response(request, "Store path not configured");
    }

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"));

    let title = match args.and_then(|a| a.get("title")).and_then(|v| v.as_str()) {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => return tool_error_response(request, "title parameter is required"),
    };

    let description = match args.and_then(|a| a.get("description")).and_then(|v| v.as_str()) {
        Some(d) if !d.is_empty() => d.to_string(),
        _ => return tool_error_response(request, "description parameter is required"),
    };

    let trigger = match args.and_then(|a| a.get("trigger")).and_then(|v| v.as_str()) {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => return tool_error_response(request, "trigger parameter is required"),
    };

    let resolution = args
        .and_then(|a| a.get("resolution"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let related_files: Vec<String> = args
        .and_then(|a| a.get("related_files"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let lesson_id = format!("LES-{}", uuid::Uuid::new_v4().as_simple());
    let created_at = chrono::Utc::now().to_rfc3339();

    let lesson = Lesson {
        id: lesson_id.clone(),
        title,
        description,
        trigger,
        resolution,
        related_files,
        contributed_by: "mcp-agent".to_string(),
        created_at,
        embedding_ref: None,
    };

    // Get embed text and embed it
    let embed_text = engram_store::lesson_embed_text(&lesson);
    let embedding = match state.provider.embed(&[&embed_text]).await {
        Ok(mut vecs) if !vecs.is_empty() => vecs.remove(0),
        Ok(_) => return tool_error_response(request, "Embedding returned empty result"),
        Err(e) => return tool_error_response(request, &format!("Embedding failed: {e}")),
    };

    let store_path = std::path::Path::new(&state.store_path);
    let dimensions = state.provider.dimensions();

    // Write lesson with embedding
    let yaml_path = match engram_store::write_lesson_with_embedding(
        store_path,
        &lesson,
        &embedding,
        dimensions,
    ) {
        Ok(p) => p,
        Err(e) => {
            return tool_error_response(request, &format!("Failed to write lesson: {e}"))
        }
    };

    let rel_path = yaml_path
        .strip_prefix(store_path)
        .unwrap_or(&yaml_path)
        .to_string_lossy()
        .to_string();

    // Commit to git
    let committed = match git2::Repository::open(store_path) {
        Ok(repo) => {
            match engram_store::commit_changes(&repo, &format!("knowledge: record lesson {lesson_id}")) {
                Ok(Some(_)) => true,
                Ok(None) => false,
                Err(e) => {
                    return tool_error_response(
                        request,
                        &format!("Failed to commit: {e}"),
                    )
                }
            }
        }
        Err(e) => {
            return tool_error_response(request, &format!("Failed to open repository: {e}"))
        }
    };

    let response_data = json!({
        "id": lesson_id,
        "path": rel_path,
        "committed": committed,
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

async fn handle_engram_record_pattern(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    if state.store_path.is_empty() {
        return tool_error_response(request, "Store path not configured");
    }

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"));

    let name = match args.and_then(|a| a.get("name")).and_then(|v| v.as_str()) {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => return tool_error_response(request, "name parameter is required"),
    };

    let description = match args.and_then(|a| a.get("description")).and_then(|v| v.as_str()) {
        Some(d) if !d.is_empty() => d.to_string(),
        _ => return tool_error_response(request, "description parameter is required"),
    };

    let examples: Vec<String> = args
        .and_then(|a| a.get("examples"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let anti_patterns: Vec<String> = args
        .and_then(|a| a.get("anti_patterns"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let pattern_id = format!("PAT-{}", uuid::Uuid::new_v4().as_simple());
    let created_at = chrono::Utc::now().to_rfc3339();

    let pattern = Pattern {
        id: pattern_id.clone(),
        name,
        description,
        examples,
        anti_patterns,
        contributed_by: "mcp-agent".to_string(),
        created_at,
        embedding_ref: None,
    };

    // Get embed text and embed it
    let embed_text = engram_store::pattern_embed_text(&pattern);
    let embedding = match state.provider.embed(&[&embed_text]).await {
        Ok(mut vecs) if !vecs.is_empty() => vecs.remove(0),
        Ok(_) => return tool_error_response(request, "Embedding returned empty result"),
        Err(e) => return tool_error_response(request, &format!("Embedding failed: {e}")),
    };

    let store_path = std::path::Path::new(&state.store_path);
    let dimensions = state.provider.dimensions();

    // Write pattern with embedding
    let yaml_path = match engram_store::write_pattern_with_embedding(
        store_path,
        &pattern,
        &embedding,
        dimensions,
    ) {
        Ok(p) => p,
        Err(e) => {
            return tool_error_response(request, &format!("Failed to write pattern: {e}"))
        }
    };

    let rel_path = yaml_path
        .strip_prefix(store_path)
        .unwrap_or(&yaml_path)
        .to_string_lossy()
        .to_string();

    // Commit to git
    let committed = match git2::Repository::open(store_path) {
        Ok(repo) => {
            match engram_store::commit_changes(&repo, &format!("knowledge: record pattern {pattern_id}")) {
                Ok(Some(_)) => true,
                Ok(None) => false,
                Err(e) => {
                    return tool_error_response(
                        request,
                        &format!("Failed to commit: {e}"),
                    )
                }
            }
        }
        Err(e) => {
            return tool_error_response(request, &format!("Failed to open repository: {e}"))
        }
    };

    let response_data = json!({
        "id": pattern_id,
        "path": rel_path,
        "committed": committed,
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

async fn handle_engram_record_glossary(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    if state.store_path.is_empty() {
        return tool_error_response(request, "Store path not configured");
    }

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"));

    let term = match args.and_then(|a| a.get("term")).and_then(|v| v.as_str()) {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => return tool_error_response(request, "term parameter is required"),
    };

    let definition = match args.and_then(|a| a.get("definition")).and_then(|v| v.as_str()) {
        Some(d) if !d.is_empty() => d.to_string(),
        _ => return tool_error_response(request, "definition parameter is required"),
    };

    let context = args
        .and_then(|a| a.get("context"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let store_path = std::path::Path::new(&state.store_path);

    // Check if term already exists to determine action
    let terms_path = store_path.join("knowledge/glossary/terms.yaml");
    let action = if terms_path.exists() {
        let content = match std::fs::read_to_string(&terms_path) {
            Ok(c) => c,
            Err(e) => {
                return tool_error_response(
                    request,
                    &format!("Failed to read terms.yaml: {e}"),
                )
            }
        };
        let existing: Vec<GlossaryEntry> = match serde_yaml::from_str(&content) {
            Ok(entries) => entries,
            Err(e) => {
                return tool_error_response(
                    request,
                    &format!("Failed to parse terms.yaml: {e}"),
                )
            }
        };
        if existing.iter().any(|e| e.term == term) {
            "updated"
        } else {
            "created"
        }
    } else {
        "created"
    };

    let created_at = chrono::Utc::now().to_rfc3339();

    let entry = GlossaryEntry {
        term: term.clone(),
        definition,
        context,
        contributed_by: "mcp-agent".to_string(),
        created_at,
    };

    // Write glossary entry (handles create/update logic)
    match engram_store::write_glossary_entry(store_path, &entry) {
        Ok(_) => {}
        Err(e) => {
            return tool_error_response(
                request,
                &format!("Failed to write glossary entry: {e}"),
            )
        }
    };

    // Commit to git
    let committed = match git2::Repository::open(store_path) {
        Ok(repo) => {
            match engram_store::commit_changes(
                &repo,
                &format!("knowledge: record glossary term {term}"),
            ) {
                Ok(Some(_)) => true,
                Ok(None) => false,
                Err(e) => {
                    return tool_error_response(
                        request,
                        &format!("Failed to commit: {e}"),
                    )
                }
            }
        }
        Err(e) => {
            return tool_error_response(request, &format!("Failed to open repository: {e}"))
        }
    };

    let response_data = json!({
        "term": term,
        "action": action,
        "committed": committed,
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

async fn handle_engram_snapshot(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    if state.store_path.is_empty() {
        return tool_error_response(request, "Store path not configured");
    }

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"));

    let session_id = match args.and_then(|a| a.get("session_id")).and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => return tool_error_response(request, "session_id parameter is required"),
    };

    let summary = match args.and_then(|a| a.get("summary")).and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => return tool_error_response(request, "summary parameter is required"),
    };

    let key_context: Vec<String> = match args.and_then(|a| a.get("key_context")).and_then(|v| v.as_array()) {
        Some(arr) => arr.iter()
            .filter_map(|item| item.as_str().map(|s| s.to_string()))
            .collect(),
        None => return tool_error_response(request, "key_context parameter is required"),
    };

    let full_transcript = match args.and_then(|a| a.get("full_transcript")).and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => return tool_error_response(request, "full_transcript parameter is required"),
    };

    let created_at = chrono::Utc::now().to_rfc3339();

    let snapshot = Snapshot {
        session_id: session_id.clone(),
        summary: summary.clone(),
        key_context,
        full_transcript,
        created_at,
        tier: SnapshotTier::Active,
        embedding_ref: None,
    };

    // Embed summary + key_context for searchability
    let embed_text = engram_store::snapshot_embed_text(&snapshot);
    let embedding = match state.provider.embed(&[&embed_text]).await {
        Ok(mut vecs) if !vecs.is_empty() => vecs.remove(0),
        Ok(_) => return tool_error_response(request, "Embedding returned empty result"),
        Err(e) => return tool_error_response(request, &format!("Embedding failed: {e}")),
    };

    let store_path = std::path::Path::new(&state.store_path);
    let dimensions = state.provider.dimensions();

    // Write snapshot with embedding
    let yaml_path = match engram_store::write_snapshot_with_embedding(
        store_path,
        &snapshot,
        &embedding,
        dimensions,
    ) {
        Ok(p) => p,
        Err(e) => {
            return tool_error_response(request, &format!("Failed to write snapshot: {e}"))
        }
    };

    let rel_path = yaml_path
        .strip_prefix(store_path)
        .unwrap_or(&yaml_path)
        .to_string_lossy()
        .to_string();

    // Commit to git
    let committed = match git2::Repository::open(store_path) {
        Ok(repo) => {
            match engram_store::commit_changes(
                &repo,
                &format!("engram: snapshot session {session_id}"),
            ) {
                Ok(Some(_)) => true,
                Ok(None) => false,
                Err(e) => {
                    return tool_error_response(
                        request,
                        &format!("Failed to commit: {e}"),
                    )
                }
            }
        }
        Err(e) => {
            return tool_error_response(request, &format!("Failed to open repository: {e}"))
        }
    };

    let response_data = json!({
        "session_id": session_id,
        "path": rel_path,
        "tier": "active",
        "committed": committed,
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

/// Read raw source text from a source repo for the given file path.
fn read_source_content(state: &EngineState, repo: &str, file: &str) -> Option<String> {
    let root = state.source_roots.get(repo)?;
    let path = root.join(file);
    std::fs::read_to_string(path).ok()
}

async fn handle_engram_onboard(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"));

    // Optional: repo name — defaults to first source root
    let repo_name = args
        .and_then(|a| a.get("repo"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Resolve repo path from source_roots
    let (resolved_name, repo_path) = if repo_name.is_empty() {
        match state.source_roots.iter().next() {
            Some((name, path)) => (name.clone(), path.clone()),
            None => return tool_error_response(request, "No source roots configured"),
        }
    } else {
        match state.source_roots.get(repo_name) {
            Some(path) => (repo_name.to_string(), path.clone()),
            None => return tool_error_response(
                request,
                &format!("Unknown repo: {}. Available: {:?}", repo_name, state.source_roots.keys().collect::<Vec<_>>()),
            ),
        }
    };

    // Optional: depth — defaults to standard
    let depth = match args.and_then(|a| a.get("depth")).and_then(|v| v.as_str()) {
        Some("quick") => OnboardingDepth::Quick,
        Some("deep") => OnboardingDepth::Deep,
        Some("standard") | None => OnboardingDepth::Standard,
        Some(other) => return tool_error_response(
            request,
            &format!("Invalid depth '{}'. Must be quick, standard, or deep", other),
        ),
    };

    let store_path = std::path::PathBuf::from(&state.store_path);
    let source_config = SourceConfig {
        name: resolved_name,
        path: repo_path.to_string_lossy().to_string(),
        include: vec![],
        exclude: vec![],
    };

    match engram_ingest::run_onboarding(&repo_path, &store_path, &source_config, depth).await {
        Ok(report) => {
            let result = serde_json::to_value(&report).unwrap_or_default();
            JsonRpcResponse::success(
                request.id.clone(),
                json!({
                    "content": [{
                        "type": "text",
                        "text": result.to_string()
                    }],
                    "isError": false
                }),
            )
        }
        Err(e) => tool_error_response(request, &format!("Onboarding failed: {}", e)),
    }
}

async fn handle_engram_assess_context(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    let args = request.params.as_ref().and_then(|p| p.get("arguments"));
    let task_description = args
        .and_then(|a| a.get("task_description"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    let events = state.session.snapshot();
    let search_count = events.len();

    // Collect unique queries
    let mut unique_queries: Vec<String> = Vec::new();
    for e in &events {
        if !unique_queries.contains(&e.query) {
            unique_queries.push(e.query.clone());
        }
    }

    // Aggregate by scope
    let mut scope_counts: HashMap<String, usize> = HashMap::new();
    for e in &events {
        *scope_counts.entry(e.scope.clone()).or_insert(0) += 1;
    }

    // Total results and timing
    let total_code_results: usize = events.iter().map(|e| e.code_result_count).sum();
    let total_knowledge_results: usize = events.iter().map(|e| e.knowledge_result_count).sum();
    let total_search_time_ms: u64 = events.iter().map(|e| e.search_time_ms).sum();

    // Build queries list for response
    let queries_list: Vec<serde_json::Value> = events
        .iter()
        .map(|e| {
            json!({
                "query": e.query,
                "scope": e.scope,
                "code_results": e.code_result_count,
                "knowledge_results": e.knowledge_result_count,
                "timestamp": e.timestamp,
            })
        })
        .collect();

    // Build the structured evaluation prompt
    let task_line = if task_description.is_empty() {
        String::new()
    } else {
        format!("\nTask: {task_description}\n")
    };

    let evaluation_prompt = format!(
        "## Context Assessment{task_line}\n\
         You have performed {search_count} search(es) with {unique_count} unique query/queries.\n\
         - Code/doc results retrieved: {total_code_results}\n\
         - Knowledge items surfaced: {total_knowledge_results}\n\
         - Total search time: {total_search_time_ms}ms\n\n\
         ### Evaluation\n\
         Consider the following before proceeding:\n\
         1. Have you searched for all relevant concepts, files, and symbols related to the task?\n\
         2. Did the knowledge results provide sufficient architectural context and past decisions?\n\
         3. Are there any gaps — areas you suspect are relevant but haven't queried yet?\n\
         4. Is the retrieved context fresh enough, or should you check for staleness?\n\n\
         If you believe you have enough context, proceed with the task. \
         Otherwise, perform additional targeted searches before continuing.",
        unique_count = unique_queries.len(),
    );

    let response_data = json!({
        "session_summary": {
            "search_count": search_count,
            "unique_queries": unique_queries,
            "scope_distribution": scope_counts,
            "total_code_results": total_code_results,
            "total_knowledge_results": total_knowledge_results,
            "total_search_time_ms": total_search_time_ms,
        },
        "searches": queries_list,
        "evaluation_prompt": evaluation_prompt,
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

async fn handle_engram_check_staleness(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    let chunks = state.session.retrieved_chunks_snapshot();

    if chunks.is_empty() {
        let response_data = json!({
            "total_retrieved": 0,
            "stale_count": 0,
            "fresh_count": 0,
            "files": [],
            "evaluation_prompt": "## Staleness Check\n\nNo chunks have been retrieved in this session yet. \
                Perform a search first, then call engram_check_staleness to verify freshness."
        });

        return JsonRpcResponse::success(
            request.id.clone(),
            json!({
                "content": [{
                    "type": "text",
                    "text": response_data.to_string()
                }],
                "isError": false
            }),
        );
    }

    // Group chunks by file and compute per-file staleness
    let mut file_map: HashMap<String, Vec<&RetrievedChunk>> = HashMap::new();
    for chunk in &chunks {
        file_map
            .entry(if chunk.file.is_empty() {
                chunk.chunk_id.clone()
            } else {
                chunk.file.clone()
            })
            .or_default()
            .push(chunk);
    }

    let stale_count = chunks.iter().filter(|c| c.stale).count();
    let fresh_count = chunks.len() - stale_count;

    // Build per-file summary
    let mut file_summaries: Vec<serde_json::Value> = file_map
        .iter()
        .map(|(file, file_chunks)| {
            let stale_in_file = file_chunks.iter().filter(|c| c.stale).count();
            let chunk_ids: Vec<&str> = file_chunks.iter().map(|c| c.chunk_id.as_str()).collect();
            let latest_indexed = file_chunks
                .iter()
                .filter(|c| !c.indexed_at.is_empty())
                .map(|c| c.indexed_at.as_str())
                .max()
                .unwrap_or("unknown");

            json!({
                "file": file,
                "total_chunks": file_chunks.len(),
                "stale_chunks": stale_in_file,
                "chunk_ids": chunk_ids,
                "latest_indexed_at": latest_indexed,
            })
        })
        .collect();
    file_summaries.sort_by(|a, b| {
        b["stale_chunks"]
            .as_u64()
            .unwrap_or(0)
            .cmp(&a["stale_chunks"].as_u64().unwrap_or(0))
    });

    // Deduplicate chunk IDs for total unique count
    let mut seen_ids: Vec<String> = Vec::new();
    for chunk in &chunks {
        if !seen_ids.contains(&chunk.chunk_id) {
            seen_ids.push(chunk.chunk_id.clone());
        }
    }

    let stale_pct = if chunks.is_empty() {
        0.0
    } else {
        (stale_count as f64 / chunks.len() as f64) * 100.0
    };

    let stale_files: Vec<&str> = file_summaries
        .iter()
        .filter(|f| f["stale_chunks"].as_u64().unwrap_or(0) > 0)
        .filter_map(|f| f["file"].as_str())
        .collect();

    let evaluation_prompt = format!(
        "## Staleness Check\n\n\
         Retrieved {total} chunks ({unique} unique) across {file_count} files.\n\
         - Fresh: {fresh_count} ({fresh_pct:.0}%)\n\
         - Stale: {stale_count} ({stale_pct:.0}%)\n\n\
         {stale_detail}\
         ### Decision\n\
         Consider the following before proceeding:\n\
         1. Are the stale chunks in critical files for your current task?\n\
         2. Could the staleness reflect minor formatting changes, or substantive logic changes?\n\
         3. Would re-indexing (via a new ingest) resolve the staleness before you proceed?\n\
         4. Is it safe to proceed with stale context, noting that some information may be outdated?\n\n\
         If staleness is in non-critical files or unlikely to affect correctness, proceed with caution. \
         Otherwise, consider re-indexing or searching for updated context.",
        total = chunks.len(),
        unique = seen_ids.len(),
        file_count = file_map.len(),
        fresh_pct = 100.0 - stale_pct,
        stale_detail = if stale_files.is_empty() {
            "All retrieved chunks are fresh.\n\n".to_string()
        } else {
            format!(
                "Stale files: {}\n\n",
                stale_files.join(", ")
            )
        },
    );

    let response_data = json!({
        "total_retrieved": chunks.len(),
        "unique_chunks": seen_ids.len(),
        "stale_count": stale_count,
        "fresh_count": fresh_count,
        "stale_percentage": (stale_pct * 10.0).round() / 10.0,
        "files": file_summaries,
        "evaluation_prompt": evaluation_prompt,
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

async fn handle_engram_graph(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Engine not initialized"),
    };

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"))
        .cloned()
        .unwrap_or(json!({}));

    let symbol = match args.get("symbol").and_then(|s| s.as_str()) {
        Some(s) => s,
        None => return tool_error_response(request, "Missing required parameter: symbol"),
    };

    let direction = match args.get("direction").and_then(|d| d.as_str()).unwrap_or("both") {
        "callers" => Direction::Callers,
        "callees" => Direction::Callees,
        "both" => Direction::Both,
        other => {
            return tool_error_response(
                request,
                &format!("Invalid direction '{}': must be callers, callees, or both", other),
            )
        }
    };

    let depth = args
        .get("depth")
        .and_then(|d| d.as_u64())
        .unwrap_or(2) as usize;

    let result = state.graph.traverse(symbol, direction, depth);

    let nodes: Vec<serde_json::Value> = result
        .nodes
        .iter()
        .map(|n| {
            json!({
                "chunk_id": n.chunk_id,
                "name": n.name,
                "file": n.file,
                "repo": n.repo,
                "kind": n.kind,
            })
        })
        .collect();

    let edges: Vec<serde_json::Value> = result
        .edges
        .iter()
        .map(|e| {
            json!({
                "source": {
                    "name": e.source.name,
                    "file": e.source.file,
                    "repo": e.source.repo,
                },
                "target": {
                    "name": e.target.name,
                    "file": e.target.file,
                    "repo": e.target.repo,
                },
                "relationship": e.relationship,
            })
        })
        .collect();

    let response_data = json!({
        "symbol": symbol,
        "direction": args.get("direction").and_then(|d| d.as_str()).unwrap_or("both"),
        "depth": depth,
        "nodes": nodes,
        "edges": edges,
        "node_count": nodes.len(),
        "edge_count": edges.len(),
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

async fn handle_engram_related(
    request: &JsonRpcRequest,
    state: Option<&EngineState>,
) -> JsonRpcResponse {
    let state = match state {
        Some(s) => s,
        None => return tool_error_response(request, "Search engine not initialized"),
    };

    let args = request
        .params
        .as_ref()
        .and_then(|p| p.get("arguments"))
        .cloned()
        .unwrap_or(json!({}));

    let chunk_id = args.get("chunk_id").and_then(|c| c.as_str());
    let symbol = args.get("symbol").and_then(|s| s.as_str());

    if chunk_id.is_none() && symbol.is_none() {
        return tool_error_response(
            request,
            "At least one of chunk_id or symbol is required",
        );
    }

    let top_k = args
        .get("top_k")
        .and_then(|k| k.as_u64())
        .unwrap_or(10) as usize;

    let results = match state.search.find_related(chunk_id, symbol, top_k) {
        Ok(r) => r,
        Err(e) => return tool_error_response(request, &format!("Related search failed: {e}")),
    };

    let formatted: Vec<serde_json::Value> = results
        .iter()
        .map(|r| format_code_result(r, false))
        .collect();

    let response_data = json!({
        "results": formatted,
        "meta": {
            "result_count": formatted.len(),
            "chunk_id": chunk_id,
            "symbol": symbol,
            "top_k": top_k,
        }
    });

    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": response_data.to_string()
            }],
            "isError": false
        }),
    )
}

fn tool_error_response(request: &JsonRpcRequest, message: &str) -> JsonRpcResponse {
    JsonRpcResponse::success(
        request.id.clone(),
        json!({
            "content": [{
                "type": "text",
                "text": message
            }],
            "isError": true
        }),
    )
}

fn is_code_kind(kind: &ChunkKind) -> bool {
    matches!(
        kind,
        ChunkKind::Function
            | ChunkKind::Class
            | ChunkKind::Method
            | ChunkKind::Type
            | ChunkKind::Impl
            | ChunkKind::Module
            | ChunkKind::Other
    )
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("{}...", &name[..max_len - 3])
    }
}

async fn write_response<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    response: &JsonRpcResponse,
) -> engram_core::Result<()> {
    let json = serde_json::to_string(response)
        .map_err(|e| engram_core::EngramError::Serialize(e.to_string()))?;
    writer
        .write_all(json.as_bytes())
        .await
        .map_err(|e| engram_core::EngramError::Mcp(format!("Failed to write response: {e}")))?;
    writer
        .write_all(b"\n")
        .await
        .map_err(|e| engram_core::EngramError::Mcp(format!("Failed to write newline: {e}")))?;
    writer
        .flush()
        .await
        .map_err(|e| engram_core::EngramError::Mcp(format!("Failed to flush: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use engram_core::EmbedError;
    use engram_query::{Bm25Document, Bm25Index, HnswIndex, DOCS_KEY_OFFSET, KNOWLEDGE_KEY_OFFSET};

    fn make_request(id: i64, method: &str, params: Option<serde_json::Value>) -> String {
        let mut req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
        });
        if let Some(p) = params {
            req["params"] = p;
        }
        serde_json::to_string(&req).unwrap()
    }

    async fn run_server(input: &str) -> Vec<JsonRpcResponse> {
        let server = McpServer::new();
        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    // --- Mock embedding provider ---

    struct MockEmbeddingProvider {
        dims: usize,
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        fn name(&self) -> &str {
            "mock/test"
        }
        fn dimensions(&self) -> usize {
            self.dims
        }
        fn max_batch_size(&self) -> usize {
            32
        }
        async fn embed(
            &self,
            texts: &[&str],
        ) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
            // Return a simple deterministic vector based on text length
            Ok(texts
                .iter()
                .map(|t| {
                    (0..self.dims)
                        .map(|i| ((i as f32 + t.len() as f32) * 0.1).sin())
                        .collect()
                })
                .collect())
        }
    }

    // --- Test helpers for building a test search engine ---

    fn make_vector(dimensions: usize, seed: f32) -> Vec<f32> {
        (0..dimensions)
            .map(|i| ((i as f32 + seed) * 0.1).sin())
            .collect()
    }

    fn make_entry(key: u64, name: &str, file: &str, kind: ChunkKind) -> (u64, ChunkEntry) {
        (
            key,
            ChunkEntry {
                chunk_id: format!("repo#{}#{}", file, name),
                kind,
                name: name.to_string(),
                signature: Some(format!("fn {}()", name)),
                file: file.to_string(),
                repo: "test-repo".to_string(),
                start_line: 1,
                end_line: 10,
                stale: false,
                tags: Vec::new(),
                indexed_at: String::new(),
            },
        )
    }

    fn make_doc(key: u64, name: &str, signature: Option<&str>, tags: &[&str]) -> Bm25Document {
        Bm25Document {
            key,
            name: name.to_string(),
            signature: signature.map(|s| s.to_string()),
            tags: tags.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn build_test_engine() -> (HybridSearch, Box<dyn EmbeddingProvider>) {
        let dims = 32;
        let v0 = make_vector(dims, 0.0);
        let v1 = make_vector(dims, 1.0);
        let v2 = make_vector(dims, 2.0);
        let v3 = make_vector(dims, 3.0);

        let doc_key = DOCS_KEY_OFFSET;
        let entries: Vec<(u64, &[f32])> = vec![(0, &v0), (1, &v1), (2, &v2), (doc_key, &v3)];
        let hnsw = HnswIndex::build(&entries, dims).unwrap();

        let docs = vec![
            make_doc(
                0,
                "calculate_total",
                Some("fn calculate_total(items: &[Item]) -> f64"),
                &["math"],
            ),
            make_doc(
                1,
                "render_button",
                Some("fn render_button(label: &str)"),
                &["ui"],
            ),
            make_doc(
                2,
                "parse_config",
                Some("fn parse_config(path: &Path) -> Config"),
                &["config"],
            ),
            make_doc(doc_key, "README", None, &["docs"]),
        ];
        let bm25 = Bm25Index::build(&docs);

        let metadata: HashMap<u64, ChunkEntry> = vec![
            make_entry(0, "calculate_total", "src/math.rs", ChunkKind::Function),
            make_entry(1, "render_button", "src/ui.rs", ChunkKind::Function),
            make_entry(2, "parse_config", "src/config.rs", ChunkKind::Function),
        ]
        .into_iter()
        .collect();
        // README in docs partition (key >= DOCS_KEY_OFFSET)
        let mut metadata = metadata;
        metadata.insert(doc_key, ChunkEntry {
            chunk_id: "repo#README.md#README".to_string(),
            kind: ChunkKind::Readme,
            name: "README".to_string(),
            signature: None,
            file: "README.md".to_string(),
            repo: "test-repo".to_string(),
            start_line: 1,
            end_line: 10,
            stale: false,
            tags: vec!["docs".to_string()],
            indexed_at: String::new(),
        });

        let search = HybridSearch::new(hnsw, bm25, metadata);
        let provider: Box<dyn EmbeddingProvider> = Box::new(MockEmbeddingProvider { dims });
        (search, provider)
    }

    fn build_test_engine_with_knowledge() -> (HybridSearch, Box<dyn EmbeddingProvider>) {
        let dims = 32;
        let v0 = make_vector(dims, 0.0);
        let v1 = make_vector(dims, 1.0);
        let v2 = make_vector(dims, 2.0);
        let v3 = make_vector(dims, 3.0);
        let v_kn = make_vector(dims, 5.0);

        let doc_key = DOCS_KEY_OFFSET;
        let kn_key = KNOWLEDGE_KEY_OFFSET;

        let entries: Vec<(u64, &[f32])> = vec![
            (0, &v0), (1, &v1), (2, &v2), (doc_key, &v3), (kn_key, &v_kn),
        ];
        let hnsw = HnswIndex::build(&entries, dims).unwrap();

        let docs = vec![
            make_doc(0, "calculate_total", Some("fn calculate_total(items: &[Item]) -> f64"), &["math"]),
            make_doc(1, "render_button", Some("fn render_button(label: &str)"), &["ui"]),
            make_doc(2, "parse_config", Some("fn parse_config(path: &Path) -> Config"), &["config"]),
            make_doc(doc_key, "README", None, &["docs"]),
            make_doc(kn_key, "use_hnsw_decision", None, &["decision"]),
        ];
        let bm25 = Bm25Index::build(&docs);

        let mut metadata: HashMap<u64, ChunkEntry> = vec![
            make_entry(0, "calculate_total", "src/math.rs", ChunkKind::Function),
            make_entry(1, "render_button", "src/ui.rs", ChunkKind::Function),
            make_entry(2, "parse_config", "src/config.rs", ChunkKind::Function),
        ]
        .into_iter()
        .collect();
        metadata.insert(doc_key, ChunkEntry {
            chunk_id: "repo#README.md#README".to_string(),
            kind: ChunkKind::Readme,
            name: "README".to_string(),
            signature: None,
            file: "README.md".to_string(),
            repo: "test-repo".to_string(),
            start_line: 1,
            end_line: 10,
            stale: false,
            tags: vec!["docs".to_string()],
            indexed_at: String::new(),
        });

        // Add a knowledge item with a recent created_at
        metadata.insert(kn_key, ChunkEntry {
            chunk_id: "knowledge#decision:DEC-001".to_string(),
            kind: ChunkKind::Knowledge,
            name: "Use HNSW for vector search".to_string(),
            signature: None,
            file: String::new(),
            repo: "knowledge".to_string(),
            start_line: 0,
            end_line: 0,
            stale: false,
            tags: vec!["decision".to_string()],
            indexed_at: chrono::Utc::now().to_rfc3339(),
        });

        let search = HybridSearch::new(hnsw, bm25, metadata);
        let provider: Box<dyn EmbeddingProvider> = Box::new(MockEmbeddingProvider { dims });
        (search, provider)
    }

    async fn run_server_with_engine(input: &str) -> Vec<JsonRpcResponse> {
        let (search, provider) = build_test_engine();
        let server = McpServer::with_engine(search, provider);
        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    async fn run_server_with_knowledge_engine(input: &str) -> Vec<JsonRpcResponse> {
        let (search, provider) = build_test_engine_with_knowledge();
        let server = McpServer::with_engine(search, provider);
        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn parse_tool_text(response: &JsonRpcResponse) -> serde_json::Value {
        let result = response.result.as_ref().unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        serde_json::from_str(text).unwrap()
    }

    fn matches_scope(kind: &ChunkKind, scope: &str) -> bool {
        match scope {
            "code" => is_code_kind(kind),
            "docs" => matches!(
                kind,
                ChunkKind::DocSection | ChunkKind::Readme | ChunkKind::CommentBlock
            ),
            "knowledge" => matches!(kind, ChunkKind::Knowledge),
            _ => true,
        }
    }

    // --- Existing server tests (updated for McpServer::new()) ---

    #[tokio::test]
    async fn test_initialize_returns_capabilities() {
        let input = make_request(1, "initialize", Some(json!({})));
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
        assert!(result["capabilities"]["tools"].is_object());
        assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
        assert_eq!(result["serverInfo"]["version"], SERVER_VERSION);
    }

    #[tokio::test]
    async fn test_tools_list_returns_phase1_tools() {
        let input = make_request(1, "tools/list", None);
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 13);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"engram_search"));
        assert!(names.contains(&"engram_lookup"));
        assert!(names.contains(&"engram_status"));
        assert!(names.contains(&"engram_related"));
        assert!(names.contains(&"engram_graph"));
        assert!(names.contains(&"engram_record_decision"));
        assert!(names.contains(&"engram_record_lesson"));
        assert!(names.contains(&"engram_record_pattern"));
        assert!(names.contains(&"engram_record_glossary"));
        assert!(names.contains(&"engram_snapshot"));
        assert!(names.contains(&"engram_onboard"));
        assert!(names.contains(&"engram_assess_context"));
        assert!(names.contains(&"engram_check_staleness"));
    }

    #[tokio::test]
    async fn test_tools_call_dispatches() {
        // Without engine, engram_status returns error
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not initialized"));
    }

    #[tokio::test]
    async fn test_unknown_tool_returns_error_content() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "nonexistent_tool", "arguments": {}})),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_unknown_method_returns_error() {
        let input = make_request(1, "nonexistent/method", None);
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let error = responses[0].error.as_ref().unwrap();
        assert_eq!(error.code, METHOD_NOT_FOUND);
        assert!(error.message.contains("nonexistent/method"));
    }

    #[tokio::test]
    async fn test_parse_error_returns_error_response() {
        let input = "this is not json\n";
        let responses = run_server(input).await;
        assert_eq!(responses.len(), 1);
        let error = responses[0].error.as_ref().unwrap();
        assert_eq!(error.code, PARSE_ERROR);
    }

    #[tokio::test]
    async fn test_notification_no_response() {
        // Notification has no id
        let input = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let responses = run_server(&format!("{input}\n")).await;
        assert_eq!(responses.len(), 0);
    }

    #[tokio::test]
    async fn test_empty_lines_skipped() {
        let req = make_request(1, "initialize", Some(json!({})));
        let input = format!("\n\n{req}\n\n");
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
    }

    #[tokio::test]
    async fn test_multiple_requests_in_sequence() {
        let r1 = make_request(1, "initialize", Some(json!({})));
        let r2 = make_request(2, "tools/list", None);
        let input = format!("{r1}\n{r2}\n");
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].id, Some(serde_json::Value::Number(1.into())));
        assert_eq!(responses[1].id, Some(serde_json::Value::Number(2.into())));
    }

    #[tokio::test]
    async fn test_response_id_matches_request() {
        let input = make_request(42, "initialize", Some(json!({})));
        let responses = run_server(&input).await;
        assert_eq!(responses[0].id, Some(serde_json::Value::Number(42.into())));
    }

    #[tokio::test]
    async fn test_response_is_valid_jsonrpc() {
        let input = make_request(1, "initialize", Some(json!({})));
        let responses = run_server(&input).await;
        assert_eq!(responses[0].jsonrpc, "2.0");
    }

    #[tokio::test]
    async fn test_logs_to_stderr() {
        // This test verifies the server doesn't write non-JSON to stdout.
        // The run_server helper captures stdout output and parses it as JSON.
        // If logging went to stdout, parsing would fail.
        let input = make_request(1, "initialize", Some(json!({})));
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        // If we got here, all output was valid JSON (no stderr leakage into stdout)
    }

    // --- engram_search tool tests ---

    #[tokio::test]
    async fn test_search_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "test"}})),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not initialized"));
    }

    #[tokio::test]
    async fn test_search_missing_query_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("query"));
    }

    #[tokio::test]
    async fn test_search_returns_results_with_meta() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate"}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        assert!(data["code_results"].is_array());
        assert!(data["knowledge_results"].is_array());
        assert!(data["meta"].is_object());
        assert!(data["meta"]["code_count"].is_number());
        assert!(data["meta"]["knowledge_count"].is_number());
        assert!(data["meta"]["search_time_ms"].is_number());
        assert!(data["meta"]["compact"].is_boolean());
    }

    #[tokio::test]
    async fn test_search_results_have_required_fields() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["code_results"].as_array().unwrap();
        assert!(!results.is_empty());

        let r = &results[0];
        assert!(r["chunk_id"].is_string());
        assert!(r["score"].is_number());
        assert!(r["kind"].is_string());
        assert!(r["name"].is_string());
        assert!(r["file"].is_string());
        assert!(r["repo"].is_string());
        assert!(r["lines"].is_array());
        assert!(!r["stale"].is_null());
    }

    #[tokio::test]
    async fn test_search_includes_signature_by_default() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate", "compact": false}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["code_results"].as_array().unwrap();

        // At least one result with a function kind should have a signature
        let has_sig = results
            .iter()
            .any(|r| r["kind"] == "function" && r.get("signature").is_some());
        assert!(has_sig, "non-compact results should include signatures");
    }

    #[tokio::test]
    async fn test_search_compact_omits_signature() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate", "compact": true}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["meta"]["compact"], true);

        let results = data["code_results"].as_array().unwrap();
        for r in results {
            assert!(
                r.get("signature").is_none(),
                "compact mode should omit signatures"
            );
        }
    }

    #[tokio::test]
    async fn test_search_scope_code_filters_docs() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "README", "scope": "code"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["code_results"].as_array().unwrap();

        // No doc kinds should appear with scope=code
        for r in results {
            let kind = r["kind"].as_str().unwrap();
            assert!(
                !matches!(kind, "doc_section" | "readme" | "comment_block"),
                "code scope should not include doc kinds, got: {kind}"
            );
        }
    }

    #[tokio::test]
    async fn test_search_scope_docs_filters_code() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "README docs", "scope": "docs"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["code_results"].as_array().unwrap();

        // No code kinds should appear with scope=docs
        for r in results {
            let kind = r["kind"].as_str().unwrap();
            assert!(
                matches!(kind, "doc_section" | "readme" | "comment_block"),
                "docs scope should only include doc kinds, got: {kind}"
            );
        }
    }

    #[tokio::test]
    async fn test_search_invalid_scope_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "test", "scope": "invalid_scope"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Invalid scope"));
    }

    #[tokio::test]
    async fn test_search_custom_top_k() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "function", "top_k": 2}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["code_results"].as_array().unwrap();
        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn test_search_default_scope_is_all() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "README calculate"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["code_results"].as_array().unwrap();

        // With scope=all (default), we should get both code and doc results
        let kinds: Vec<&str> = results
            .iter()
            .map(|r| r["kind"].as_str().unwrap())
            .collect();
        let has_code = kinds.iter().any(|k| *k == "function");
        let has_docs = kinds.iter().any(|k| *k == "readme");
        assert!(
            has_code || has_docs,
            "default scope should include results from any kind"
        );
    }

    #[tokio::test]
    async fn test_search_code_count_in_meta() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let code_count = data["meta"]["code_count"].as_u64().unwrap();
        let results = data["code_results"].as_array().unwrap();

        // code_count should equal number of code-kind results
        let actual_code = results
            .iter()
            .filter(|r| {
                let kind = r["kind"].as_str().unwrap();
                !matches!(kind, "doc_section" | "readme" | "comment_block")
            })
            .count() as u64;
        assert_eq!(code_count, actual_code);
    }

    // --- Sidecar knowledge search tests ---

    #[tokio::test]
    async fn test_search_all_includes_knowledge_results() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "decision hnsw vector"}})),
        );
        let responses = run_server_with_knowledge_engine(&input).await;
        let data = parse_tool_text(&responses[0]);

        assert!(data["code_results"].is_array());
        assert!(data["knowledge_results"].is_array());
        assert!(data["meta"]["knowledge_count"].is_number());
    }

    #[tokio::test]
    async fn test_search_scope_knowledge_returns_only_knowledge() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "decision hnsw", "scope": "knowledge"}})),
        );
        let responses = run_server_with_knowledge_engine(&input).await;
        let data = parse_tool_text(&responses[0]);

        // code_results should be empty when scope is knowledge
        let code = data["code_results"].as_array().unwrap();
        assert!(code.is_empty(), "knowledge scope should not return code results");

        // knowledge_results should have results
        let knowledge = data["knowledge_results"].as_array().unwrap();
        assert!(!knowledge.is_empty(), "knowledge scope should return knowledge results");
    }

    #[tokio::test]
    async fn test_search_scope_code_excludes_knowledge() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "decision", "scope": "code"}})),
        );
        let responses = run_server_with_knowledge_engine(&input).await;
        let data = parse_tool_text(&responses[0]);

        // knowledge_results should be empty when scope is code
        let knowledge = data["knowledge_results"].as_array().unwrap();
        assert!(knowledge.is_empty(), "code scope should not return knowledge results");
    }

    #[tokio::test]
    async fn test_knowledge_results_have_required_fields() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "decision hnsw", "scope": "knowledge"}})),
        );
        let responses = run_server_with_knowledge_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let knowledge = data["knowledge_results"].as_array().unwrap();
        assert!(!knowledge.is_empty());

        let kr = &knowledge[0];
        assert!(kr["kind"].is_string(), "knowledge result must have kind");
        assert!(kr["id"].is_string(), "knowledge result must have id");
        assert!(kr["title"].is_string(), "knowledge result must have title");
        assert!(kr["relevance_score"].is_number(), "knowledge result must have relevance_score");
        assert!(kr["created_at"].is_string(), "knowledge result must have created_at");
    }

    #[tokio::test]
    async fn test_knowledge_result_kind_and_id_parsed() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "decision hnsw", "scope": "knowledge"}})),
        );
        let responses = run_server_with_knowledge_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let knowledge = data["knowledge_results"].as_array().unwrap();
        assert!(!knowledge.is_empty());

        let kr = &knowledge[0];
        assert_eq!(kr["kind"].as_str().unwrap(), "decision");
        assert_eq!(kr["id"].as_str().unwrap(), "DEC-001");
    }

    #[tokio::test]
    async fn test_knowledge_recency_boost_applied() {
        // The test engine creates a knowledge item with created_at = now (< 30 days)
        // So the recency boost should be applied (score * 1.1)
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "decision hnsw", "scope": "knowledge"}})),
        );
        let responses = run_server_with_knowledge_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let knowledge = data["knowledge_results"].as_array().unwrap();
        assert!(!knowledge.is_empty());

        // Score should exist and be positive (boosted)
        let score = knowledge[0]["relevance_score"].as_f64().unwrap();
        assert!(score > 0.0, "knowledge result should have positive score");
    }

    #[tokio::test]
    async fn test_search_scope_docs_excludes_knowledge() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "decision", "scope": "docs"}})),
        );
        let responses = run_server_with_knowledge_engine(&input).await;
        let data = parse_tool_text(&responses[0]);

        let knowledge = data["knowledge_results"].as_array().unwrap();
        assert!(knowledge.is_empty(), "docs scope should not return knowledge results");
    }

    #[tokio::test]
    async fn test_search_knowledge_count_in_meta() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "decision hnsw"}})),
        );
        let responses = run_server_with_knowledge_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let knowledge = data["knowledge_results"].as_array().unwrap();
        let knowledge_count = data["meta"]["knowledge_count"].as_u64().unwrap();
        assert_eq!(knowledge_count, knowledge.len() as u64);
    }

    // --- Helper function unit tests ---

    #[test]
    fn test_truncate_name_short() {
        assert_eq!(truncate_name("hello", 50), "hello");
    }

    #[test]
    fn test_truncate_name_long() {
        let long_name = "a".repeat(60);
        let truncated = truncate_name(&long_name, 50);
        assert_eq!(truncated.len(), 50);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_truncate_name_exact() {
        let name = "a".repeat(50);
        assert_eq!(truncate_name(&name, 50), name);
    }

    #[test]
    fn test_is_code_kind() {
        assert!(is_code_kind(&ChunkKind::Function));
        assert!(is_code_kind(&ChunkKind::Class));
        assert!(is_code_kind(&ChunkKind::Method));
        assert!(is_code_kind(&ChunkKind::Type));
        assert!(is_code_kind(&ChunkKind::Impl));
        assert!(is_code_kind(&ChunkKind::Module));
        assert!(is_code_kind(&ChunkKind::Other));
        assert!(!is_code_kind(&ChunkKind::DocSection));
        assert!(!is_code_kind(&ChunkKind::Readme));
        assert!(!is_code_kind(&ChunkKind::CommentBlock));
    }

    #[test]
    fn test_matches_scope_code() {
        assert!(matches_scope(&ChunkKind::Function, "code"));
        assert!(!matches_scope(&ChunkKind::Readme, "code"));
    }

    #[test]
    fn test_matches_scope_docs() {
        assert!(matches_scope(&ChunkKind::DocSection, "docs"));
        assert!(matches_scope(&ChunkKind::Readme, "docs"));
        assert!(!matches_scope(&ChunkKind::Function, "docs"));
    }

    #[test]
    fn test_matches_scope_knowledge() {
        assert!(matches_scope(&ChunkKind::Knowledge, "knowledge"));
        assert!(!matches_scope(&ChunkKind::Function, "knowledge"));
        assert!(!matches_scope(&ChunkKind::Readme, "knowledge"));
    }

    #[test]
    fn test_matches_scope_all() {
        assert!(matches_scope(&ChunkKind::Function, "all"));
        assert!(matches_scope(&ChunkKind::Readme, "all"));
        assert!(matches_scope(&ChunkKind::DocSection, "all"));
        assert!(matches_scope(&ChunkKind::Knowledge, "all"));
    }

    // --- engram_lookup tool tests ---

    #[tokio::test]
    async fn test_lookup_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "foo"}})),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not initialized"));
    }

    #[tokio::test]
    async fn test_lookup_missing_identifier_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("identifier"));
    }

    #[tokio::test]
    async fn test_lookup_by_chunk_id() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "repo#src/math.rs#calculate_total"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], "calculate_total");
        assert_eq!(results[0]["chunk_id"], "repo#src/math.rs#calculate_total");
        assert_eq!(data["meta"]["count"], 1);
    }

    #[tokio::test]
    async fn test_lookup_by_file_path() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "src/math.rs"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["file"], "src/math.rs");
    }

    #[tokio::test]
    async fn test_lookup_by_symbol_name() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "calculate_total"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], "calculate_total");
    }

    #[tokio::test]
    async fn test_lookup_no_matches_returns_empty_array() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "nonexistent_symbol"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert!(results.is_empty());
        assert_eq!(data["meta"]["count"], 0);
    }

    #[tokio::test]
    async fn test_lookup_results_have_required_fields() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "repo#src/ui.rs#render_button"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);

        let r = &results[0];
        assert!(r["chunk_id"].is_string());
        assert!(r["kind"].is_string());
        assert!(r["name"].is_string());
        assert!(r["file"].is_string());
        assert!(r["repo"].is_string());
        assert!(r["lines"].is_array());
        assert!(!r["stale"].is_null());
    }

    #[tokio::test]
    async fn test_lookup_includes_signature() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "repo#src/math.rs#calculate_total"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert!(results[0]["signature"].is_string());
    }

    #[tokio::test]
    async fn test_lookup_by_file_with_dot() {
        // A file path like "README.md" contains a dot, so it should be detected as a file path
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "README.md"}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["file"], "README.md");
    }

    #[tokio::test]
    async fn test_lookup_include_content_without_source_roots() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "calculate_total", "include_content": true}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        // Content should be null since no source_roots configured
        assert!(results[0]["content"].is_null());
    }

    #[tokio::test]
    async fn test_lookup_include_content_with_source_roots() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let src_dir = tmp_dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(src_dir.join("math.rs"), "fn calculate_total() { 42 }").unwrap();

        let (search, provider) = build_test_engine();
        let mut source_roots = HashMap::new();
        source_roots.insert("test-repo".to_string(), tmp_dir.path().to_path_buf());
        let server = McpServer::with_engine_and_sources(search, provider, source_roots);

        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": "calculate_total", "include_content": true}})),
        );

        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        let responses: Vec<JsonRpcResponse> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0]["content"].as_str().unwrap(),
            "fn calculate_total() { 42 }"
        );
    }

    #[tokio::test]
    async fn test_lookup_empty_identifier_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_lookup", "arguments": {"identifier": ""}})),
        );
        let responses = run_server_with_engine(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
    }

    // --- engram_status tool tests ---

    #[tokio::test]
    async fn test_status_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not initialized"));
    }

    #[tokio::test]
    async fn test_status_returns_total_chunks() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["total_chunks"], 4);
    }

    #[tokio::test]
    async fn test_status_returns_source_repos() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let repos = data["source_repos"].as_array().unwrap();
        assert_eq!(repos.len(), 1);
        assert_eq!(repos[0]["name"], "test-repo");
        assert_eq!(repos[0]["chunk_count"], 4);
        assert_eq!(repos[0]["stale_chunks"], 0);
    }

    #[tokio::test]
    async fn test_status_returns_embedding_provider() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["embedding_provider"], "mock/test");
    }

    #[tokio::test]
    async fn test_status_returns_boot_time_ms() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        assert!(data["boot_time_ms"].is_number());
    }

    #[tokio::test]
    async fn test_status_returns_cache_status() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        assert!(data["cache_status"].is_string());
    }

    #[tokio::test]
    async fn test_status_returns_store_path() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        assert!(data["store_path"].is_string());
    }

    #[tokio::test]
    async fn test_status_has_all_required_fields() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        assert!(data["total_chunks"].is_number());
        assert!(data["source_repos"].is_array());
        assert!(data["cache_status"].is_string());
        assert!(data["boot_time_ms"].is_number());
        assert!(data["embedding_provider"].is_string());
        assert!(data["store_path"].is_string());
    }

    #[tokio::test]
    async fn test_status_no_params_required() {
        // engram_status should work with no arguments at all
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status"})),
        );
        let responses = run_server_with_engine(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);
    }

    // --- engram_record_decision tool tests ---

    fn build_test_engine_with_store() -> (HybridSearch, Box<dyn EmbeddingProvider>, tempfile::TempDir)
    {
        let (search, provider) = build_test_engine();
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path();

        // Initialize a git repo at the store path
        let repo = git2::Repository::init(store_path).unwrap();
        {
            let mut index = repo.index().unwrap();
            let tree_oid = index.write_tree().unwrap();
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = git2::Signature::now("test", "test@test").unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "initial", &tree, &[])
                .unwrap();
        }

        // Create knowledge directories
        std::fs::create_dir_all(store_path.join("knowledge/decisions")).unwrap();
        std::fs::create_dir_all(store_path.join("knowledge/lessons")).unwrap();
        std::fs::create_dir_all(store_path.join("knowledge/patterns")).unwrap();
        std::fs::create_dir_all(store_path.join("knowledge/glossary")).unwrap();

        (search, provider, tmp)
    }

    async fn run_server_with_store(
        input: &str,
        store_path: &str,
    ) -> Vec<JsonRpcResponse> {
        let (search, provider) = build_test_engine();
        let mut server = McpServer::with_engine(search, provider);
        server.set_boot_info(0, store_path.to_string(), "none".to_string());
        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn test_record_decision_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_decision",
                "arguments": {
                    "title": "Test",
                    "context": "Test context",
                    "decision": "Test decision"
                }
            })),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not initialized"));
    }

    #[tokio::test]
    async fn test_record_decision_missing_title_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_decision",
                "arguments": {
                    "context": "Test context",
                    "decision": "Test decision"
                }
            })),
        );
        let responses =
            run_server_with_store(&input, tmp.path().to_str().unwrap()).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("title"));
    }

    #[tokio::test]
    async fn test_record_decision_missing_context_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_decision",
                "arguments": {
                    "title": "Test",
                    "decision": "Test decision"
                }
            })),
        );
        let responses =
            run_server_with_store(&input, tmp.path().to_str().unwrap()).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("context"));
    }

    #[tokio::test]
    async fn test_record_decision_missing_decision_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_decision",
                "arguments": {
                    "title": "Test",
                    "context": "Test context"
                }
            })),
        );
        let responses =
            run_server_with_store(&input, tmp.path().to_str().unwrap()).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("decision"));
    }

    #[tokio::test]
    async fn test_record_decision_writes_and_commits() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_decision",
                "arguments": {
                    "title": "Use PostgreSQL",
                    "context": "Need a relational database",
                    "decision": "Use PostgreSQL for persistence"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        assert!(data["id"].as_str().unwrap().starts_with("DEC-"));
        assert!(data["path"].as_str().unwrap().contains("knowledge/decisions/"));
        assert_eq!(data["committed"], true);

        // Verify the YAML file exists on disk
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        assert!(yaml_path.exists());

        // Verify the decision was committed to git
        let repo = git2::Repository::open(tmp.path()).unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        let msg = head.message().unwrap();
        assert!(msg.contains("knowledge: record decision"));
    }

    #[tokio::test]
    async fn test_record_decision_with_optional_params() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_decision",
                "arguments": {
                    "title": "Use HNSW",
                    "context": "Need vector search",
                    "decision": "Use HNSW for ANN",
                    "consequences": ["Fast queries", "More memory"],
                    "related_files": ["src/index.rs"],
                    "status": "proposed"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        let content = std::fs::read_to_string(yaml_path).unwrap();
        let decision: engram_core::Decision = serde_yaml::from_str(&content).unwrap();
        assert_eq!(decision.status, "proposed");
        assert_eq!(decision.consequences, vec!["Fast queries", "More memory"]);
        assert_eq!(decision.related_files, vec!["src/index.rs"]);
    }

    #[tokio::test]
    async fn test_record_decision_default_status_is_accepted() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_decision",
                "arguments": {
                    "title": "Default status test",
                    "context": "Testing defaults",
                    "decision": "Should default to accepted"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let data = parse_tool_text(&responses[0]);
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        let content = std::fs::read_to_string(yaml_path).unwrap();
        let decision: engram_core::Decision = serde_yaml::from_str(&content).unwrap();
        assert_eq!(decision.status, "accepted");
    }

    #[tokio::test]
    async fn test_record_decision_auto_generates_fields() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_decision",
                "arguments": {
                    "title": "Auto fields test",
                    "context": "Testing auto generation",
                    "decision": "Fields should be auto-generated"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let data = parse_tool_text(&responses[0]);
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        let content = std::fs::read_to_string(yaml_path).unwrap();
        let decision: engram_core::Decision = serde_yaml::from_str(&content).unwrap();

        assert!(decision.id.starts_with("DEC-"));
        assert_eq!(decision.contributed_by, "mcp-agent");
        assert!(!decision.created_at.is_empty());
        // embedding_ref should be set after write_decision_with_embedding
        assert!(decision.embedding_ref.is_some());
    }

    #[tokio::test]
    async fn test_record_decision_returns_json_format() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_decision",
                "arguments": {
                    "title": "JSON format test",
                    "context": "Testing response format",
                    "decision": "Response should have id, path, committed"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let data = parse_tool_text(&responses[0]);

        // Verify all required response fields exist
        assert!(data["id"].is_string());
        assert!(data["path"].is_string());
        assert!(data["committed"].is_boolean());
    }

    // --- engram_record_lesson tool tests ---

    #[tokio::test]
    async fn test_record_lesson_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_lesson",
                "arguments": {
                    "title": "Test",
                    "description": "Test description",
                    "trigger": "Test trigger"
                }
            })),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not initialized"));
    }

    #[tokio::test]
    async fn test_record_lesson_missing_title_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_lesson",
                "arguments": {
                    "description": "Test description",
                    "trigger": "Test trigger"
                }
            })),
        );
        let responses =
            run_server_with_store(&input, tmp.path().to_str().unwrap()).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("title"));
    }

    #[tokio::test]
    async fn test_record_lesson_missing_description_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_lesson",
                "arguments": {
                    "title": "Test",
                    "trigger": "Test trigger"
                }
            })),
        );
        let responses =
            run_server_with_store(&input, tmp.path().to_str().unwrap()).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("description"));
    }

    #[tokio::test]
    async fn test_record_lesson_missing_trigger_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_lesson",
                "arguments": {
                    "title": "Test",
                    "description": "Test description"
                }
            })),
        );
        let responses =
            run_server_with_store(&input, tmp.path().to_str().unwrap()).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("trigger"));
    }

    #[tokio::test]
    async fn test_record_lesson_writes_and_commits() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_lesson",
                "arguments": {
                    "title": "Scope git2 borrows",
                    "description": "git2 Repository borrows must be scoped before moving",
                    "trigger": "Borrow checker error"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        assert!(data["id"].as_str().unwrap().starts_with("LES-"));
        assert!(data["path"].as_str().unwrap().contains("knowledge/lessons/"));
        assert_eq!(data["committed"], true);

        // Verify the YAML file exists on disk
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        assert!(yaml_path.exists());

        // Verify the lesson was committed to git
        let repo = git2::Repository::open(tmp.path()).unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        let msg = head.message().unwrap();
        assert!(msg.contains("knowledge: record lesson"));
    }

    #[tokio::test]
    async fn test_record_lesson_with_optional_params() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_lesson",
                "arguments": {
                    "title": "Always check errors",
                    "description": "Error handling is important",
                    "trigger": "Silent failure in production",
                    "resolution": "Add error checks everywhere",
                    "related_files": ["src/main.rs", "src/lib.rs"]
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        let content = std::fs::read_to_string(yaml_path).unwrap();
        let lesson: engram_core::Lesson = serde_yaml::from_str(&content).unwrap();
        assert_eq!(lesson.resolution, "Add error checks everywhere");
        assert_eq!(lesson.related_files, vec!["src/main.rs", "src/lib.rs"]);
    }

    #[tokio::test]
    async fn test_record_lesson_auto_generates_fields() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_lesson",
                "arguments": {
                    "title": "Auto fields test",
                    "description": "Testing auto generation",
                    "trigger": "Need to verify auto fields"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let data = parse_tool_text(&responses[0]);
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        let content = std::fs::read_to_string(yaml_path).unwrap();
        let lesson: engram_core::Lesson = serde_yaml::from_str(&content).unwrap();

        assert!(lesson.id.starts_with("LES-"));
        assert_eq!(lesson.contributed_by, "mcp-agent");
        assert!(!lesson.created_at.is_empty());
        assert!(lesson.embedding_ref.is_some());
    }

    #[tokio::test]
    async fn test_record_lesson_returns_json_format() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_lesson",
                "arguments": {
                    "title": "JSON format test",
                    "description": "Testing response format",
                    "trigger": "Need to verify JSON response"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let data = parse_tool_text(&responses[0]);

        // Verify all required response fields exist
        assert!(data["id"].is_string());
        assert!(data["path"].is_string());
        assert!(data["committed"].is_boolean());
    }

    #[tokio::test]
    async fn test_status_stale_chunks_counted() {
        let dims = 32;
        let v0 = make_vector(dims, 0.0);
        let v1 = make_vector(dims, 1.0);
        let entries: Vec<(u64, &[f32])> = vec![(0, &v0), (1, &v1)];
        let hnsw = HnswIndex::build(&entries, dims).unwrap();
        let docs = vec![
            make_doc(0, "func_a", Some("fn func_a()"), &[]),
            make_doc(1, "func_b", Some("fn func_b()"), &[]),
        ];
        let bm25 = Bm25Index::build(&docs);

        let mut metadata = HashMap::new();
        metadata.insert(0, ChunkEntry {
            chunk_id: "repo#a.rs#func_a".to_string(),
            kind: ChunkKind::Function,
            name: "func_a".to_string(),
            signature: Some("fn func_a()".to_string()),
            file: "a.rs".to_string(),
            repo: "repo".to_string(),
            start_line: 1,
            end_line: 5,
            stale: true,
            tags: Vec::new(),
            indexed_at: String::new(),
        });
        metadata.insert(1, ChunkEntry {
            chunk_id: "repo#b.rs#func_b".to_string(),
            kind: ChunkKind::Function,
            name: "func_b".to_string(),
            signature: Some("fn func_b()".to_string()),
            file: "b.rs".to_string(),
            repo: "repo".to_string(),
            start_line: 1,
            end_line: 5,
            stale: false,
            tags: Vec::new(),
            indexed_at: String::new(),
        });

        let search = HybridSearch::new(hnsw, bm25, metadata);
        let provider: Box<dyn EmbeddingProvider> = Box::new(MockEmbeddingProvider { dims });
        let server = McpServer::with_engine(search, provider);

        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_status", "arguments": {}})),
        );
        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        let responses: Vec<JsonRpcResponse> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["total_chunks"], 2);
        let repos = data["source_repos"].as_array().unwrap();
        assert_eq!(repos[0]["stale_chunks"], 1);
    }

    // --- engram_record_pattern tool tests ---

    #[tokio::test]
    async fn test_record_pattern_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_pattern",
                "arguments": {
                    "name": "Test",
                    "description": "Test description"
                }
            })),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not initialized"));
    }

    #[tokio::test]
    async fn test_record_pattern_missing_name_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_pattern",
                "arguments": {
                    "description": "Test description"
                }
            })),
        );
        let responses =
            run_server_with_store(&input, tmp.path().to_str().unwrap()).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("name"));
    }

    #[tokio::test]
    async fn test_record_pattern_missing_description_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_pattern",
                "arguments": {
                    "name": "Test"
                }
            })),
        );
        let responses =
            run_server_with_store(&input, tmp.path().to_str().unwrap()).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("description"));
    }

    #[tokio::test]
    async fn test_record_pattern_writes_and_commits() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_pattern",
                "arguments": {
                    "name": "One module per concept",
                    "description": "Each concept gets its own module file"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        assert!(data["id"].as_str().unwrap().starts_with("PAT-"));
        assert!(data["path"].as_str().unwrap().contains("knowledge/patterns/"));
        assert_eq!(data["committed"], true);

        // Verify the YAML file exists on disk
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        assert!(yaml_path.exists());

        // Verify the pattern was committed to git
        let repo = git2::Repository::open(tmp.path()).unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        let msg = head.message().unwrap();
        assert!(msg.contains("knowledge: record pattern"));
    }

    #[tokio::test]
    async fn test_record_pattern_with_optional_params() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_pattern",
                "arguments": {
                    "name": "Builder pattern",
                    "description": "Use builder for complex construction",
                    "examples": ["Config::builder().port(8080).build()", "Query::new().filter(f).limit(10)"],
                    "anti_patterns": ["Telescoping constructors", "Too many optional params"]
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        let content = std::fs::read_to_string(yaml_path).unwrap();
        let pattern: engram_core::Pattern = serde_yaml::from_str(&content).unwrap();
        assert_eq!(pattern.examples.len(), 2);
        assert_eq!(pattern.anti_patterns.len(), 2);
        assert_eq!(pattern.anti_patterns[0], "Telescoping constructors");
    }

    #[tokio::test]
    async fn test_record_pattern_auto_generates_fields() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_pattern",
                "arguments": {
                    "name": "Auto fields test",
                    "description": "Testing auto generation"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let data = parse_tool_text(&responses[0]);
        let yaml_path = tmp.path().join(data["path"].as_str().unwrap());
        let content = std::fs::read_to_string(yaml_path).unwrap();
        let pattern: engram_core::Pattern = serde_yaml::from_str(&content).unwrap();

        assert!(pattern.id.starts_with("PAT-"));
        assert_eq!(pattern.contributed_by, "mcp-agent");
        assert!(!pattern.created_at.is_empty());
        assert!(pattern.embedding_ref.is_some());
    }

    #[tokio::test]
    async fn test_record_pattern_returns_json_format() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_pattern",
                "arguments": {
                    "name": "JSON format test",
                    "description": "Testing response format"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let data = parse_tool_text(&responses[0]);

        // Verify all required response fields exist
        assert!(data["id"].is_string());
        assert!(data["path"].is_string());
        assert!(data["committed"].is_boolean());
    }

    // --- engram_record_glossary tool tests ---

    #[tokio::test]
    async fn test_record_glossary_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_glossary",
                "arguments": {
                    "term": "chunk",
                    "definition": "A semantic unit of code"
                }
            })),
        );
        let responses = run_server(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
    }

    #[tokio::test]
    async fn test_record_glossary_missing_term_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_glossary",
                "arguments": {
                    "definition": "A semantic unit of code"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
    }

    #[tokio::test]
    async fn test_record_glossary_missing_definition_returns_error() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_glossary",
                "arguments": {
                    "term": "chunk"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
    }

    #[tokio::test]
    async fn test_record_glossary_creates_new_term() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_glossary",
                "arguments": {
                    "term": "chunk",
                    "definition": "A semantic unit of code"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["term"], "chunk");
        assert_eq!(data["action"], "created");
        assert_eq!(data["committed"], true);

        // Verify terms.yaml exists
        let terms_path = tmp.path().join("knowledge/glossary/terms.yaml");
        assert!(terms_path.exists());

        // Verify committed to git
        let repo = git2::Repository::open(tmp.path()).unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        let msg = head.message().unwrap();
        assert!(msg.contains("knowledge: record glossary term chunk"));
    }

    #[tokio::test]
    async fn test_record_glossary_updates_existing_term() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();

        // Create the term first
        let input1 = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_glossary",
                "arguments": {
                    "term": "chunk",
                    "definition": "Original definition"
                }
            })),
        );
        run_server_with_store(&input1, store_path).await;

        // Update the term
        let input2 = make_request(
            2,
            "tools/call",
            Some(json!({
                "name": "engram_record_glossary",
                "arguments": {
                    "term": "chunk",
                    "definition": "Updated definition"
                }
            })),
        );
        let responses = run_server_with_store(&input2, store_path).await;
        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["term"], "chunk");
        assert_eq!(data["action"], "updated");
        assert_eq!(data["committed"], true);

        // Verify only one entry exists with updated definition
        let content =
            std::fs::read_to_string(tmp.path().join("knowledge/glossary/terms.yaml")).unwrap();
        let entries: Vec<engram_core::GlossaryEntry> = serde_yaml::from_str(&content).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].definition, "Updated definition");
    }

    #[tokio::test]
    async fn test_record_glossary_with_optional_context() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_glossary",
                "arguments": {
                    "term": "HNSW",
                    "definition": "Hierarchical Navigable Small World graph",
                    "context": "Used in engram-query for vector similarity search"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let content =
            std::fs::read_to_string(tmp.path().join("knowledge/glossary/terms.yaml")).unwrap();
        let entries: Vec<engram_core::GlossaryEntry> = serde_yaml::from_str(&content).unwrap();
        assert_eq!(entries[0].context, "Used in engram-query for vector similarity search");
    }

    #[tokio::test]
    async fn test_record_glossary_auto_generates_fields() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_glossary",
                "arguments": {
                    "term": "BM25",
                    "definition": "Best Matching 25 scoring function"
                }
            })),
        );
        run_server_with_store(&input, store_path).await;

        let content =
            std::fs::read_to_string(tmp.path().join("knowledge/glossary/terms.yaml")).unwrap();
        let entries: Vec<engram_core::GlossaryEntry> = serde_yaml::from_str(&content).unwrap();
        assert_eq!(entries[0].contributed_by, "mcp-agent");
        assert!(!entries[0].created_at.is_empty());
    }

    #[tokio::test]
    async fn test_record_glossary_returns_json_format() {
        let (_search, _provider, tmp) = build_test_engine_with_store();
        let store_path = tmp.path().to_str().unwrap();
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_record_glossary",
                "arguments": {
                    "term": "embedding",
                    "definition": "A dense vector representation"
                }
            })),
        );
        let responses = run_server_with_store(&input, store_path).await;
        let data = parse_tool_text(&responses[0]);

        // Verify all required response fields exist
        assert!(data["term"].is_string());
        assert!(data["action"].is_string());
        assert!(data["committed"].is_boolean());
    }

    // --- engram_onboard tests ---

    #[tokio::test]
    async fn test_onboard_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_onboard", "arguments": {}})),
        );
        let responses = run_server(& input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Engine not initialized"));
    }

    #[tokio::test]
    async fn test_onboard_no_source_roots_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_onboard", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("No source roots configured"));
    }

    #[tokio::test]
    async fn test_onboard_unknown_repo_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let (search, provider) = build_test_engine();
        let mut source_roots = HashMap::new();
        source_roots.insert("my-repo".to_string(), tmp.path().to_path_buf());
        let server = McpServer::with_engine_and_sources(search, provider, source_roots);

        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_onboard", "arguments": {"repo": "nonexistent"}})),
        );
        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        let responses: Vec<JsonRpcResponse> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Unknown repo: nonexistent"));
    }

    #[tokio::test]
    async fn test_onboard_invalid_depth_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let (search, provider) = build_test_engine();
        let mut source_roots = HashMap::new();
        source_roots.insert("my-repo".to_string(), tmp.path().to_path_buf());
        let server = McpServer::with_engine_and_sources(search, provider, source_roots);

        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_onboard", "arguments": {"depth": "ultra"}})),
        );
        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        let responses: Vec<JsonRpcResponse> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Invalid depth"));
    }

    #[tokio::test]
    async fn test_onboard_success_with_quick_depth() {
        // Create a fake repo with a Cargo.toml so metadata detection works
        let repo_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            repo_dir.path().join("Cargo.toml"),
            "[package]\nname = \"test-project\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        ).unwrap();
        std::fs::create_dir_all(repo_dir.path().join("src")).unwrap();
        std::fs::write(repo_dir.path().join("src/main.rs"), "fn main() {}").unwrap();

        // Create a git-backed store
        let store_dir = tempfile::tempdir().unwrap();
        git2::Repository::init(store_dir.path()).unwrap();

        let (search, provider) = build_test_engine();
        let mut source_roots = HashMap::new();
        source_roots.insert("test-repo".to_string(), repo_dir.path().to_path_buf());
        let mut server = McpServer::with_engine_and_sources(search, provider, source_roots);
        server.set_boot_info(0, store_dir.path().to_string_lossy().to_string(), "none".to_string());

        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_onboard", "arguments": {"depth": "quick"}})),
        );
        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        let responses: Vec<JsonRpcResponse> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["depth"], "quick");
        assert_eq!(data["metadata_detected"], true);
        assert_eq!(data["commands_extracted"], true);
        assert_eq!(data["architecture_analyzed"], false);
        assert_eq!(data["abstractions_extracted"], false);
        assert!(data["files_written"].as_array().unwrap().len() >= 2);
    }

    #[tokio::test]
    async fn test_onboard_tools_listed() {
        let input = make_request(1, "tools/list", None);
        let responses = run_server(&input).await;
        let tools = responses[0].result.as_ref().unwrap()["tools"]
            .as_array()
            .unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"engram_onboard"));
    }

    // --- engram_assess_context tests ---

    #[tokio::test]
    async fn test_assess_context_no_engine() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_assess_context", "arguments": {}})),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("not initialized"));
    }

    #[tokio::test]
    async fn test_assess_context_empty_session() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_assess_context", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["session_summary"]["search_count"], 0);
        assert!(data["session_summary"]["unique_queries"]
            .as_array()
            .unwrap()
            .is_empty());
        assert_eq!(data["session_summary"]["total_code_results"], 0);
        assert_eq!(data["session_summary"]["total_knowledge_results"], 0);
        assert!(data["searches"].as_array().unwrap().is_empty());
        assert!(data["evaluation_prompt"]
            .as_str()
            .unwrap()
            .contains("0 search(es)"));
    }

    #[tokio::test]
    async fn test_assess_context_after_searches() {
        // Send two search requests then an assess_context request through the same server
        let search1 = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate total", "scope": "code"}})),
        );
        let search2 = make_request(
            2,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "render button", "scope": "all"}})),
        );
        let assess = make_request(
            3,
            "tools/call",
            Some(json!({"name": "engram_assess_context", "arguments": {}})),
        );
        let input = format!("{search1}\n{search2}\n{assess}");
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 3);

        let data = parse_tool_text(&responses[2]);
        assert_eq!(data["session_summary"]["search_count"], 2);
        let unique_queries = data["session_summary"]["unique_queries"]
            .as_array()
            .unwrap();
        assert_eq!(unique_queries.len(), 2);
        assert!(data["session_summary"]["total_code_results"].as_u64().unwrap() > 0);
        assert_eq!(data["searches"].as_array().unwrap().len(), 2);
        assert!(data["evaluation_prompt"]
            .as_str()
            .unwrap()
            .contains("2 search(es)"));
    }

    #[tokio::test]
    async fn test_assess_context_with_task_description() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_assess_context", "arguments": {"task_description": "Fix the auth bug in login flow"}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let data = parse_tool_text(&responses[0]);
        assert!(data["evaluation_prompt"]
            .as_str()
            .unwrap()
            .contains("Fix the auth bug in login flow"));
    }

    #[tokio::test]
    async fn test_assess_context_tracks_duplicate_queries() {
        // Same query twice should show 2 searches but 1 unique query
        let search1 = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate total"}})),
        );
        let search2 = make_request(
            2,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate total"}})),
        );
        let assess = make_request(
            3,
            "tools/call",
            Some(json!({"name": "engram_assess_context", "arguments": {}})),
        );
        let input = format!("{search1}\n{search2}\n{assess}");
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 3);

        let data = parse_tool_text(&responses[2]);
        assert_eq!(data["session_summary"]["search_count"], 2);
        assert_eq!(
            data["session_summary"]["unique_queries"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn test_assess_context_scope_distribution() {
        let search1 = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "foo", "scope": "code"}})),
        );
        let search2 = make_request(
            2,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "bar", "scope": "code"}})),
        );
        let search3 = make_request(
            3,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "baz", "scope": "docs"}})),
        );
        let assess = make_request(
            4,
            "tools/call",
            Some(json!({"name": "engram_assess_context", "arguments": {}})),
        );
        let input = format!("{search1}\n{search2}\n{search3}\n{assess}");
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 4);

        let data = parse_tool_text(&responses[3]);
        let scope_dist = &data["session_summary"]["scope_distribution"];
        assert_eq!(scope_dist["code"], 2);
        assert_eq!(scope_dist["docs"], 1);
    }

    #[tokio::test]
    async fn test_assess_context_listed_in_tools() {
        let input = make_request(1, "tools/list", None);
        let responses = run_server(&input).await;
        let tools = responses[0].result.as_ref().unwrap()["tools"]
            .as_array()
            .unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"engram_assess_context"));
    }

    // --- engram_check_staleness tool tests ---

    #[tokio::test]
    async fn test_check_staleness_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_check_staleness", "arguments": {}})),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not initialized"));
    }

    #[tokio::test]
    async fn test_check_staleness_no_searches_returns_empty() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_check_staleness", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], false);

        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["total_retrieved"], 0);
        assert_eq!(data["stale_count"], 0);
        assert_eq!(data["fresh_count"], 0);
        assert!(data["evaluation_prompt"].as_str().unwrap().contains("No chunks"));
    }

    #[tokio::test]
    async fn test_check_staleness_after_search_returns_chunks() {
        let search = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate"}})),
        );
        let staleness = make_request(
            2,
            "tools/call",
            Some(json!({"name": "engram_check_staleness", "arguments": {}})),
        );
        let input = format!("{search}\n{staleness}");
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 2);

        let data = parse_tool_text(&responses[1]);
        assert!(data["total_retrieved"].as_u64().unwrap() > 0);
        assert!(data["files"].as_array().unwrap().len() > 0);
        assert!(data["evaluation_prompt"].as_str().unwrap().contains("Staleness Check"));
        assert!(data["stale_percentage"].is_number());
    }

    #[tokio::test]
    async fn test_check_staleness_reports_fresh_chunks() {
        // Default test engine has stale=false on all chunks
        let search = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate"}})),
        );
        let staleness = make_request(
            2,
            "tools/call",
            Some(json!({"name": "engram_check_staleness", "arguments": {}})),
        );
        let input = format!("{search}\n{staleness}");
        let responses = run_server_with_engine(&input).await;

        let data = parse_tool_text(&responses[1]);
        assert_eq!(data["stale_count"], 0);
        assert!(data["fresh_count"].as_u64().unwrap() > 0);
        assert!(data["evaluation_prompt"].as_str().unwrap().contains("All retrieved chunks are fresh"));
    }

    #[tokio::test]
    async fn test_check_staleness_listed_in_tools() {
        let input = make_request(1, "tools/list", None);
        let responses = run_server(&input).await;
        let tools = responses[0].result.as_ref().unwrap()["tools"]
            .as_array()
            .unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"engram_check_staleness"));
    }

    #[tokio::test]
    async fn test_check_staleness_file_grouping() {
        // Two searches should accumulate chunks, grouped by file
        let search1 = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "calculate"}})),
        );
        let search2 = make_request(
            2,
            "tools/call",
            Some(json!({"name": "engram_search", "arguments": {"query": "config"}})),
        );
        let staleness = make_request(
            3,
            "tools/call",
            Some(json!({"name": "engram_check_staleness", "arguments": {}})),
        );
        let input = format!("{search1}\n{search2}\n{staleness}");
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 3);

        let data = parse_tool_text(&responses[2]);
        let files = data["files"].as_array().unwrap();
        // Each file entry should have the required fields
        for f in files {
            assert!(f["file"].is_string());
            assert!(f["total_chunks"].is_number());
            assert!(f["stale_chunks"].is_number());
            assert!(f["chunk_ids"].is_array());
            assert!(f["latest_indexed_at"].is_string());
        }
    }

    // --- engram_graph tests ---

    #[tokio::test]
    async fn test_graph_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_graph", "arguments": {"symbol": "Foo"}})),
        );
        let responses = run_server(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
    }

    #[tokio::test]
    async fn test_graph_missing_symbol_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_graph", "arguments": {}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("symbol"));
    }

    #[tokio::test]
    async fn test_graph_nonexistent_symbol_returns_empty() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_graph", "arguments": {"symbol": "NonexistentSymbol"}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["node_count"], 0);
        assert_eq!(data["edge_count"], 0);
        assert_eq!(data["symbol"], "NonexistentSymbol");
    }

    #[tokio::test]
    async fn test_graph_default_direction_is_both() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_graph", "arguments": {"symbol": "Foo"}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["direction"], "both");
    }

    #[tokio::test]
    async fn test_graph_default_depth_is_2() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_graph", "arguments": {"symbol": "Foo"}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["depth"], 2);
    }

    #[tokio::test]
    async fn test_graph_invalid_direction_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_graph", "arguments": {"symbol": "Foo", "direction": "invalid"}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let result = responses[0].result.as_ref().unwrap();
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("invalid"));
    }

    #[tokio::test]
    async fn test_graph_with_graph_data() {
        use engram_core::SymbolId;

        let (search, provider) = build_test_engine();
        let mut server = McpServer::with_engine(search, provider);

        // Build a graph with actual data
        let exports = vec![engram_core::ExportedSymbol {
            id: SymbolId {
                file: std::path::PathBuf::from("src/foo.ts"),
                name: "Foo".to_string(),
                kind: "class".to_string(),
            },
            line: 1,
            is_public: true,
            doc: None,
            chunk_id: Some("repo-a#src/foo.ts#Foo".to_string()),
        }];
        let imports = vec![engram_core::ResolvedImport {
            import_path: "import { Foo }".to_string(),
            resolved_symbol: SymbolId {
                file: std::path::PathBuf::from("src/foo.ts"),
                name: "Foo".to_string(),
                kind: "class".to_string(),
            },
            importing_file: std::path::PathBuf::from("src/main.ts"),
            line: 1,
            importing_repo: "repo-b".to_string(),
            source_repo: "repo-a".to_string(),
            source_file: std::path::PathBuf::from("src/foo.ts"),
            resolved_chunk: Some("repo-a#src/foo.ts#Foo".to_string()),
            resolution: "heuristic".to_string(),
        }];
        let graph = SymbolGraph::build(&exports, &imports);
        server.set_graph(graph);

        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_graph", "arguments": {"symbol": "Foo", "direction": "callers", "depth": 1}})),
        );
        let reader = tokio::io::BufReader::new(input.as_bytes());
        let mut output = Vec::new();
        server.run(reader, &mut output).await.unwrap();
        let output_str = String::from_utf8(output).unwrap();
        let responses: Vec<JsonRpcResponse> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(responses.len(), 1);
        let data = parse_tool_text(&responses[0]);
        assert_eq!(data["symbol"], "Foo");
        assert_eq!(data["direction"], "callers");
        assert_eq!(data["depth"], 1);

        let nodes = data["nodes"].as_array().unwrap();
        assert!(!nodes.is_empty(), "Should have nodes for Foo and its caller");

        let edges = data["edges"].as_array().unwrap();
        assert!(!edges.is_empty(), "Should have at least one edge");

        // Check edge structure
        let edge = &edges[0];
        assert!(edge["source"].is_object());
        assert!(edge["target"].is_object());
        assert!(edge["relationship"].is_string());
        assert_eq!(edge["relationship"], "cross_repo_import");

        // Nodes should have required fields
        for node in nodes {
            assert!(node["name"].is_string());
            assert!(node["file"].is_string());
            assert!(node["repo"].is_string());
        }
    }

    #[tokio::test]
    async fn test_graph_response_json_structure() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({"name": "engram_graph", "arguments": {"symbol": "anything"}})),
        );
        let responses = run_server_with_engine(&input).await;
        assert_eq!(responses.len(), 1);
        let data = parse_tool_text(&responses[0]);
        // All required fields should be present
        assert!(data["symbol"].is_string());
        assert!(data["direction"].is_string());
        assert!(data["depth"].is_number());
        assert!(data["nodes"].is_array());
        assert!(data["edges"].is_array());
        assert!(data["node_count"].is_number());
        assert!(data["edge_count"].is_number());
    }

    #[tokio::test]
    async fn test_graph_listed_in_tools() {
        let input = make_request(1, "tools/list", None);
        let responses = run_server(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        let tools = result["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"engram_graph"));
    }

    // --- engram_related tests ---

    #[tokio::test]
    async fn test_related_without_engine_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_related",
                "arguments": { "chunk_id": "repo#src/math.rs#calculate_total" }
            })),
        );
        let responses = run_server(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert!(result["isError"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_related_missing_both_params_returns_error() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_related",
                "arguments": {}
            })),
        );
        let responses = run_server_with_engine(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert!(result["isError"].as_bool().unwrap());
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("At least one of chunk_id or symbol is required"));
    }

    #[tokio::test]
    async fn test_related_by_chunk_id_returns_results() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_related",
                "arguments": { "chunk_id": "repo#src/math.rs#calculate_total" }
            })),
        );
        let responses = run_server_with_engine(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert!(!result["isError"].as_bool().unwrap_or(true));
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        // Should return other chunks, excluding the input chunk itself
        for r in results {
            assert_ne!(r["chunk_id"].as_str().unwrap(), "repo#src/math.rs#calculate_total");
        }
    }

    #[tokio::test]
    async fn test_related_by_symbol_returns_results() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_related",
                "arguments": { "symbol": "calculate_total" }
            })),
        );
        let responses = run_server_with_engine(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        assert!(!result["isError"].as_bool().unwrap_or(true));
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        // Should return related chunks, excluding the symbol's own chunks
        for r in results {
            assert_ne!(r["name"].as_str().unwrap(), "calculate_total");
        }
    }

    #[tokio::test]
    async fn test_related_nonexistent_chunk_id_returns_empty() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_related",
                "arguments": { "chunk_id": "nonexistent#chunk#id" }
            })),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_related_nonexistent_symbol_returns_empty() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_related",
                "arguments": { "symbol": "nonexistent_symbol" }
            })),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_related_respects_top_k() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_related",
                "arguments": { "chunk_id": "repo#src/math.rs#calculate_total", "top_k": 1 }
            })),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        assert!(results.len() <= 1);
    }

    #[tokio::test]
    async fn test_related_returns_code_result_format() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_related",
                "arguments": { "chunk_id": "repo#src/math.rs#calculate_total" }
            })),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        let results = data["results"].as_array().unwrap();
        if !results.is_empty() {
            let r = &results[0];
            // Verify it has the same format as engram_search code results
            assert!(r.get("chunk_id").is_some());
            assert!(r.get("score").is_some());
            assert!(r.get("kind").is_some());
            assert!(r.get("name").is_some());
            assert!(r.get("file").is_some());
            assert!(r.get("repo").is_some());
            assert!(r.get("lines").is_some());
            assert!(r.get("stale").is_some());
        }
    }

    #[tokio::test]
    async fn test_related_meta_fields() {
        let input = make_request(
            1,
            "tools/call",
            Some(json!({
                "name": "engram_related",
                "arguments": { "symbol": "render_button", "top_k": 5 }
            })),
        );
        let responses = run_server_with_engine(&input).await;
        let data = parse_tool_text(&responses[0]);
        assert!(data.get("meta").is_some());
        let meta = &data["meta"];
        assert!(meta.get("result_count").is_some());
        assert!(meta.get("top_k").is_some());
        assert_eq!(meta["top_k"].as_u64().unwrap(), 5);
    }

    #[tokio::test]
    async fn test_related_listed_in_tools() {
        let input = make_request(1, "tools/list", None);
        let responses = run_server(&input).await;
        let result = responses[0].result.as_ref().unwrap();
        let tools = result["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"engram_related"));
    }
}
