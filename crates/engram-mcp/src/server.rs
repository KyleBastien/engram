use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use engram_core::{ChunkKind, Decision, EmbeddingProvider, GlossaryEntry, Lesson, Pattern, Snapshot, SnapshotTier};
use engram_query::{ChunkEntry, HybridSearch, SearchResult, DEFAULT_ALPHA};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse, METHOD_NOT_FOUND, PARSE_ERROR};
use crate::tools::{knowledge_tool_definitions, phase1_tool_definitions};

const SERVER_NAME: &str = "engram";
const SERVER_VERSION: &str = "0.1.0";
const PROTOCOL_VERSION: &str = "2024-11-05";

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
            }),
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

    // Validate scope (Phase 1: code, docs, all only)
    if !matches!(scope, "code" | "docs" | "all") {
        return tool_error_response(
            request,
            &format!("Invalid scope: {scope}. Must be code, docs, or all"),
        );
    }

    let start = Instant::now();

    // Embed the query
    let embedding = match state.provider.embed(&[query]).await {
        Ok(mut vecs) if !vecs.is_empty() => vecs.remove(0),
        Ok(_) => return tool_error_response(request, "Embedding returned empty result"),
        Err(e) => return tool_error_response(request, &format!("Embedding failed: {e}")),
    };

    // Run hybrid search
    let results = match state
        .search
        .search(query, &embedding, top_k, DEFAULT_ALPHA)
        .await
    {
        Ok(r) => r,
        Err(e) => return tool_error_response(request, &format!("Search failed: {e}")),
    };

    // Filter by scope
    let results: Vec<&SearchResult> = results
        .iter()
        .filter(|r| matches_scope(&r.kind, scope))
        .collect();

    let code_count = results.iter().filter(|r| is_code_kind(&r.kind)).count();
    let search_time_ms = start.elapsed().as_millis() as u64;

    // Format results
    let formatted: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
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
        })
        .collect();

    let response_data = json!({
        "results": formatted,
        "meta": {
            "code_count": code_count,
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

fn matches_scope(kind: &ChunkKind, scope: &str) -> bool {
    match scope {
        "code" => is_code_kind(kind),
        "docs" => matches!(
            kind,
            ChunkKind::DocSection | ChunkKind::Readme | ChunkKind::CommentBlock
        ),
        _ => true, // "all" or any other value
    }
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
    use engram_query::{Bm25Document, Bm25Index, HnswIndex};

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

        let entries: Vec<(u64, &[f32])> = vec![(0, &v0), (1, &v1), (2, &v2), (3, &v3)];
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
            make_doc(3, "README", None, &["docs"]),
        ];
        let bm25 = Bm25Index::build(&docs);

        let metadata: HashMap<u64, ChunkEntry> = vec![
            make_entry(0, "calculate_total", "src/math.rs", ChunkKind::Function),
            make_entry(1, "render_button", "src/ui.rs", ChunkKind::Function),
            make_entry(2, "parse_config", "src/config.rs", ChunkKind::Function),
            make_entry(3, "README", "README.md", ChunkKind::Readme),
        ]
        .into_iter()
        .collect();

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

    fn parse_tool_text(response: &JsonRpcResponse) -> serde_json::Value {
        let result = response.result.as_ref().unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        serde_json::from_str(text).unwrap()
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
        assert_eq!(tools.len(), 8);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"engram_search"));
        assert!(names.contains(&"engram_lookup"));
        assert!(names.contains(&"engram_status"));
        assert!(names.contains(&"engram_record_decision"));
        assert!(names.contains(&"engram_record_lesson"));
        assert!(names.contains(&"engram_record_pattern"));
        assert!(names.contains(&"engram_record_glossary"));
        assert!(names.contains(&"engram_snapshot"));
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
        assert!(data["results"].is_array());
        assert!(data["meta"].is_object());
        assert!(data["meta"]["code_count"].is_number());
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
        let results = data["results"].as_array().unwrap();
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
        let results = data["results"].as_array().unwrap();

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

        let results = data["results"].as_array().unwrap();
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
        let results = data["results"].as_array().unwrap();

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
        let results = data["results"].as_array().unwrap();

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
            Some(json!({"name": "engram_search", "arguments": {"query": "test", "scope": "knowledge"}})),
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
        let results = data["results"].as_array().unwrap();
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
        let results = data["results"].as_array().unwrap();

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
        let results = data["results"].as_array().unwrap();

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
    fn test_matches_scope_all() {
        assert!(matches_scope(&ChunkKind::Function, "all"));
        assert!(matches_scope(&ChunkKind::Readme, "all"));
        assert!(matches_scope(&ChunkKind::DocSection, "all"));
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
}
