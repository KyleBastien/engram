use crate::lang::LangServerConfig;
use engram_core::EngramError;
use lsp_types::{
    ClientCapabilities, DocumentSymbolClientCapabilities, DocumentSymbolParams,
    DocumentSymbolResponse, DynamicRegistrationClientCapabilities, InitializeParams,
    InitializeResult, Position, ReferenceContext, ReferenceParams,
    TextDocumentClientCapabilities, TextDocumentIdentifier, TextDocumentItem,
    TextDocumentPositionParams, TypeHierarchyClientCapabilities, TypeHierarchyItem,
    TypeHierarchyPrepareParams, TypeHierarchySubtypesParams, TypeHierarchySupertypesParams,
    Uri, WorkspaceFolder,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Child;
use tokio::sync::{Mutex, oneshot};

/// A JSON-RPC request message.
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: i64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

/// A JSON-RPC notification (no id).
#[derive(Debug, Serialize)]
struct JsonRpcNotification {
    jsonrpc: &'static str,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

/// A JSON-RPC response message.
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<i64>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

/// JSON-RPC error.
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i64,
    message: String,
}

type PendingMap = Arc<Mutex<HashMap<i64, oneshot::Sender<Value>>>>;

/// Convert a file path to a `file://` URI.
pub fn path_to_uri(path: &Path) -> engram_core::Result<Uri> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(EngramError::Io)?
            .join(path)
    };
    let uri_str = format!("file://{}", abs.display());
    uri_str
        .parse()
        .map_err(|e| EngramError::Mcp(format!("Invalid URI '{}': {}", uri_str, e)))
}

/// Low-level LSP client that communicates with a single language server process.
pub struct LspClient {
    stdin: Mutex<tokio::process::ChildStdin>,
    pending: PendingMap,
    next_id: AtomicI64,
    _child: Child,
    reader_handle: tokio::task::JoinHandle<()>,
    pub root_uri: Uri,
    pub language_id: String,
    opened_files: Mutex<std::collections::HashSet<String>>,
}

impl LspClient {
    /// Start a language server process and prepare the client for communication.
    /// Call `initialize()` after this to complete the LSP handshake.
    pub async fn start(
        config: &LangServerConfig,
        root_path: &Path,
    ) -> engram_core::Result<Self> {
        let mut child = tokio::process::Command::new(config.command)
            .args(config.args)
            .current_dir(root_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| {
                EngramError::Mcp(format!(
                    "Failed to start language server '{}': {}",
                    config.command, e
                ))
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            EngramError::Mcp("Failed to get stdin for language server".into())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            EngramError::Mcp("Failed to get stdout for language server".into())
        })?;

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let pending_reader = pending.clone();

        let reader_handle = tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            loop {
                match Self::read_message(&mut reader).await {
                    Ok(Some(msg)) => {
                        if let Some(id) = msg.id {
                            let mut map = pending_reader.lock().await;
                            if let Some(sender) = map.remove(&id) {
                                let value = if let Some(err) = msg.error {
                                    Value::String(format!("LSP error: {}", err.message))
                                } else {
                                    msg.result.unwrap_or(Value::Null)
                                };
                                let _ = sender.send(value);
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(_) => break,
                }
            }
        });

        let root_uri = path_to_uri(root_path)?;

        Ok(Self {
            stdin: Mutex::new(stdin),
            pending,
            next_id: AtomicI64::new(1),
            _child: child,
            reader_handle,
            root_uri,
            language_id: config.language_id.to_string(),
            opened_files: Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Read a single LSP message from the reader.
    async fn read_message<R: tokio::io::AsyncBufRead + Unpin>(
        reader: &mut R,
    ) -> engram_core::Result<Option<JsonRpcResponse>> {
        let mut content_length: Option<usize> = None;
        loop {
            let mut line = String::new();
            let bytes_read = reader
                .read_line(&mut line)
                .await
                .map_err(EngramError::Io)?;
            if bytes_read == 0 {
                return Ok(None);
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some(len_str) = trimmed.strip_prefix("Content-Length:") {
                content_length = len_str.trim().parse().ok();
            }
        }

        let length = content_length.ok_or_else(|| {
            EngramError::Mcp("Missing Content-Length header".into())
        })?;

        let mut body = vec![0u8; length];
        reader
            .read_exact(&mut body)
            .await
            .map_err(EngramError::Io)?;

        match serde_json::from_slice(&body) {
            Ok(msg) => Ok(Some(msg)),
            Err(_) => Ok(Some(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: None,
                result: None,
                error: None,
            })),
        }
    }

    /// Send a JSON-RPC request and await the response.
    pub async fn request<R: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> engram_core::Result<R> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_string(),
            params,
        };

        self.send_raw(&serde_json::to_string(&request).map_err(|e| {
            EngramError::Serialize(format!("Failed to serialize request: {}", e))
        })?)
        .await?;

        let value = rx.await.map_err(|_| {
            EngramError::Mcp("LSP response channel closed".into())
        })?;

        if let Value::String(ref s) = value {
            if s.starts_with("LSP error:") {
                return Err(EngramError::Mcp(s.clone()));
            }
        }

        serde_json::from_value(value).map_err(|e| {
            EngramError::Serialize(format!("Failed to deserialize LSP response: {}", e))
        })
    }

    /// Send a JSON-RPC notification (no response expected).
    pub async fn notify(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> engram_core::Result<()> {
        let notification = JsonRpcNotification {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
        };

        self.send_raw(&serde_json::to_string(&notification).map_err(|e| {
            EngramError::Serialize(format!("Failed to serialize notification: {}", e))
        })?)
        .await
    }

    /// Write a raw LSP message to stdin.
    async fn send_raw(&self, body: &str) -> engram_core::Result<()> {
        let message = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(message.as_bytes())
            .await
            .map_err(|e| EngramError::Mcp(format!("Failed to write to LSP: {}", e)))?;
        stdin
            .flush()
            .await
            .map_err(|e| EngramError::Mcp(format!("Failed to flush LSP stdin: {}", e)))?;
        Ok(())
    }

    /// Initialize the LSP connection with the server.
    pub async fn initialize(&self) -> engram_core::Result<InitializeResult> {
        #[allow(deprecated)]
        let params = InitializeParams {
            root_uri: Some(self.root_uri.clone()),
            capabilities: ClientCapabilities {
                text_document: Some(TextDocumentClientCapabilities {
                    document_symbol: Some(DocumentSymbolClientCapabilities {
                        hierarchical_document_symbol_support: Some(true),
                        ..Default::default()
                    }),
                    references: Some(DynamicRegistrationClientCapabilities {
                        dynamic_registration: Some(false),
                    }),
                    type_hierarchy: Some(TypeHierarchyClientCapabilities {
                        dynamic_registration: Some(false),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            workspace_folders: Some(vec![WorkspaceFolder {
                uri: self.root_uri.clone(),
                name: "root".to_string(),
            }]),
            ..Default::default()
        };

        let result: InitializeResult = self
            .request(
                "initialize",
                Some(serde_json::to_value(params).unwrap()),
            )
            .await?;

        self.notify(
            "initialized",
            Some(serde_json::to_value(lsp_types::InitializedParams {}).unwrap()),
        )
        .await?;

        Ok(result)
    }

    /// Ensure a file is opened in the language server.
    pub async fn open_file(&self, path: &Path) -> engram_core::Result<()> {
        let uri_str = path.to_string_lossy().to_string();

        let mut opened = self.opened_files.lock().await;
        if opened.contains(&uri_str) {
            return Ok(());
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(EngramError::Io)?;

        let uri = path_to_uri(path)?;

        let params = lsp_types::DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri,
                language_id: self.language_id.clone(),
                version: 1,
                text: content,
            },
        };

        self.notify(
            "textDocument/didOpen",
            Some(serde_json::to_value(params).unwrap()),
        )
        .await?;

        opened.insert(uri_str);
        Ok(())
    }

    /// Get document symbols for a file.
    pub async fn document_symbols(
        &self,
        path: &Path,
    ) -> engram_core::Result<Option<DocumentSymbolResponse>> {
        self.open_file(path).await?;

        let params = DocumentSymbolParams {
            text_document: TextDocumentIdentifier { uri: path_to_uri(path)? },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        self.request(
            "textDocument/documentSymbol",
            Some(serde_json::to_value(params).unwrap()),
        )
        .await
    }

    /// Find all references to a symbol at a given position.
    pub async fn find_references(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> engram_core::Result<Option<Vec<lsp_types::Location>>> {
        self.open_file(path).await?;

        let params = ReferenceParams {
            text_document_position: TextDocumentPositionParams {
                text_document: TextDocumentIdentifier { uri: path_to_uri(path)? },
                position: Position::new(line, character),
            },
            context: ReferenceContext {
                include_declaration: true,
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        self.request(
            "textDocument/references",
            Some(serde_json::to_value(params).unwrap()),
        )
        .await
    }

    /// Prepare type hierarchy at a position.
    pub async fn prepare_type_hierarchy(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> engram_core::Result<Option<Vec<TypeHierarchyItem>>> {
        self.open_file(path).await?;

        let params = TypeHierarchyPrepareParams {
            text_document_position_params: TextDocumentPositionParams {
                text_document: TextDocumentIdentifier { uri: path_to_uri(path)? },
                position: Position::new(line, character),
            },
            work_done_progress_params: Default::default(),
        };

        self.request(
            "textDocument/prepareTypeHierarchy",
            Some(serde_json::to_value(params).unwrap()),
        )
        .await
    }

    /// Get supertypes of a type hierarchy item.
    pub async fn supertypes(
        &self,
        item: TypeHierarchyItem,
    ) -> engram_core::Result<Option<Vec<TypeHierarchyItem>>> {
        let params = TypeHierarchySupertypesParams {
            item,
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        self.request(
            "typeHierarchy/supertypes",
            Some(serde_json::to_value(params).unwrap()),
        )
        .await
    }

    /// Get subtypes of a type hierarchy item.
    pub async fn subtypes(
        &self,
        item: TypeHierarchyItem,
    ) -> engram_core::Result<Option<Vec<TypeHierarchyItem>>> {
        let params = TypeHierarchySubtypesParams {
            item,
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        self.request(
            "typeHierarchy/subtypes",
            Some(serde_json::to_value(params).unwrap()),
        )
        .await
    }

    /// Shut down the language server gracefully.
    pub async fn shutdown(&self) -> engram_core::Result<()> {
        let _: Value = self.request("shutdown", None).await?;
        self.notify("exit", None).await?;
        Ok(())
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        self.reader_handle.abort();
    }
}
