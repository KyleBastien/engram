use async_trait::async_trait;
use engram_core::{
    EngramError, ExportedSymbol, ResolvedImport, SymbolId, SymbolReference, SymbolResolver,
    TypeHierarchy,
};
use lsp_types::{DocumentSymbolResponse, SymbolKind};
use std::path::{Path, PathBuf};

use crate::manager::ServerManager;

/// LSP-based implementation of `SymbolResolver`.
///
/// Uses on-demand language servers (TypeScript, Rust, Python, Go) for exact symbol resolution
/// instead of tree-sitter heuristics.
pub struct LspSymbolResolver {
    manager: ServerManager,
}

impl LspSymbolResolver {
    /// Create a new LSP symbol resolver for the given workspace root.
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            manager: ServerManager::new(root_path),
        }
    }

    /// Shut down all running language servers.
    pub async fn shutdown(&self) -> engram_core::Result<()> {
        self.manager.shutdown_all().await
    }

    /// Find the 0-based line number of a symbol name in a file.
    async fn find_symbol_line(file: &Path, name: &str) -> engram_core::Result<u32> {
        let search_name = name.rsplit("::").next().unwrap_or(name);

        let content = tokio::fs::read_to_string(file)
            .await
            .map_err(EngramError::Io)?;

        for (idx, line_text) in content.lines().enumerate() {
            if line_text.contains(search_name) {
                return Ok(idx as u32);
            }
        }

        Ok(0)
    }
}

/// Convert an LSP SymbolKind to a string kind name.
fn symbol_kind_to_string(kind: SymbolKind) -> String {
    match kind {
        SymbolKind::FILE => "file",
        SymbolKind::MODULE => "module",
        SymbolKind::NAMESPACE => "namespace",
        SymbolKind::PACKAGE => "package",
        SymbolKind::CLASS => "class",
        SymbolKind::METHOD => "method",
        SymbolKind::PROPERTY => "property",
        SymbolKind::FIELD => "field",
        SymbolKind::CONSTRUCTOR => "constructor",
        SymbolKind::ENUM => "enum",
        SymbolKind::INTERFACE => "interface",
        SymbolKind::FUNCTION => "function",
        SymbolKind::VARIABLE => "variable",
        SymbolKind::CONSTANT => "constant",
        SymbolKind::STRING => "string",
        SymbolKind::NUMBER => "number",
        SymbolKind::BOOLEAN => "boolean",
        SymbolKind::ARRAY => "array",
        SymbolKind::OBJECT => "object",
        SymbolKind::KEY => "key",
        SymbolKind::NULL => "null",
        SymbolKind::ENUM_MEMBER => "enum_member",
        SymbolKind::STRUCT => "struct",
        SymbolKind::EVENT => "event",
        SymbolKind::OPERATOR => "operator",
        SymbolKind::TYPE_PARAMETER => "type_parameter",
        _ => "unknown",
    }
    .to_string()
}

/// Extract exported symbols from an LSP DocumentSymbolResponse.
fn symbols_from_response(
    file: &Path,
    response: &DocumentSymbolResponse,
) -> Vec<ExportedSymbol> {
    match response {
        DocumentSymbolResponse::Flat(symbols) => symbols
            .iter()
            .map(|s| ExportedSymbol {
                id: SymbolId {
                    file: file.to_path_buf(),
                    name: s.name.clone(),
                    kind: symbol_kind_to_string(s.kind),
                },
                line: s.location.range.start.line as usize + 1,
                is_public: true,
                doc: None,
                chunk_id: None,
            })
            .collect(),
        DocumentSymbolResponse::Nested(symbols) => {
            let mut result = Vec::new();
            collect_nested_symbols(file, symbols, &mut result, None);
            result
        }
    }
}

/// Recursively collect symbols from a nested DocumentSymbol tree.
fn collect_nested_symbols(
    file: &Path,
    symbols: &[lsp_types::DocumentSymbol],
    result: &mut Vec<ExportedSymbol>,
    parent_name: Option<&str>,
) {
    for sym in symbols {
        let name = match parent_name {
            Some(p) => format!("{}::{}", p, sym.name),
            None => sym.name.clone(),
        };

        result.push(ExportedSymbol {
            id: SymbolId {
                file: file.to_path_buf(),
                name: name.clone(),
                kind: symbol_kind_to_string(sym.kind),
            },
            line: sym.range.start.line as usize + 1,
            is_public: true,
            doc: sym.detail.clone(),
            chunk_id: None,
        });

        if let Some(children) = &sym.children {
            collect_nested_symbols(file, children, result, Some(&name));
        }
    }
}

/// Extract a file path from an LSP URI.
fn uri_to_path(uri: &lsp_types::Uri) -> Option<PathBuf> {
    let s = uri.as_str();
    s.strip_prefix("file://").map(PathBuf::from)
}

#[async_trait]
impl SymbolResolver for LspSymbolResolver {
    async fn extract_exports(
        &self,
        file: &Path,
    ) -> engram_core::Result<Vec<ExportedSymbol>> {
        let key = match self.manager.ensure_server(file).await? {
            Some(k) => k,
            None => return Ok(Vec::new()),
        };

        let servers = self.manager.servers().lock().await;
        let client = servers.get(&key).unwrap();
        let response = client.document_symbols(file).await?;
        match response {
            Some(resp) => Ok(symbols_from_response(file, &resp)),
            None => Ok(Vec::new()),
        }
    }

    async fn resolve_imports(
        &self,
        _file: &Path,
    ) -> engram_core::Result<Vec<ResolvedImport>> {
        // LSP doesn't have a direct "resolve imports" method.
        // A full implementation would use textDocument/definition on import statements.
        Ok(Vec::new())
    }

    async fn find_references(
        &self,
        symbol: &SymbolId,
    ) -> engram_core::Result<Vec<SymbolReference>> {
        let file = &symbol.file;
        let line = Self::find_symbol_line(file, &symbol.name).await?;

        let key = match self.manager.ensure_server(file).await? {
            Some(k) => k,
            None => return Ok(Vec::new()),
        };

        let servers = self.manager.servers().lock().await;
        let client = servers.get(&key).unwrap();
        let locations = client.find_references(file, line, 0).await?;

        match locations {
            Some(locs) => Ok(locs
                .into_iter()
                .filter_map(|loc| {
                    let ref_path = uri_to_path(&loc.uri)?;
                    Some(SymbolReference {
                        symbol: symbol.clone(),
                        file: ref_path,
                        line: loc.range.start.line as usize + 1,
                        column: loc.range.start.character as usize,
                    })
                })
                .collect()),
            None => Ok(Vec::new()),
        }
    }

    async fn type_hierarchy(
        &self,
        symbol: &SymbolId,
    ) -> engram_core::Result<Option<TypeHierarchy>> {
        let file = &symbol.file;
        let line = Self::find_symbol_line(file, &symbol.name).await?;

        let key = match self.manager.ensure_server(file).await? {
            Some(k) => k,
            None => return Ok(None),
        };

        let servers = self.manager.servers().lock().await;
        let client = servers.get(&key).unwrap();

        let items = client.prepare_type_hierarchy(file, line, 0).await?;

        let items = match items {
            Some(items) if !items.is_empty() => items,
            _ => return Ok(None),
        };

        let item = items.into_iter().next().unwrap();

        let parents = client
            .supertypes(item.clone())
            .await?
            .unwrap_or_default()
            .into_iter()
            .filter_map(|ti| {
                let path = uri_to_path(&ti.uri)?;
                Some(SymbolId {
                    file: path,
                    name: ti.name,
                    kind: symbol_kind_to_string(ti.kind),
                })
            })
            .collect();

        let children = client
            .subtypes(item)
            .await?
            .unwrap_or_default()
            .into_iter()
            .filter_map(|ti| {
                let path = uri_to_path(&ti.uri)?;
                Some(SymbolId {
                    file: path,
                    name: ti.name,
                    kind: symbol_kind_to_string(ti.kind),
                })
            })
            .collect();

        Ok(Some(TypeHierarchy {
            symbol: symbol.clone(),
            parents,
            children,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_kind_conversion() {
        assert_eq!(symbol_kind_to_string(SymbolKind::FUNCTION), "function");
        assert_eq!(symbol_kind_to_string(SymbolKind::CLASS), "class");
        assert_eq!(symbol_kind_to_string(SymbolKind::STRUCT), "struct");
        assert_eq!(symbol_kind_to_string(SymbolKind::METHOD), "method");
        assert_eq!(symbol_kind_to_string(SymbolKind::INTERFACE), "interface");
        assert_eq!(symbol_kind_to_string(SymbolKind::ENUM), "enum");
        assert_eq!(symbol_kind_to_string(SymbolKind::MODULE), "module");
        assert_eq!(symbol_kind_to_string(SymbolKind::VARIABLE), "variable");
        assert_eq!(symbol_kind_to_string(SymbolKind::CONSTANT), "constant");
        assert_eq!(
            symbol_kind_to_string(SymbolKind::TYPE_PARAMETER),
            "type_parameter"
        );
    }

    #[test]
    fn flat_symbol_response() {
        use lsp_types::{Location, Range, SymbolInformation};

        let uri: lsp_types::Uri = "file:///test/foo.rs".parse().unwrap();
        #[allow(deprecated)]
        let response = DocumentSymbolResponse::Flat(vec![SymbolInformation {
            name: "my_func".to_string(),
            kind: SymbolKind::FUNCTION,
            tags: None,
            deprecated: None,
            location: Location {
                uri,
                range: Range {
                    start: lsp_types::Position::new(9, 0),
                    end: lsp_types::Position::new(15, 1),
                },
            },
            container_name: None,
        }]);

        let exports = symbols_from_response(Path::new("/test/foo.rs"), &response);
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "my_func");
        assert_eq!(exports[0].id.kind, "function");
        assert_eq!(exports[0].line, 10);
    }

    #[test]
    fn nested_symbol_response() {
        use lsp_types::{DocumentSymbol, Range};

        #[allow(deprecated)]
        let response = DocumentSymbolResponse::Nested(vec![DocumentSymbol {
            name: "MyStruct".to_string(),
            detail: Some("A struct".to_string()),
            kind: SymbolKind::STRUCT,
            tags: None,
            deprecated: None,
            range: Range {
                start: lsp_types::Position::new(0, 0),
                end: lsp_types::Position::new(10, 1),
            },
            selection_range: Range {
                start: lsp_types::Position::new(0, 0),
                end: lsp_types::Position::new(0, 8),
            },
            children: Some(vec![DocumentSymbol {
                name: "my_method".to_string(),
                detail: None,
                kind: SymbolKind::METHOD,
                tags: None,
                deprecated: None,
                range: Range {
                    start: lsp_types::Position::new(5, 4),
                    end: lsp_types::Position::new(8, 5),
                },
                selection_range: Range {
                    start: lsp_types::Position::new(5, 4),
                    end: lsp_types::Position::new(5, 13),
                },
                children: None,
            }]),
        }]);

        let exports = symbols_from_response(Path::new("/test/foo.rs"), &response);
        assert_eq!(exports.len(), 2);
        assert_eq!(exports[0].id.name, "MyStruct");
        assert_eq!(exports[0].id.kind, "struct");
        assert_eq!(exports[0].doc, Some("A struct".to_string()));
        assert_eq!(exports[1].id.name, "MyStruct::my_method");
        assert_eq!(exports[1].id.kind, "method");
        assert_eq!(exports[1].line, 6);
    }

    #[test]
    fn uri_to_path_conversion() {
        let uri: lsp_types::Uri = "file:///home/user/foo.rs".parse().unwrap();
        let path = uri_to_path(&uri).unwrap();
        assert_eq!(path, PathBuf::from("/home/user/foo.rs"));
    }

    #[test]
    fn uri_to_path_non_file() {
        let uri: lsp_types::Uri = "https://example.com".parse().unwrap();
        let path = uri_to_path(&uri);
        assert!(path.is_none());
    }

    #[test]
    fn resolver_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LspSymbolResolver>();
    }

    #[test]
    fn resolver_implements_trait() {
        fn _accepts_dyn(_r: &dyn SymbolResolver) {}
        let r = LspSymbolResolver::new(PathBuf::from("/tmp"));
        _accepts_dyn(&r);
    }

    #[tokio::test]
    async fn find_symbol_line_in_content() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.rs");
        {
            let mut f = std::fs::File::create(&file).unwrap();
            writeln!(f, "fn foo() {{}}").unwrap();
            writeln!(f, "fn bar() {{}}").unwrap();
            writeln!(f, "struct Baz;").unwrap();
        }

        assert_eq!(
            LspSymbolResolver::find_symbol_line(&file, "foo")
                .await
                .unwrap(),
            0
        );
        assert_eq!(
            LspSymbolResolver::find_symbol_line(&file, "bar")
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            LspSymbolResolver::find_symbol_line(&file, "Baz")
                .await
                .unwrap(),
            2
        );
        assert_eq!(
            LspSymbolResolver::find_symbol_line(&file, "NotHere")
                .await
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn find_symbol_line_strips_qualifier() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.rs");
        {
            let mut f = std::fs::File::create(&file).unwrap();
            writeln!(f, "impl Foo {{").unwrap();
            writeln!(f, "    fn my_method() {{}}").unwrap();
            writeln!(f, "}}").unwrap();
        }

        assert_eq!(
            LspSymbolResolver::find_symbol_line(&file, "Foo::my_method")
                .await
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn extract_exports_unknown_extension() {
        let resolver = LspSymbolResolver::new(PathBuf::from("/tmp"));
        let result = resolver
            .extract_exports(Path::new("/tmp/readme.md"))
            .await
            .unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn resolve_imports_returns_empty() {
        let resolver = LspSymbolResolver::new(PathBuf::from("/tmp"));
        let result = resolver
            .resolve_imports(Path::new("/tmp/foo.rs"))
            .await
            .unwrap();
        assert!(result.is_empty());
    }
}
