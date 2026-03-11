use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A unique identifier for a symbol within the codebase.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymbolId {
    /// The file path containing the symbol.
    pub file: PathBuf,
    /// The fully-qualified name of the symbol (e.g., "MyStruct::my_method").
    pub name: String,
    /// The kind of symbol (e.g., "function", "class", "type").
    pub kind: String,
}

/// A symbol exported from a file or module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportedSymbol {
    /// The identity of the exported symbol.
    pub id: SymbolId,
    /// The line number where the symbol is defined.
    pub line: usize,
    /// Whether the export is public (vs. crate-internal or re-exported).
    pub is_public: bool,
    /// Optional documentation string for the symbol.
    pub doc: Option<String>,
    /// The chunk_id linking this export to its indexed chunk.
    #[serde(default)]
    pub chunk_id: Option<String>,
}

/// A resolved import mapping an import path to its source symbol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedImport {
    /// The import path as written in source code (e.g., "use crate::foo::Bar").
    pub import_path: String,
    /// The resolved symbol that this import refers to.
    pub resolved_symbol: SymbolId,
    /// The file containing the import statement.
    pub importing_file: PathBuf,
    /// The line number of the import statement.
    pub line: usize,
    /// The repo containing the import statement.
    #[serde(default)]
    pub importing_repo: String,
    /// The repo containing the source symbol.
    #[serde(default)]
    pub source_repo: String,
    /// The file containing the source symbol.
    #[serde(default)]
    pub source_file: PathBuf,
    /// The chunk_id of the resolved source symbol.
    #[serde(default)]
    pub resolved_chunk: Option<String>,
    /// The resolution method used (e.g., "heuristic").
    #[serde(default)]
    pub resolution: String,
}

/// A reference to a symbol at a specific location in the codebase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolReference {
    /// The symbol being referenced.
    pub symbol: SymbolId,
    /// The file containing the reference.
    pub file: PathBuf,
    /// The line number of the reference.
    pub line: usize,
    /// The column offset of the reference.
    pub column: usize,
}

/// Type inheritance and implementation relationships for a symbol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeHierarchy {
    /// The symbol whose hierarchy is being described.
    pub symbol: SymbolId,
    /// Parent types (e.g., superclasses, traits being implemented).
    pub parents: Vec<SymbolId>,
    /// Child types (e.g., subclasses, implementors).
    pub children: Vec<SymbolId>,
}

/// Trait for resolving symbols, imports, and references across a codebase.
#[async_trait]
pub trait SymbolResolver: Send + Sync {
    /// Extracts all exported symbols from the given file.
    async fn extract_exports(
        &self,
        file: &Path,
    ) -> crate::Result<Vec<ExportedSymbol>>;

    /// Resolves imports in the given file to their source symbols.
    async fn resolve_imports(
        &self,
        file: &Path,
    ) -> crate::Result<Vec<ResolvedImport>>;

    /// Finds all references to the given symbol across the indexed codebase.
    async fn find_references(
        &self,
        symbol: &SymbolId,
    ) -> crate::Result<Vec<SymbolReference>>;

    /// Returns the type hierarchy (parents and children) for the given symbol.
    async fn type_hierarchy(
        &self,
        symbol: &SymbolId,
    ) -> crate::Result<Option<TypeHierarchy>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_id_equality() {
        let a = SymbolId {
            file: PathBuf::from("src/main.rs"),
            name: "main".into(),
            kind: "function".into(),
        };
        let b = SymbolId {
            file: PathBuf::from("src/main.rs"),
            name: "main".into(),
            kind: "function".into(),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn symbol_id_hash() {
        use std::collections::HashSet;
        let id = SymbolId {
            file: PathBuf::from("src/lib.rs"),
            name: "MyStruct".into(),
            kind: "struct".into(),
        };
        let mut set = HashSet::new();
        set.insert(id.clone());
        assert!(set.contains(&id));
    }

    #[test]
    fn exported_symbol_serde_round_trip() {
        let sym = ExportedSymbol {
            id: SymbolId {
                file: PathBuf::from("src/lib.rs"),
                name: "Foo".into(),
                kind: "struct".into(),
            },
            line: 10,
            is_public: true,
            doc: Some("A foo struct.".into()),
            chunk_id: None,
        };
        let json = serde_json::to_string(&sym).unwrap();
        let deserialized: ExportedSymbol = serde_json::from_str(&json).unwrap();
        assert_eq!(sym, deserialized);
    }

    #[test]
    fn resolved_import_serde_round_trip() {
        let imp = ResolvedImport {
            import_path: "use crate::foo::Bar".into(),
            resolved_symbol: SymbolId {
                file: PathBuf::from("src/foo.rs"),
                name: "Bar".into(),
                kind: "struct".into(),
            },
            importing_file: PathBuf::from("src/main.rs"),
            line: 3,
            importing_repo: "my-repo".into(),
            source_repo: "other-repo".into(),
            source_file: PathBuf::from("src/foo.rs"),
            resolved_chunk: Some("other-repo#src/foo.rs#Bar".into()),
            resolution: "heuristic".into(),
        };
        let json = serde_json::to_string(&imp).unwrap();
        let deserialized: ResolvedImport = serde_json::from_str(&json).unwrap();
        assert_eq!(imp, deserialized);
    }

    #[test]
    fn resolved_import_backward_compat_deserialization() {
        // Old format without cross-repo fields should deserialize with defaults
        let json = r#"{"import_path":"use crate::foo","resolved_symbol":{"file":"src/foo.rs","name":"foo","kind":"function"},"importing_file":"src/main.rs","line":1}"#;
        let imp: ResolvedImport = serde_json::from_str(json).unwrap();
        assert_eq!(imp.importing_repo, "");
        assert_eq!(imp.source_repo, "");
        assert_eq!(imp.source_file, PathBuf::new());
        assert_eq!(imp.resolved_chunk, None);
        assert_eq!(imp.resolution, "");
    }

    #[test]
    fn symbol_reference_serde_round_trip() {
        let reference = SymbolReference {
            symbol: SymbolId {
                file: PathBuf::from("src/lib.rs"),
                name: "process".into(),
                kind: "function".into(),
            },
            file: PathBuf::from("src/main.rs"),
            line: 42,
            column: 8,
        };
        let json = serde_json::to_string(&reference).unwrap();
        let deserialized: SymbolReference = serde_json::from_str(&json).unwrap();
        assert_eq!(reference, deserialized);
    }

    #[test]
    fn type_hierarchy_serde_round_trip() {
        let hierarchy = TypeHierarchy {
            symbol: SymbolId {
                file: PathBuf::from("src/animals.rs"),
                name: "Dog".into(),
                kind: "struct".into(),
            },
            parents: vec![SymbolId {
                file: PathBuf::from("src/animals.rs"),
                name: "Animal".into(),
                kind: "trait".into(),
            }],
            children: vec![],
        };
        let json = serde_json::to_string(&hierarchy).unwrap();
        let deserialized: TypeHierarchy = serde_json::from_str(&json).unwrap();
        assert_eq!(hierarchy, deserialized);
    }

    #[test]
    fn trait_is_send_sync() {
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn SymbolResolver>();
    }

    struct MockResolver;

    #[async_trait]
    impl SymbolResolver for MockResolver {
        async fn extract_exports(
            &self,
            _file: &Path,
        ) -> crate::Result<Vec<ExportedSymbol>> {
            Ok(vec![ExportedSymbol {
                id: SymbolId {
                    file: PathBuf::from("src/lib.rs"),
                    name: "hello".into(),
                    kind: "function".into(),
                },
                line: 1,
                is_public: true,
                doc: None,
                chunk_id: None,
            }])
        }

        async fn resolve_imports(
            &self,
            _file: &Path,
        ) -> crate::Result<Vec<ResolvedImport>> {
            Ok(vec![])
        }

        async fn find_references(
            &self,
            _symbol: &SymbolId,
        ) -> crate::Result<Vec<SymbolReference>> {
            Ok(vec![])
        }

        async fn type_hierarchy(
            &self,
            _symbol: &SymbolId,
        ) -> crate::Result<Option<TypeHierarchy>> {
            Ok(None)
        }
    }

    #[tokio::test]
    async fn mock_resolver_works() {
        let resolver = MockResolver;
        let exports = resolver
            .extract_exports(Path::new("src/lib.rs"))
            .await
            .unwrap();
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "hello");

        let imports = resolver
            .resolve_imports(Path::new("src/main.rs"))
            .await
            .unwrap();
        assert!(imports.is_empty());

        let refs = resolver
            .find_references(&exports[0].id)
            .await
            .unwrap();
        assert!(refs.is_empty());

        let hierarchy = resolver
            .type_hierarchy(&exports[0].id)
            .await
            .unwrap();
        assert!(hierarchy.is_none());
    }

    #[test]
    fn trait_is_object_safe() {
        fn _accepts_dyn(_resolver: &dyn SymbolResolver) {}
    }
}
