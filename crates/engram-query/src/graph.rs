use std::collections::HashMap;
use std::fs;
use std::path::Path;

use engram_core::{EngramError, ExportedSymbol, ResolvedImport, Result};
use serde::{Deserialize, Serialize};

/// A node in the cross-repo symbol graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    /// Chunk ID linking to the indexed chunk (if available).
    pub chunk_id: Option<String>,
    /// Symbol name.
    pub name: String,
    /// File path containing the symbol.
    pub file: String,
    /// Repository name.
    pub repo: String,
    /// Symbol kind (e.g., "function", "class").
    pub kind: String,
}

/// Traversal direction for graph queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Symbols that this symbol imports/depends on.
    Callees,
    /// Symbols that import/depend on this symbol.
    Callers,
    /// Both directions.
    Both,
}

/// A directed graph of cross-repo symbol dependencies.
///
/// Built from `_xrefs/exports.jsonl` and `_xrefs/imports.jsonl` during boot.
/// Edges represent "repo A file X imports symbol Y from repo B file Z".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolGraph {
    nodes: Vec<GraphNode>,
    /// Map from node key to node index for deduplication.
    node_index: HashMap<String, usize>,
    /// Forward edges (outgoing): importer → [(exported symbol index, relationship)].
    outgoing: HashMap<usize, Vec<(usize, String)>>,
    /// Reverse edges (incoming): exported symbol → [(importer index, relationship)].
    incoming: HashMap<usize, Vec<(usize, String)>>,
}

impl SymbolGraph {
    /// Build a symbol graph from exports and resolved imports.
    ///
    /// Each resolved import creates an edge from the importing symbol's file
    /// to the exported symbol's file, representing a cross-repo dependency.
    pub fn build(exports: &[ExportedSymbol], imports: &[ResolvedImport]) -> Self {
        let mut graph = Self {
            nodes: Vec::new(),
            node_index: HashMap::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        };

        // Register all exported symbols as nodes
        for export in exports {
            let repo = export
                .id
                .file
                .to_string_lossy()
                .split('#')
                .next()
                .unwrap_or("unknown")
                .to_string();
            graph.get_or_insert_node(
                export.chunk_id.clone(),
                &export.id.name,
                &export.id.file.to_string_lossy(),
                &repo,
                &export.id.kind,
            );
        }

        // Create edges from imports
        for imp in imports {
            // Source node: the importing file/symbol
            let importer_key = format!(
                "{}#{}#{}",
                imp.importing_repo,
                imp.importing_file.to_string_lossy(),
                imp.import_path,
            );
            let importer_idx = graph.get_or_insert_node_by_key(
                &importer_key,
                None,
                &imp.import_path,
                &imp.importing_file.to_string_lossy(),
                &imp.importing_repo,
                "import",
            );

            // Target node: the resolved exported symbol
            let target_idx = graph.get_or_insert_node(
                imp.resolved_chunk.clone(),
                &imp.resolved_symbol.name,
                &imp.resolved_symbol.file.to_string_lossy(),
                &imp.source_repo,
                &imp.resolved_symbol.kind,
            );

            // Add directed edge: importer → target
            let relationship = if imp.importing_repo != imp.source_repo {
                "cross_repo_import".to_string()
            } else {
                "import".to_string()
            };

            graph
                .outgoing
                .entry(importer_idx)
                .or_default()
                .push((target_idx, relationship.clone()));
            graph
                .incoming
                .entry(target_idx)
                .or_default()
                .push((importer_idx, relationship));
        }

        graph
    }

    /// Build a graph from the JSONL files in a store root.
    pub fn build_from_store(store_root: &Path) -> Result<Self> {
        let exports = read_exports_jsonl(store_root)?;
        let imports = read_imports_jsonl(store_root)?;
        Ok(Self::build(&exports, &imports))
    }

    /// Get all symbols that a given symbol depends on (callees/targets).
    pub fn callees(&self, symbol_name: &str) -> Vec<&GraphNode> {
        self.find_nodes_by_name(symbol_name)
            .into_iter()
            .flat_map(|idx| {
                self.outgoing
                    .get(&idx)
                    .map(|edges| edges.iter().map(|(target, _)| &self.nodes[*target]).collect::<Vec<_>>())
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Get all symbols that depend on a given symbol (callers/sources).
    pub fn callers(&self, symbol_name: &str) -> Vec<&GraphNode> {
        self.find_nodes_by_name(symbol_name)
            .into_iter()
            .flat_map(|idx| {
                self.incoming
                    .get(&idx)
                    .map(|edges| edges.iter().map(|(source, _)| &self.nodes[*source]).collect::<Vec<_>>())
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Traverse the graph from a symbol in a given direction up to a maximum depth.
    ///
    /// Returns all reachable nodes and the edges connecting them.
    pub fn traverse(
        &self,
        symbol_name: &str,
        direction: Direction,
        depth: usize,
    ) -> TraversalResult {
        let start_indices = self.find_nodes_by_name(symbol_name);
        if start_indices.is_empty() {
            return TraversalResult {
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        }

        let mut visited: HashMap<usize, usize> = HashMap::new(); // idx -> depth
        let mut result_edges: Vec<TraversalEdge> = Vec::new();
        let mut queue: Vec<(usize, usize)> = Vec::new(); // (node_idx, current_depth)

        for &idx in &start_indices {
            visited.insert(idx, 0);
            queue.push((idx, 0));
        }

        while let Some((current, current_depth)) = queue.pop() {
            if current_depth >= depth {
                continue;
            }

            let edges_to_follow: Vec<(usize, String)> = match direction {
                Direction::Callees => self
                    .outgoing
                    .get(&current)
                    .cloned()
                    .unwrap_or_default(),
                Direction::Callers => self
                    .incoming
                    .get(&current)
                    .cloned()
                    .unwrap_or_default(),
                Direction::Both => {
                    let mut edges = self
                        .outgoing
                        .get(&current)
                        .cloned()
                        .unwrap_or_default();
                    edges.extend(
                        self.incoming
                            .get(&current)
                            .cloned()
                            .unwrap_or_default(),
                    );
                    edges
                }
            };

            for (neighbor, relationship) in edges_to_follow {
                let next_depth = current_depth + 1;

                // Determine edge direction for the result
                let (source, target) = match direction {
                    Direction::Callers => (neighbor, current),
                    _ => (current, neighbor),
                };

                result_edges.push(TraversalEdge {
                    source: self.nodes[source].clone(),
                    target: self.nodes[target].clone(),
                    relationship,
                });

                if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(neighbor) {
                    e.insert(next_depth);
                    queue.push((neighbor, next_depth));
                }
            }
        }

        let result_nodes: Vec<GraphNode> = visited
            .keys()
            .map(|&idx| self.nodes[idx].clone())
            .collect();

        TraversalResult {
            nodes: result_nodes,
            edges: result_edges,
        }
    }

    /// Returns the total number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.outgoing.values().map(|edges| edges.len()).sum()
    }

    /// Save the graph to a JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json =
            serde_json::to_vec(self).map_err(|e| EngramError::Serialize(e.to_string()))?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load the graph from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| EngramError::Serialize(e.to_string()))
    }

    /// Find all node indices matching a symbol name.
    fn find_nodes_by_name(&self, symbol_name: &str) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.name == symbol_name)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get or insert a node, returning its index. Uses chunk_id + name + file as key.
    fn get_or_insert_node(
        &mut self,
        chunk_id: Option<String>,
        name: &str,
        file: &str,
        repo: &str,
        kind: &str,
    ) -> usize {
        let key = format!("{}#{}#{}", repo, file, name);
        self.get_or_insert_node_by_key(&key, chunk_id, name, file, repo, kind)
    }

    fn get_or_insert_node_by_key(
        &mut self,
        key: &str,
        chunk_id: Option<String>,
        name: &str,
        file: &str,
        repo: &str,
        kind: &str,
    ) -> usize {
        if let Some(&idx) = self.node_index.get(key) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(GraphNode {
            chunk_id,
            name: name.to_string(),
            file: file.to_string(),
            repo: repo.to_string(),
            kind: kind.to_string(),
        });
        self.node_index.insert(key.to_string(), idx);
        idx
    }
}

/// Result of a graph traversal.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// All nodes reachable within the depth limit.
    pub nodes: Vec<GraphNode>,
    /// All edges traversed.
    pub edges: Vec<TraversalEdge>,
}

/// An edge in a traversal result.
#[derive(Debug, Clone)]
pub struct TraversalEdge {
    /// The source node of the edge.
    pub source: GraphNode,
    /// The target node of the edge.
    pub target: GraphNode,
    /// The relationship type (e.g., "import", "cross_repo_import").
    pub relationship: String,
}

// --- JSONL reading (inlined to avoid engram-ingest dependency cycle) ---

fn read_exports_jsonl(store_root: &Path) -> Result<Vec<ExportedSymbol>> {
    let path = store_root.join("index").join("_xrefs").join("exports.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = fs::read_to_string(&path)?;
    let mut symbols = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let sym: ExportedSymbol =
            serde_json::from_str(line).map_err(|e| EngramError::Serialize(e.to_string()))?;
        symbols.push(sym);
    }
    Ok(symbols)
}

fn read_imports_jsonl(store_root: &Path) -> Result<Vec<ResolvedImport>> {
    let path = store_root.join("index").join("_xrefs").join("imports.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = fs::read_to_string(&path)?;
    let mut imports = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let imp: ResolvedImport =
            serde_json::from_str(line).map_err(|e| EngramError::Serialize(e.to_string()))?;
        imports.push(imp);
    }
    Ok(imports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::SymbolId;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn make_export(name: &str, file: &str, kind: &str, chunk_id: Option<&str>) -> ExportedSymbol {
        ExportedSymbol {
            id: SymbolId {
                file: PathBuf::from(file),
                name: name.to_string(),
                kind: kind.to_string(),
            },
            line: 1,
            is_public: true,
            doc: None,
            chunk_id: chunk_id.map(|s| s.to_string()),
        }
    }

    fn make_import(
        import_path: &str,
        symbol_name: &str,
        symbol_file: &str,
        symbol_kind: &str,
        importing_file: &str,
        importing_repo: &str,
        source_repo: &str,
        resolved_chunk: Option<&str>,
    ) -> ResolvedImport {
        ResolvedImport {
            import_path: import_path.to_string(),
            resolved_symbol: SymbolId {
                file: PathBuf::from(symbol_file),
                name: symbol_name.to_string(),
                kind: symbol_kind.to_string(),
            },
            importing_file: PathBuf::from(importing_file),
            line: 1,
            importing_repo: importing_repo.to_string(),
            source_repo: source_repo.to_string(),
            source_file: PathBuf::from(symbol_file),
            resolved_chunk: resolved_chunk.map(|s| s.to_string()),
            resolution: "heuristic".to_string(),
        }
    }

    #[test]
    fn empty_graph() {
        let graph = SymbolGraph::build(&[], &[]);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn exports_only_creates_nodes() {
        let exports = vec![
            make_export("Foo", "src/foo.ts", "class", Some("repo-a#src/foo.ts#Foo")),
            make_export("Bar", "src/bar.ts", "function", Some("repo-a#src/bar.ts#Bar")),
        ];
        let graph = SymbolGraph::build(&exports, &[]);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn import_creates_edge() {
        let exports = vec![make_export(
            "Foo",
            "src/foo.ts",
            "class",
            Some("repo-a#src/foo.ts#Foo"),
        )];
        let imports = vec![make_import(
            "import { Foo } from './foo'",
            "Foo",
            "src/foo.ts",
            "class",
            "src/main.ts",
            "repo-b",
            "repo-a",
            Some("repo-a#src/foo.ts#Foo"),
        )];

        let graph = SymbolGraph::build(&exports, &imports);
        assert!(graph.edge_count() > 0);

        // The importer should have Foo as a callee
        let callees = graph.callees("import { Foo } from './foo'");
        assert_eq!(callees.len(), 1);
        assert_eq!(callees[0].name, "Foo");

        // Foo should have the importer as a caller
        let callers = graph.callers("Foo");
        assert_eq!(callers.len(), 1);
    }

    #[test]
    fn cross_repo_import_labeled() {
        let exports = vec![make_export("Config", "src/config.rs", "struct", None)];
        let imports = vec![make_import(
            "use shared::Config",
            "Config",
            "src/config.rs",
            "struct",
            "src/main.rs",
            "app",
            "shared",
            None,
        )];

        let graph = SymbolGraph::build(&exports, &imports);
        let result = graph.traverse("use shared::Config", Direction::Callees, 1);
        assert!(!result.edges.is_empty());
        assert_eq!(result.edges[0].relationship, "cross_repo_import");
    }

    #[test]
    fn same_repo_import_labeled() {
        let exports = vec![make_export("helper", "src/utils.rs", "function", None)];
        let imports = vec![make_import(
            "use crate::utils::helper",
            "helper",
            "src/utils.rs",
            "function",
            "src/main.rs",
            "my-repo",
            "my-repo",
            None,
        )];

        let graph = SymbolGraph::build(&exports, &imports);
        let result = graph.traverse("use crate::utils::helper", Direction::Callees, 1);
        assert!(!result.edges.is_empty());
        assert_eq!(result.edges[0].relationship, "import");
    }

    #[test]
    fn traverse_callees_depth_limited() {
        // A -> B -> C, depth 1 from A should only reach B
        let exports = vec![
            make_export("A", "a.ts", "function", None),
            make_export("B", "b.ts", "function", None),
            make_export("C", "c.ts", "function", None),
        ];
        let imports = vec![
            make_import("import A", "B", "b.ts", "function", "a.ts", "r", "r", None),
            make_import("import B", "C", "c.ts", "function", "b.ts", "r", "r", None),
        ];

        let graph = SymbolGraph::build(&exports, &imports);

        let result_depth1 = graph.traverse("import A", Direction::Callees, 1);
        let names: Vec<&str> = result_depth1.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"B"), "depth 1 should reach B");
        // C might not be reached at depth 1 since it's 2 hops away
        // The import A -> B is 1 hop, then B -> C would need another hop

        let result_depth2 = graph.traverse("import A", Direction::Callees, 2);
        assert!(
            result_depth2.nodes.len() >= result_depth1.nodes.len(),
            "depth 2 should reach at least as many nodes as depth 1"
        );
    }

    #[test]
    fn traverse_callers() {
        let exports = vec![make_export("SharedUtil", "src/shared.ts", "function", None)];
        let imports = vec![
            make_import(
                "import { SharedUtil }",
                "SharedUtil",
                "src/shared.ts",
                "function",
                "src/a.ts",
                "app-a",
                "shared",
                None,
            ),
            make_import(
                "import { SharedUtil }",
                "SharedUtil",
                "src/shared.ts",
                "function",
                "src/b.ts",
                "app-b",
                "shared",
                None,
            ),
        ];

        let graph = SymbolGraph::build(&exports, &imports);
        let result = graph.traverse("SharedUtil", Direction::Callers, 1);
        assert!(result.nodes.len() >= 2, "SharedUtil should have at least 2 callers");
    }

    #[test]
    fn traverse_both_directions() {
        let exports = vec![
            make_export("A", "a.ts", "function", None),
            make_export("B", "b.ts", "function", None),
            make_export("C", "c.ts", "function", None),
        ];
        let imports = vec![
            make_import("import A uses B", "B", "b.ts", "function", "a.ts", "r", "r", None),
            make_import("import C uses B", "B", "b.ts", "function", "c.ts", "r", "r", None),
        ];

        let graph = SymbolGraph::build(&exports, &imports);
        let result = graph.traverse("B", Direction::Both, 1);
        // B should be connected to both the import from A and the import from C
        assert!(
            result.nodes.len() >= 2,
            "Both direction should find callers and callees"
        );
    }

    #[test]
    fn save_and_load_round_trip() {
        let exports = vec![make_export("Foo", "src/foo.ts", "class", Some("r#src/foo.ts#Foo"))];
        let imports = vec![make_import(
            "import { Foo }",
            "Foo",
            "src/foo.ts",
            "class",
            "src/main.ts",
            "app",
            "lib",
            Some("r#src/foo.ts#Foo"),
        )];

        let graph = SymbolGraph::build(&exports, &imports);
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("graph.json");

        graph.save(&path).unwrap();
        let loaded = SymbolGraph::load(&path).unwrap();

        assert_eq!(loaded.node_count(), graph.node_count());
        assert_eq!(loaded.edge_count(), graph.edge_count());
    }

    #[test]
    fn build_from_store_empty() {
        let tmp = TempDir::new().unwrap();
        let graph = SymbolGraph::build_from_store(tmp.path()).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn build_from_store_with_jsonl() {
        let tmp = TempDir::new().unwrap();
        let xrefs_dir = tmp.path().join("index").join("_xrefs");
        fs::create_dir_all(&xrefs_dir).unwrap();

        let export = make_export("Foo", "src/foo.ts", "class", Some("lib#src/foo.ts#Foo"));
        let export_json = serde_json::to_string(&export).unwrap();
        fs::write(xrefs_dir.join("exports.jsonl"), format!("{export_json}\n")).unwrap();

        let import = make_import(
            "import { Foo }",
            "Foo",
            "src/foo.ts",
            "class",
            "src/app.ts",
            "app",
            "lib",
            Some("lib#src/foo.ts#Foo"),
        );
        let import_json = serde_json::to_string(&import).unwrap();
        fs::write(xrefs_dir.join("imports.jsonl"), format!("{import_json}\n")).unwrap();

        let graph = SymbolGraph::build_from_store(tmp.path()).unwrap();
        assert!(graph.node_count() > 0);
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn default_is_empty() {
        let graph = SymbolGraph::default();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn nonexistent_symbol_returns_empty() {
        let graph = SymbolGraph::build(&[], &[]);
        assert!(graph.callers("nonexistent").is_empty());
        assert!(graph.callees("nonexistent").is_empty());

        let result = graph.traverse("nonexistent", Direction::Both, 5);
        assert!(result.nodes.is_empty());
        assert!(result.edges.is_empty());
    }
}
