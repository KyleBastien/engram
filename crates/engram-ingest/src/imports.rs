use engram_core::{ExportedSymbol, ResolvedImport, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tree_sitter::{Node, Parser};

use crate::chunker::Language;
use crate::detect::{detect_language, ChunkerKind};

/// An unresolved import parsed from source code.
#[derive(Debug, Clone)]
pub struct RawImport {
    /// The import path as written in source (e.g., "import { Foo } from './bar'").
    pub import_path: String,
    /// The individual symbol names imported (e.g., ["Foo", "Bar"]).
    pub imported_names: Vec<String>,
    /// The module/path being imported from (e.g., "./bar", "crate::foo", "os.path").
    pub module_path: String,
    /// The line number of the import statement.
    pub line: usize,
}

/// Extracts and resolves imports from source files using tree-sitter AST parsing.
pub struct TreeSitterImportResolver;

impl TreeSitterImportResolver {
    pub fn new() -> Self {
        Self
    }

    /// Extract raw imports from source code given its language.
    pub fn extract_imports_from_source(
        &self,
        file: &Path,
        source: &str,
        language: Language,
    ) -> Vec<RawImport> {
        let mut parser = Parser::new();
        let ts_language: tree_sitter::Language = match language {
            Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::Python => tree_sitter_python::LANGUAGE.into(),
            Language::Go => tree_sitter_go::LANGUAGE.into(),
            Language::Java => tree_sitter_java::LANGUAGE.into(),
        };
        parser
            .set_language(&ts_language)
            .expect("failed to set language");

        let tree = match parser.parse(source, None) {
            Some(tree) => tree,
            None => return vec![],
        };

        let root = tree.root_node();
        match language {
            Language::TypeScript => self.extract_ts_imports(&root, source, file),
            Language::Rust => self.extract_rust_imports(&root, source, file),
            Language::Python => self.extract_python_imports(&root, source, file),
            Language::Go => vec![], // Go import extraction not yet implemented
            Language::Java => vec![], // Java import extraction not yet implemented
        }
    }

    /// Extract raw imports from a file on disk.
    pub fn extract_imports_from_file(&self, file: &Path) -> Result<Vec<RawImport>> {
        let source = fs::read_to_string(file)?;
        let lang = match detect_language(file) {
            ChunkerKind::TreeSitter(lang) => lang,
            _ => return Ok(vec![]),
        };
        Ok(self.extract_imports_from_source(file, &source, lang))
    }

    /// Resolve raw imports against known exports using heuristic name matching.
    /// Returns ResolvedImport for each import that can be matched to an export.
    pub fn resolve_imports(
        &self,
        raw_imports: &[RawImport],
        exports: &[ExportedSymbol],
        importing_repo: &str,
        importing_file: &Path,
    ) -> Vec<ResolvedImport> {
        // Build a lookup: symbol name -> Vec<ExportedSymbol>
        let mut export_index: HashMap<String, Vec<&ExportedSymbol>> = HashMap::new();
        for exp in exports {
            export_index
                .entry(exp.id.name.clone())
                .or_default()
                .push(exp);
        }

        let mut resolved = Vec::new();

        for raw in raw_imports {
            for name in &raw.imported_names {
                if let Some(candidates) = export_index.get(name) {
                    // Pick the best candidate heuristically:
                    // 1. Prefer exports from a different repo (cross-repo)
                    // 2. Prefer public exports
                    // 3. Prefer exports with chunk_id
                    let best = pick_best_candidate(candidates, &raw.module_path);
                    if let Some(exp) = best {
                        // Determine source_repo from the export's file path context
                        // For cross-repo, the source_repo comes from the export index
                        let source_repo = extract_repo_from_export(exp);
                        resolved.push(ResolvedImport {
                            import_path: raw.import_path.clone(),
                            resolved_symbol: exp.id.clone(),
                            importing_file: importing_file.to_path_buf(),
                            line: raw.line,
                            importing_repo: importing_repo.to_string(),
                            source_repo,
                            source_file: exp.id.file.clone(),
                            resolved_chunk: exp.chunk_id.clone(),
                            resolution: "heuristic".to_string(),
                        });
                    }
                }
            }
        }

        resolved
    }

    // --- TypeScript ---

    fn extract_ts_imports(
        &self,
        root: &Node,
        source: &str,
        _file: &Path,
    ) -> Vec<RawImport> {
        let mut imports = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            if child.kind() == "import_statement" {
                if let Some(raw) = self.extract_ts_import_statement(&child, source) {
                    imports.push(raw);
                }
            }
        }

        imports
    }

    fn extract_ts_import_statement(
        &self,
        node: &Node,
        source: &str,
    ) -> Option<RawImport> {
        let line = node.start_position().row + 1;
        let import_text = node_text(node, source);

        // Extract the source/module path from the "source" field
        let source_node = node.child_by_field_name("source")?;
        let module_path = node_text(&source_node, source)
            .trim_matches(|c| c == '\'' || c == '"')
            .to_string();

        let mut names = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "import_clause" => {
                    self.extract_ts_import_clause_names(&child, source, &mut names);
                }
                "named_imports" => {
                    self.extract_ts_named_import_names(&child, source, &mut names);
                }
                "identifier" => {
                    // Default import: import foo from 'bar'
                    names.push(node_text(&child, source));
                }
                "namespace_import" => {
                    // import * as foo from 'bar'
                    if let Some(name_node) = child.child_by_field_name("name") {
                        names.push(node_text(&name_node, source));
                    } else {
                        // Fallback: parse "* as name" from text
                        let text = node_text(&child, source);
                        if let Some(alias) = text.strip_prefix("* as ") {
                            names.push(alias.trim().to_string());
                        }
                    }
                }
                _ => {}
            }
        }

        // If no names were extracted, try the full import text
        if names.is_empty() {
            // Side-effect import like `import './styles.css'` — skip
            return None;
        }

        Some(RawImport {
            import_path: import_text,
            imported_names: names,
            module_path,
            line,
        })
    }

    fn extract_ts_import_clause_names(
        &self,
        node: &Node,
        source: &str,
        names: &mut Vec<String>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "identifier" => {
                    names.push(node_text(&child, source));
                }
                "named_imports" => {
                    self.extract_ts_named_import_names(&child, source, names);
                }
                "namespace_import" => {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        names.push(node_text(&name_node, source));
                    } else {
                        let text = node_text(&child, source);
                        if let Some(alias) = text.strip_prefix("* as ") {
                            names.push(alias.trim().to_string());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn extract_ts_named_import_names(
        &self,
        node: &Node,
        source: &str,
        names: &mut Vec<String>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "import_specifier" {
                // Use alias if present, otherwise use name
                let name = if let Some(alias) = child.child_by_field_name("alias") {
                    node_text(&alias, source)
                } else if let Some(name_node) = child.child_by_field_name("name") {
                    node_text(&name_node, source)
                } else {
                    continue;
                };
                names.push(name);
            }
        }
    }

    // --- Rust ---

    fn extract_rust_imports(
        &self,
        root: &Node,
        source: &str,
        _file: &Path,
    ) -> Vec<RawImport> {
        let mut imports = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            if child.kind() == "use_declaration" {
                self.extract_rust_use(&child, source, &mut imports);
            }
        }

        imports
    }

    fn extract_rust_use(
        &self,
        node: &Node,
        source: &str,
        imports: &mut Vec<RawImport>,
    ) {
        let line = node.start_position().row + 1;
        let import_text = node_text(node, source);

        // Extract the use path text, stripping "use " prefix and ";" suffix
        let path_text = import_text
            .trim_start_matches("pub ")
            .trim_start_matches("use ")
            .trim_end_matches(';')
            .trim()
            .to_string();

        // Parse names from the use path
        let (module_path, names) = parse_rust_use_path(&path_text);

        if !names.is_empty() {
            imports.push(RawImport {
                import_path: import_text,
                imported_names: names,
                module_path,
                line,
            });
        }
    }

    // --- Python ---

    fn extract_python_imports(
        &self,
        root: &Node,
        source: &str,
        _file: &Path,
    ) -> Vec<RawImport> {
        let mut imports = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                "import_statement" => {
                    self.extract_python_import_stmt(&child, source, &mut imports);
                }
                "import_from_statement" => {
                    self.extract_python_from_import(&child, source, &mut imports);
                }
                _ => {}
            }
        }

        imports
    }

    fn extract_python_import_stmt(
        &self,
        node: &Node,
        source: &str,
        imports: &mut Vec<RawImport>,
    ) {
        let line = node.start_position().row + 1;
        let import_text = node_text(node, source);

        let mut names = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "dotted_name" => {
                    let text = node_text(&child, source);
                    // For `import os.path`, the imported name is the top-level module
                    let top = text.split('.').next().unwrap_or(&text).to_string();
                    names.push(top);
                }
                "aliased_import" => {
                    // import foo as bar — use the alias
                    if let Some(alias) = child.child_by_field_name("alias") {
                        names.push(node_text(&alias, source));
                    } else if let Some(name_node) = child.child_by_field_name("name") {
                        let text = node_text(&name_node, source);
                        let top = text.split('.').next().unwrap_or(&text).to_string();
                        names.push(top);
                    }
                }
                _ => {}
            }
        }

        // Module path is the full dotted name
        let module_path = import_text
            .trim_start_matches("import ")
            .split(" as ")
            .next()
            .unwrap_or("")
            .trim()
            .to_string();

        if !names.is_empty() {
            imports.push(RawImport {
                import_path: import_text,
                imported_names: names,
                module_path,
                line,
            });
        }
    }

    fn extract_python_from_import(
        &self,
        node: &Node,
        source: &str,
        imports: &mut Vec<RawImport>,
    ) {
        let line = node.start_position().row + 1;
        let import_text = node_text(node, source);

        // Extract module name from "from X import ..."
        let module_path = node
            .child_by_field_name("module_name")
            .map(|n| node_text(&n, source))
            .unwrap_or_default();

        let mut names = Vec::new();
        let mut cursor = node.walk();
        let mut found_import_keyword = false;

        for child in node.children(&mut cursor) {
            // After the "import" keyword, collect names
            if child.kind() == "import" {
                found_import_keyword = true;
                continue;
            }
            if !found_import_keyword {
                continue;
            }

            match child.kind() {
                "dotted_name" | "identifier" => {
                    names.push(node_text(&child, source));
                }
                "aliased_import" => {
                    if let Some(alias) = child.child_by_field_name("alias") {
                        names.push(node_text(&alias, source));
                    } else if let Some(name_node) = child.child_by_field_name("name") {
                        names.push(node_text(&name_node, source));
                    }
                }
                "wildcard_import" => {
                    names.push("*".to_string());
                }
                _ => {}
            }
        }

        if !names.is_empty() {
            imports.push(RawImport {
                import_path: import_text,
                imported_names: names,
                module_path,
                line,
            });
        }
    }
}

impl Default for TreeSitterImportResolver {
    fn default() -> Self {
        Self::new()
    }
}

// --- Resolution helpers ---

/// Pick the best export candidate for heuristic matching.
fn pick_best_candidate<'a>(
    candidates: &[&'a ExportedSymbol],
    module_hint: &str,
) -> Option<&'a ExportedSymbol> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates[0]);
    }

    // Score each candidate
    let mut best: Option<(&ExportedSymbol, i32)> = None;
    for &cand in candidates {
        let mut score = 0;
        // Prefer public exports
        if cand.is_public {
            score += 10;
        }
        // Prefer exports with chunk_id
        if cand.chunk_id.is_some() {
            score += 5;
        }
        // Prefer exports whose file path contains part of the module hint
        if !module_hint.is_empty() {
            let hint_parts: Vec<&str> = module_hint
                .split(['/', '.', ':'])
                .filter(|s| !s.is_empty() && *s != "*")
                .collect();
            let file_str = cand.id.file.to_string_lossy();
            for part in &hint_parts {
                if file_str.contains(part) {
                    score += 3;
                }
            }
        }
        if best.is_none() || score > best.unwrap().1 {
            best = Some((cand, score));
        }
    }

    best.map(|(exp, _)| exp)
}

/// Extract repo name from an ExportedSymbol's chunk_id.
/// chunk_id format: `{file}#{name}#{kind}` — no repo prefix in current format.
/// Falls back to extracting from the file path.
fn extract_repo_from_export(exp: &ExportedSymbol) -> String {
    // If chunk_id is present and contains a repo prefix, use it
    if let Some(ref chunk_id) = exp.chunk_id {
        // chunk_id format: file#name#kind
        // The "file" part might contain a repo prefix like "repo_name/src/..."
        let file_part = chunk_id.split('#').next().unwrap_or("");
        if let Some(repo) = file_part.split('/').next() {
            if !repo.is_empty() && repo != "src" && repo != "lib" {
                return repo.to_string();
            }
        }
    }
    // Fallback: use the first path component of the file
    exp.id
        .file
        .components()
        .next()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .unwrap_or_default()
}

// --- JSONL persistence ---

/// Write resolved imports as JSONL to `{store_root}/index/_xrefs/imports.jsonl`.
pub fn write_imports_jsonl(store_root: &Path, imports: &[ResolvedImport]) -> Result<Vec<u8>> {
    let dir = store_root.join("index").join("_xrefs");
    fs::create_dir_all(&dir)?;
    let path = dir.join("imports.jsonl");

    let mut content = if path.exists() {
        fs::read_to_string(&path)?
    } else {
        String::new()
    };

    for imp in imports {
        let line = serde_json::to_string(imp)
            .map_err(|e| engram_core::EngramError::Serialize(e.to_string()))?;
        content.push_str(&line);
        content.push('\n');
    }

    fs::write(&path, &content)?;
    Ok(content.into_bytes())
}

/// Read resolved imports from `{store_root}/index/_xrefs/imports.jsonl`.
pub fn read_imports_jsonl(store_root: &Path) -> Result<Vec<ResolvedImport>> {
    let path = store_root.join("index").join("_xrefs").join("imports.jsonl");
    if !path.exists() {
        return Ok(vec![]);
    }
    let content = fs::read_to_string(&path)?;
    let mut imports = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let imp: ResolvedImport = serde_json::from_str(trimmed)
            .map_err(|e| engram_core::EngramError::Serialize(e.to_string()))?;
        imports.push(imp);
    }
    Ok(imports)
}

// --- Helpers ---

fn node_text(node: &Node, source: &str) -> String {
    source[node.start_byte()..node.end_byte()].to_string()
}

/// Parse a Rust use path into (module_path, Vec<imported_names>).
/// Handles: `foo::bar`, `foo::{bar, baz}`, `foo::bar as alias`, `foo::*`.
fn parse_rust_use_path(path: &str) -> (String, Vec<String>) {
    let path = path.trim();

    // Handle tree list: `foo::{bar, baz}`
    if let Some(brace_start) = path.find('{') {
        let module = path[..brace_start].trim_end_matches("::").to_string();
        let brace_end = path.rfind('}').unwrap_or(path.len());
        let items = &path[brace_start + 1..brace_end];
        let names: Vec<String> = items
            .split(',')
            .map(|s| {
                let s = s.trim();
                // Handle `bar as alias`
                if let Some(alias_part) = s.split(" as ").nth(1) {
                    alias_part.trim().to_string()
                } else {
                    // Take the last segment after ::
                    s.rsplit("::").next().unwrap_or(s).trim().to_string()
                }
            })
            .filter(|s| !s.is_empty())
            .collect();
        (module, names)
    } else if path.ends_with("::*") {
        // Glob import: `foo::*`
        let module = path.trim_end_matches("::*").to_string();
        (module, vec!["*".to_string()])
    } else if path.contains(" as ") {
        // Aliased: `foo::bar as baz`
        let parts: Vec<&str> = path.splitn(2, " as ").collect();
        let full_path = parts[0].trim();
        let alias = parts[1].trim().to_string();
        let module = full_path
            .rsplit_once("::")
            .map(|(m, _)| m.to_string())
            .unwrap_or_default();
        (module, vec![alias])
    } else {
        // Simple: `foo::bar`
        let name = path.rsplit("::").next().unwrap_or(path).to_string();
        let module = path
            .rsplit_once("::")
            .map(|(m, _)| m.to_string())
            .unwrap_or_default();
        (module, vec![name])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::SymbolId;
    use std::path::PathBuf;

    // --- TypeScript Tests ---

    fn ts_imports(source: &str) -> Vec<RawImport> {
        let resolver = TreeSitterImportResolver::new();
        resolver.extract_imports_from_source(Path::new("test.ts"), source, Language::TypeScript)
    }

    #[test]
    fn ts_named_import() {
        let imports = ts_imports("import { foo, bar } from './utils';");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "./utils");
        assert!(imports[0].imported_names.contains(&"foo".to_string()));
        assert!(imports[0].imported_names.contains(&"bar".to_string()));
        assert_eq!(imports[0].line, 1);
    }

    #[test]
    fn ts_default_import() {
        let imports = ts_imports("import React from 'react';");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "react");
        assert!(imports[0].imported_names.contains(&"React".to_string()));
    }

    #[test]
    fn ts_namespace_import() {
        let imports = ts_imports("import * as fs from 'fs';");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "fs");
        assert!(imports[0].imported_names.contains(&"fs".to_string()));
    }

    #[test]
    fn ts_mixed_import() {
        let imports = ts_imports("import React, { useState, useEffect } from 'react';");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "react");
        assert!(imports[0].imported_names.contains(&"React".to_string()));
        assert!(imports[0].imported_names.contains(&"useState".to_string()));
        assert!(imports[0].imported_names.contains(&"useEffect".to_string()));
    }

    #[test]
    fn ts_aliased_import() {
        let imports = ts_imports("import { foo as bar } from './utils';");
        assert_eq!(imports.len(), 1);
        // The alias "bar" is the imported name
        assert!(imports[0].imported_names.contains(&"bar".to_string()));
    }

    #[test]
    fn ts_type_import() {
        let imports = ts_imports("import type { MyType } from './types';");
        assert_eq!(imports.len(), 1);
        assert!(imports[0].imported_names.contains(&"MyType".to_string()));
    }

    #[test]
    fn ts_side_effect_import_skipped() {
        let imports = ts_imports("import './styles.css';");
        assert!(imports.is_empty());
    }

    #[test]
    fn ts_multiple_imports() {
        let source = "import { a } from './a';\nimport { b } from './b';";
        let imports = ts_imports(source);
        assert_eq!(imports.len(), 2);
        assert_eq!(imports[0].line, 1);
        assert_eq!(imports[1].line, 2);
    }

    #[test]
    fn ts_no_imports() {
        let imports = ts_imports("export function greet() { return 'hi'; }");
        assert!(imports.is_empty());
    }

    #[test]
    fn ts_empty_source() {
        let imports = ts_imports("");
        assert!(imports.is_empty());
    }

    // --- Rust Tests ---

    fn rs_imports(source: &str) -> Vec<RawImport> {
        let resolver = TreeSitterImportResolver::new();
        resolver.extract_imports_from_source(Path::new("test.rs"), source, Language::Rust)
    }

    #[test]
    fn rust_simple_use() {
        let imports = rs_imports("use std::io::Read;");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "std::io");
        assert!(imports[0].imported_names.contains(&"Read".to_string()));
        assert_eq!(imports[0].line, 1);
    }

    #[test]
    fn rust_use_tree() {
        let imports = rs_imports("use std::collections::{HashMap, HashSet};");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "std::collections");
        assert!(imports[0].imported_names.contains(&"HashMap".to_string()));
        assert!(imports[0].imported_names.contains(&"HashSet".to_string()));
    }

    #[test]
    fn rust_use_glob() {
        let imports = rs_imports("use std::io::*;");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "std::io");
        assert!(imports[0].imported_names.contains(&"*".to_string()));
    }

    #[test]
    fn rust_use_alias() {
        let imports = rs_imports("use std::io::Result as IoResult;");
        assert_eq!(imports.len(), 1);
        assert!(imports[0].imported_names.contains(&"IoResult".to_string()));
    }

    #[test]
    fn rust_crate_use() {
        let imports = rs_imports("use crate::config::StoreConfig;");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "crate::config");
        assert!(imports[0].imported_names.contains(&"StoreConfig".to_string()));
    }

    #[test]
    fn rust_super_use() {
        let imports = rs_imports("use super::utils::helper;");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "super::utils");
        assert!(imports[0].imported_names.contains(&"helper".to_string()));
    }

    #[test]
    fn rust_multiple_use() {
        let source = "use std::io;\nuse std::fs;";
        let imports = rs_imports(source);
        assert_eq!(imports.len(), 2);
    }

    #[test]
    fn rust_no_imports() {
        let imports = rs_imports("pub fn main() {}");
        assert!(imports.is_empty());
    }

    // --- Python Tests ---

    fn py_imports(source: &str) -> Vec<RawImport> {
        let resolver = TreeSitterImportResolver::new();
        resolver.extract_imports_from_source(Path::new("test.py"), source, Language::Python)
    }

    #[test]
    fn python_import_module() {
        let imports = py_imports("import os");
        assert_eq!(imports.len(), 1);
        assert!(imports[0].imported_names.contains(&"os".to_string()));
        assert_eq!(imports[0].line, 1);
    }

    #[test]
    fn python_import_dotted() {
        let imports = py_imports("import os.path");
        assert_eq!(imports.len(), 1);
        assert!(imports[0].imported_names.contains(&"os".to_string()));
        assert_eq!(imports[0].module_path, "os.path");
    }

    #[test]
    fn python_from_import() {
        let imports = py_imports("from os.path import join, exists");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module_path, "os.path");
        assert!(imports[0].imported_names.contains(&"join".to_string()));
        assert!(imports[0].imported_names.contains(&"exists".to_string()));
    }

    #[test]
    fn python_from_import_alias() {
        let imports = py_imports("from collections import OrderedDict as OD");
        assert_eq!(imports.len(), 1);
        assert!(imports[0].imported_names.contains(&"OD".to_string()));
    }

    #[test]
    fn python_from_import_star() {
        let imports = py_imports("from os import *");
        assert_eq!(imports.len(), 1);
        assert!(imports[0].imported_names.contains(&"*".to_string()));
    }

    #[test]
    fn python_relative_import() {
        let imports = py_imports("from . import utils");
        assert_eq!(imports.len(), 1);
        assert!(imports[0].imported_names.contains(&"utils".to_string()));
    }

    #[test]
    fn python_import_alias() {
        let imports = py_imports("import numpy as np");
        assert_eq!(imports.len(), 1);
        assert!(imports[0].imported_names.contains(&"np".to_string()));
    }

    #[test]
    fn python_multiple_imports() {
        let source = "import os\nfrom sys import argv\nimport json";
        let imports = py_imports(source);
        assert_eq!(imports.len(), 3);
        assert_eq!(imports[0].line, 1);
        assert_eq!(imports[1].line, 2);
        assert_eq!(imports[2].line, 3);
    }

    #[test]
    fn python_no_imports() {
        let imports = py_imports("def main():\n    pass");
        assert!(imports.is_empty());
    }

    #[test]
    fn python_empty_source() {
        let imports = py_imports("");
        assert!(imports.is_empty());
    }

    // --- Resolution Tests ---

    fn make_test_export(name: &str, file: &str, kind: &str, chunk_id: Option<&str>) -> ExportedSymbol {
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

    #[test]
    fn resolve_matches_by_name() {
        let resolver = TreeSitterImportResolver::new();
        let exports = vec![
            make_test_export("Foo", "src/foo.ts", "class", Some("repo#src/foo.ts#Foo")),
        ];
        let raw = vec![RawImport {
            import_path: "import { Foo } from './foo'".to_string(),
            imported_names: vec!["Foo".to_string()],
            module_path: "./foo".to_string(),
            line: 1,
        }];

        let resolved = resolver.resolve_imports(&raw, &exports, "my-repo", Path::new("src/bar.ts"));
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].resolved_symbol.name, "Foo");
        assert_eq!(resolved[0].importing_repo, "my-repo");
        assert_eq!(resolved[0].resolution, "heuristic");
        assert_eq!(resolved[0].resolved_chunk, Some("repo#src/foo.ts#Foo".to_string()));
    }

    #[test]
    fn resolve_no_match_returns_empty() {
        let resolver = TreeSitterImportResolver::new();
        let exports = vec![
            make_test_export("Bar", "src/bar.ts", "class", None),
        ];
        let raw = vec![RawImport {
            import_path: "import { Foo } from './foo'".to_string(),
            imported_names: vec!["Foo".to_string()],
            module_path: "./foo".to_string(),
            line: 1,
        }];

        let resolved = resolver.resolve_imports(&raw, &exports, "my-repo", Path::new("src/baz.ts"));
        assert!(resolved.is_empty());
    }

    #[test]
    fn resolve_multiple_names_from_one_import() {
        let resolver = TreeSitterImportResolver::new();
        let exports = vec![
            make_test_export("Foo", "src/foo.ts", "class", None),
            make_test_export("Bar", "src/foo.ts", "class", None),
        ];
        let raw = vec![RawImport {
            import_path: "import { Foo, Bar } from './foo'".to_string(),
            imported_names: vec!["Foo".to_string(), "Bar".to_string()],
            module_path: "./foo".to_string(),
            line: 1,
        }];

        let resolved = resolver.resolve_imports(&raw, &exports, "my-repo", Path::new("src/baz.ts"));
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn resolve_cross_repo_fields_populated() {
        let resolver = TreeSitterImportResolver::new();
        let exports = vec![
            make_test_export("Config", "other-repo/src/config.ts", "type", Some("other-repo/src/config.ts#Config#type")),
        ];
        let raw = vec![RawImport {
            import_path: "import { Config } from '@other/config'".to_string(),
            imported_names: vec!["Config".to_string()],
            module_path: "@other/config".to_string(),
            line: 5,
        }];

        let resolved = resolver.resolve_imports(&raw, &exports, "my-repo", Path::new("src/app.ts"));
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].importing_repo, "my-repo");
        assert_eq!(resolved[0].source_file, PathBuf::from("other-repo/src/config.ts"));
        assert_eq!(resolved[0].line, 5);
        assert_eq!(resolved[0].importing_file, PathBuf::from("src/app.ts"));
    }

    // --- JSONL Tests ---

    #[test]
    fn write_and_read_imports_jsonl() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = tmp.path();

        let imports = vec![
            ResolvedImport {
                import_path: "import { Foo } from './foo'".to_string(),
                resolved_symbol: SymbolId {
                    file: PathBuf::from("src/foo.ts"),
                    name: "Foo".to_string(),
                    kind: "class".to_string(),
                },
                importing_file: PathBuf::from("src/bar.ts"),
                line: 1,
                importing_repo: "my-repo".to_string(),
                source_repo: "my-repo".to_string(),
                source_file: PathBuf::from("src/foo.ts"),
                resolved_chunk: Some("my-repo#src/foo.ts#Foo".to_string()),
                resolution: "heuristic".to_string(),
            },
        ];

        write_imports_jsonl(store, &imports).unwrap();
        let read_back = read_imports_jsonl(store).unwrap();
        assert_eq!(read_back.len(), 1);
        assert_eq!(read_back[0].resolved_symbol.name, "Foo");
        assert_eq!(read_back[0].importing_repo, "my-repo");
        assert_eq!(read_back[0].resolution, "heuristic");
    }

    #[test]
    fn read_imports_empty_store() {
        let tmp = tempfile::TempDir::new().unwrap();
        let imports = read_imports_jsonl(tmp.path()).unwrap();
        assert!(imports.is_empty());
    }

    #[test]
    fn imports_jsonl_path_correct() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = tmp.path();
        let imports = vec![ResolvedImport {
            import_path: "use std::io".to_string(),
            resolved_symbol: SymbolId {
                file: PathBuf::from("std/io.rs"),
                name: "io".to_string(),
                kind: "module".to_string(),
            },
            importing_file: PathBuf::from("src/main.rs"),
            line: 1,
            importing_repo: String::new(),
            source_repo: String::new(),
            source_file: PathBuf::new(),
            resolved_chunk: None,
            resolution: "heuristic".to_string(),
        }];

        write_imports_jsonl(store, &imports).unwrap();
        assert!(store.join("index/_xrefs/imports.jsonl").exists());
    }

    // --- Rust use path parser tests ---

    #[test]
    fn parse_rust_simple_path() {
        let (module, names) = parse_rust_use_path("std::io::Read");
        assert_eq!(module, "std::io");
        assert_eq!(names, vec!["Read"]);
    }

    #[test]
    fn parse_rust_tree_path() {
        let (module, names) = parse_rust_use_path("std::collections::{HashMap, HashSet}");
        assert_eq!(module, "std::collections");
        assert!(names.contains(&"HashMap".to_string()));
        assert!(names.contains(&"HashSet".to_string()));
    }

    #[test]
    fn parse_rust_glob_path() {
        let (module, names) = parse_rust_use_path("std::io::*");
        assert_eq!(module, "std::io");
        assert_eq!(names, vec!["*"]);
    }

    #[test]
    fn parse_rust_alias_path() {
        let (module, names) = parse_rust_use_path("std::io::Result as IoResult");
        assert_eq!(module, "std::io");
        assert_eq!(names, vec!["IoResult"]);
    }
}
