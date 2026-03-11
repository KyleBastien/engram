use engram_core::{ExportedSymbol, Result, SymbolId};
use std::fs;
use std::path::Path;
use tree_sitter::{Node, Parser};

use crate::chunker::Language;
use crate::detect::{detect_language, ChunkerKind};

/// Extracts exported symbols from source files using tree-sitter AST parsing.
pub struct TreeSitterExportExtractor;

impl TreeSitterExportExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract exports from a single file given its source and language.
    pub fn extract_from_source(
        &self,
        file: &Path,
        source: &str,
        language: Language,
    ) -> Vec<ExportedSymbol> {
        let mut parser = Parser::new();
        let ts_language: tree_sitter::Language = match language {
            Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::Python => tree_sitter_python::LANGUAGE.into(),
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
            Language::TypeScript => self.extract_ts_exports(&root, source, file),
            Language::Rust => self.extract_rust_exports(&root, source, file),
            Language::Python => self.extract_python_exports(&root, source, file),
        }
    }

    /// Extract exports from a file on disk.
    pub fn extract_from_file(&self, file: &Path) -> Result<Vec<ExportedSymbol>> {
        let source = fs::read_to_string(file)?;
        let lang = match detect_language(file) {
            ChunkerKind::TreeSitter(lang) => lang,
            _ => return Ok(vec![]),
        };
        Ok(self.extract_from_source(file, &source, lang))
    }

    // --- TypeScript ---

    fn extract_ts_exports(
        &self,
        root: &Node,
        source: &str,
        file: &Path,
    ) -> Vec<ExportedSymbol> {
        let mut exports = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            if child.kind() == "export_statement" {
                self.extract_ts_export_statement(&child, source, file, &mut exports);
            }
        }

        exports
    }

    fn extract_ts_export_statement(
        &self,
        node: &Node,
        source: &str,
        file: &Path,
        exports: &mut Vec<ExportedSymbol>,
    ) {
        let line = node.start_position().row + 1;
        let mut cursor = node.walk();

        // Check for "export default"
        let is_default = node_text(node, source).starts_with("export default");

        for child in node.children(&mut cursor) {
            match child.kind() {
                "function_declaration" => {
                    let name = if is_default {
                        field_text(&child, "name", source).unwrap_or_else(|| "default".to_string())
                    } else {
                        match field_text(&child, "name", source) {
                            Some(n) => n,
                            None => continue,
                        }
                    };
                    exports.push(make_export(file, &name, "function", line, true, &child, source));
                }
                "class_declaration" => {
                    let name = if is_default {
                        field_text(&child, "name", source).unwrap_or_else(|| "default".to_string())
                    } else {
                        match field_text(&child, "name", source) {
                            Some(n) => n,
                            None => continue,
                        }
                    };
                    exports.push(make_export(file, &name, "class", line, true, &child, source));
                }
                "type_alias_declaration" => {
                    if let Some(name) = field_text(&child, "name", source) {
                        exports.push(make_export(file, &name, "type", line, true, &child, source));
                    }
                }
                "interface_declaration" => {
                    if let Some(name) = field_text(&child, "name", source) {
                        exports.push(make_export(
                            file, &name, "interface", line, true, &child, source,
                        ));
                    }
                }
                "enum_declaration" => {
                    if let Some(name) = field_text(&child, "name", source) {
                        exports.push(make_export(file, &name, "enum", line, true, &child, source));
                    }
                }
                "lexical_declaration" | "variable_declaration" => {
                    // export const foo = ...
                    self.extract_ts_variable_exports(&child, source, file, line, exports);
                }
                "export_clause" => {
                    // export { a, b, c } or export { a as b }
                    self.extract_ts_named_exports(&child, node, source, file, line, exports);
                }
                _ => {}
            }
        }

        // Handle re-exports: export { ... } from '...' or export * from '...'
        // Already handled by export_clause above for named re-exports
        // export * from '...' has source child but no export_clause
        if node_text(node, source).contains("export *") {
            exports.push(ExportedSymbol {
                id: SymbolId {
                    file: file.to_path_buf(),
                    name: "*".to_string(),
                    kind: "re-export".to_string(),
                },
                line,
                is_public: true,
                doc: None,
                chunk_id: None,
            });
        }
    }

    fn extract_ts_variable_exports(
        &self,
        node: &Node,
        source: &str,
        file: &Path,
        line: usize,
        exports: &mut Vec<ExportedSymbol>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "variable_declarator" {
                if let Some(name) = field_text(&child, "name", source) {
                    let kind = if child
                        .child_by_field_name("value")
                        .is_some_and(|v| v.kind() == "arrow_function")
                    {
                        "function"
                    } else {
                        "variable"
                    };
                    exports.push(make_export(file, &name, kind, line, true, &child, source));
                }
            }
        }
    }

    fn extract_ts_named_exports(
        &self,
        clause: &Node,
        export_node: &Node,
        source: &str,
        file: &Path,
        line: usize,
        exports: &mut Vec<ExportedSymbol>,
    ) {
        // Check if this is a re-export (has "from" source)
        let is_reexport = has_source_clause(export_node, source);
        let kind = if is_reexport { "re-export" } else { "variable" };

        let mut cursor = clause.walk();
        for child in clause.children(&mut cursor) {
            if child.kind() == "export_specifier" {
                // The exported name is the "alias" if present, else the "name"
                let exported_name = field_text(&child, "alias", source)
                    .or_else(|| field_text(&child, "name", source));
                if let Some(name) = exported_name {
                    exports.push(ExportedSymbol {
                        id: SymbolId {
                            file: file.to_path_buf(),
                            name,
                            kind: kind.to_string(),
                        },
                        line,
                        is_public: true,
                        doc: None,
                        chunk_id: None,
                    });
                }
            }
        }
    }

    // --- Rust ---

    fn extract_rust_exports(
        &self,
        root: &Node,
        source: &str,
        file: &Path,
    ) -> Vec<ExportedSymbol> {
        let mut exports = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            self.extract_rust_pub_item(&child, source, file, &mut exports);
        }

        exports
    }

    fn extract_rust_pub_item(
        &self,
        node: &Node,
        source: &str,
        file: &Path,
        exports: &mut Vec<ExportedSymbol>,
    ) {
        let text = node_text(node, source);
        let line = node.start_position().row + 1;

        match node.kind() {
            "function_item" => {
                if has_pub_visibility(node) {
                    if let Some(name) = field_text(node, "name", source) {
                        exports.push(make_export(file, &name, "function", line, true, node, source));
                    }
                }
            }
            "struct_item" => {
                if has_pub_visibility(node) {
                    if let Some(name) = field_text(node, "name", source) {
                        exports.push(make_export(file, &name, "struct", line, true, node, source));
                    }
                }
            }
            "enum_item" => {
                if has_pub_visibility(node) {
                    if let Some(name) = field_text(node, "name", source) {
                        exports.push(make_export(file, &name, "enum", line, true, node, source));
                    }
                }
            }
            "trait_item" => {
                if has_pub_visibility(node) {
                    if let Some(name) = field_text(node, "name", source) {
                        exports.push(make_export(file, &name, "trait", line, true, node, source));
                    }
                }
            }
            "type_item" => {
                if has_pub_visibility(node) {
                    if let Some(name) = field_text(node, "name", source) {
                        exports.push(make_export(file, &name, "type", line, true, node, source));
                    }
                }
            }
            "const_item" => {
                if has_pub_visibility(node) {
                    if let Some(name) = field_text(node, "name", source) {
                        exports.push(make_export(file, &name, "const", line, true, node, source));
                    }
                }
            }
            "static_item" => {
                if has_pub_visibility(node) {
                    if let Some(name) = field_text(node, "name", source) {
                        exports.push(make_export(file, &name, "static", line, true, node, source));
                    }
                }
            }
            "mod_item" => {
                if has_pub_visibility(node) {
                    if let Some(name) = field_text(node, "name", source) {
                        exports.push(make_export(file, &name, "module", line, true, node, source));
                    }
                }
            }
            "macro_definition" => {
                // macro_rules! are always public if exported via pub use or #[macro_export]
                if text.contains("#[macro_export]") || has_pub_visibility(node) {
                    if let Some(name) = field_text(node, "name", source) {
                        exports.push(make_export(file, &name, "macro", line, true, node, source));
                    }
                }
            }
            "use_declaration" => {
                // pub use ... re-exports
                if has_pub_visibility(node) {
                    let use_text = text.trim_start_matches("pub ").trim_start_matches("use ");
                    let use_text = use_text.trim_end_matches(';').trim();
                    // Extract the last segment as the name
                    let name = use_text
                        .rsplit("::")
                        .next()
                        .unwrap_or(use_text)
                        .trim()
                        .to_string();
                    exports.push(ExportedSymbol {
                        id: SymbolId {
                            file: file.to_path_buf(),
                            name,
                            kind: "re-export".to_string(),
                        },
                        line,
                        is_public: true,
                        doc: None,
                        chunk_id: None,
                    });
                }
            }
            _ => {}
        }
    }

    // --- Python ---

    fn extract_python_exports(
        &self,
        root: &Node,
        source: &str,
        file: &Path,
    ) -> Vec<ExportedSymbol> {
        let mut exports = Vec::new();
        let mut cursor = root.walk();

        // First pass: look for __all__
        let mut has_all = false;
        for child in root.children(&mut cursor) {
            if let Some(all_names) = self.extract_python_all(&child, source) {
                has_all = true;
                for name in all_names {
                    exports.push(ExportedSymbol {
                        id: SymbolId {
                            file: file.to_path_buf(),
                            name,
                            kind: "variable".to_string(),
                        },
                        line: child.start_position().row + 1,
                        is_public: true,
                        doc: None,
                        chunk_id: None,
                    });
                }
            }
        }

        // If no __all__, extract top-level definitions
        if !has_all {
            let mut cursor2 = root.walk();
            for child in root.children(&mut cursor2) {
                self.extract_python_top_level_def(&child, source, file, &mut exports);
            }
        }

        exports
    }

    fn extract_python_all(&self, node: &Node, source: &str) -> Option<Vec<String>> {
        // Look for: __all__ = ["a", "b", "c"]
        if node.kind() == "expression_statement" {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "assignment" {
                    let left = child.child_by_field_name("left")?;
                    let left_text = node_text(&left, source);
                    if left_text == "__all__" {
                        let right = child.child_by_field_name("right")?;
                        return Some(extract_python_list_strings(&right, source));
                    }
                }
            }
        }
        None
    }

    fn extract_python_top_level_def(
        &self,
        node: &Node,
        source: &str,
        file: &Path,
        exports: &mut Vec<ExportedSymbol>,
    ) {
        let line = node.start_position().row + 1;

        match node.kind() {
            "function_definition" => {
                if let Some(name) = field_text(node, "name", source) {
                    if !name.starts_with('_') {
                        exports.push(make_export(file, &name, "function", line, true, node, source));
                    }
                }
            }
            "class_definition" => {
                if let Some(name) = field_text(node, "name", source) {
                    if !name.starts_with('_') {
                        exports.push(make_export(file, &name, "class", line, true, node, source));
                    }
                }
            }
            "decorated_definition" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "function_definition" => {
                            if let Some(name) = field_text(&child, "name", source) {
                                if !name.starts_with('_') {
                                    exports.push(make_export(
                                        file, &name, "function", line, true, node, source,
                                    ));
                                }
                            }
                        }
                        "class_definition" => {
                            if let Some(name) = field_text(&child, "name", source) {
                                if !name.starts_with('_') {
                                    exports.push(make_export(
                                        file, &name, "class", line, true, node, source,
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            "expression_statement" => {
                // Top-level assignments like FOO = "bar" (module-level constants)
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "assignment" {
                        if let Some(left) = child.child_by_field_name("left") {
                            let name = node_text(&left, source);
                            if left.kind() == "identifier" && !name.starts_with('_') {
                                exports.push(ExportedSymbol {
                                    id: SymbolId {
                                        file: file.to_path_buf(),
                                        name,
                                        kind: "variable".to_string(),
                                    },
                                    line,
                                    is_public: true,
                                    doc: None,
                                    chunk_id: None,
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

impl Default for TreeSitterExportExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Write exported symbols as JSONL to `{store_root}/index/_xrefs/exports.jsonl`.
/// Appends to any existing file content.
pub fn write_exports_jsonl(store_root: &Path, symbols: &[ExportedSymbol]) -> Result<Vec<u8>> {
    let dir = store_root.join("index").join("_xrefs");
    fs::create_dir_all(&dir)?;
    let path = dir.join("exports.jsonl");

    let mut content = if path.exists() {
        fs::read_to_string(&path)?
    } else {
        String::new()
    };

    for sym in symbols {
        let line = serde_json::to_string(sym)
            .map_err(|e| engram_core::EngramError::Serialize(e.to_string()))?;
        content.push_str(&line);
        content.push('\n');
    }

    fs::write(&path, &content)?;
    Ok(content.into_bytes())
}

/// Read exported symbols from `{store_root}/index/_xrefs/exports.jsonl`.
pub fn read_exports_jsonl(store_root: &Path) -> Result<Vec<ExportedSymbol>> {
    let path = store_root.join("index").join("_xrefs").join("exports.jsonl");
    if !path.exists() {
        return Ok(vec![]);
    }
    let content = fs::read_to_string(&path)?;
    let mut symbols = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let sym: ExportedSymbol = serde_json::from_str(trimmed)
            .map_err(|e| engram_core::EngramError::Serialize(e.to_string()))?;
        symbols.push(sym);
    }
    Ok(symbols)
}

// --- Helpers ---

fn make_export(
    file: &Path,
    name: &str,
    kind: &str,
    line: usize,
    is_public: bool,
    node: &Node,
    source: &str,
) -> ExportedSymbol {
    let chunk_id = format!("{}#{}#{}", file.display(), name, kind);
    let doc = extract_doc_comment(node, source);
    ExportedSymbol {
        id: SymbolId {
            file: file.to_path_buf(),
            name: name.to_string(),
            kind: kind.to_string(),
        },
        line,
        is_public,
        doc,
        chunk_id: Some(chunk_id),
    }
}

fn field_text(node: &Node, field: &str, source: &str) -> Option<String> {
    let child = node.child_by_field_name(field)?;
    Some(source[child.start_byte()..child.end_byte()].to_string())
}

fn node_text(node: &Node, source: &str) -> String {
    source[node.start_byte()..node.end_byte()].to_string()
}

fn has_pub_visibility(node: &Node) -> bool {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "visibility_modifier" {
            return true;
        }
    }
    false
}

fn has_source_clause(export_node: &Node, source: &str) -> bool {
    let text = node_text(export_node, source);
    text.contains(" from ")
}

fn extract_python_list_strings(node: &Node, source: &str) -> Vec<String> {
    let mut names = Vec::new();
    if node.kind() == "list" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "string" {
                let text = node_text(&child, source);
                // Strip quotes
                let stripped = text
                    .trim_start_matches(['"', '\''])
                    .trim_end_matches(['"', '\''])
                    .to_string();
                if !stripped.is_empty() {
                    names.push(stripped);
                }
            }
        }
    }
    names
}

fn extract_doc_comment(node: &Node, source: &str) -> Option<String> {
    // Look at the previous sibling for a comment node
    if let Some(prev) = node.prev_sibling() {
        if prev.kind() == "comment" || prev.kind() == "line_comment" || prev.kind() == "block_comment" {
            let text = node_text(&prev, source).trim().to_string();
            if !text.is_empty() {
                return Some(text);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // --- TypeScript Tests ---

    fn ts_exports(source: &str) -> Vec<ExportedSymbol> {
        let extractor = TreeSitterExportExtractor::new();
        extractor.extract_from_source(Path::new("test.ts"), source, Language::TypeScript)
    }

    #[test]
    fn ts_export_function_declaration() {
        let exports = ts_exports("export function greet(name: string): string { return name; }");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "greet");
        assert_eq!(exports[0].id.kind, "function");
        assert!(exports[0].is_public);
        assert!(exports[0].chunk_id.is_some());
    }

    #[test]
    fn ts_export_class() {
        let exports = ts_exports("export class Calculator { add(n: number) {} }");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Calculator");
        assert_eq!(exports[0].id.kind, "class");
    }

    #[test]
    fn ts_export_type_alias() {
        let exports = ts_exports("export type Point = { x: number; y: number; };");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Point");
        assert_eq!(exports[0].id.kind, "type");
    }

    #[test]
    fn ts_export_interface() {
        let exports = ts_exports("export interface Shape { area(): number; }");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Shape");
        assert_eq!(exports[0].id.kind, "interface");
    }

    #[test]
    fn ts_export_enum() {
        let exports = ts_exports("export enum Color { Red, Green, Blue }");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Color");
        assert_eq!(exports[0].id.kind, "enum");
    }

    #[test]
    fn ts_export_const() {
        let exports = ts_exports("export const VERSION = '1.0';");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "VERSION");
        assert_eq!(exports[0].id.kind, "variable");
    }

    #[test]
    fn ts_export_arrow_function() {
        let exports = ts_exports("export const add = (a: number, b: number) => a + b;");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "add");
        assert_eq!(exports[0].id.kind, "function");
    }

    #[test]
    fn ts_named_exports() {
        let source = "const a = 1;\nconst b = 2;\nexport { a, b };";
        let exports = ts_exports(source);
        assert_eq!(exports.len(), 2);
        assert_eq!(exports[0].id.name, "a");
        assert_eq!(exports[1].id.name, "b");
    }

    #[test]
    fn ts_re_export_from() {
        let source = "export { foo, bar } from './other';";
        let exports = ts_exports(source);
        assert_eq!(exports.len(), 2);
        assert_eq!(exports[0].id.name, "foo");
        assert_eq!(exports[0].id.kind, "re-export");
        assert_eq!(exports[1].id.name, "bar");
    }

    #[test]
    fn ts_re_export_star() {
        let source = "export * from './utils';";
        let exports = ts_exports(source);
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "*");
        assert_eq!(exports[0].id.kind, "re-export");
    }

    #[test]
    fn ts_export_default_function() {
        let source = "export default function main() { return 'hello'; }";
        let exports = ts_exports(source);
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "main");
        assert_eq!(exports[0].id.kind, "function");
    }

    #[test]
    fn ts_no_exports_returns_empty() {
        let exports = ts_exports("function internal() { return 1; }");
        assert!(exports.is_empty());
    }

    #[test]
    fn ts_multiple_exports() {
        let source = r#"export function foo() {}
export class Bar {}
export type Baz = string;
export const QUX = 42;"#;
        let exports = ts_exports(source);
        assert_eq!(exports.len(), 4);
        let names: Vec<&str> = exports.iter().map(|e| e.id.name.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"Bar"));
        assert!(names.contains(&"Baz"));
        assert!(names.contains(&"QUX"));
    }

    // --- Rust Tests ---

    fn rs_exports(source: &str) -> Vec<ExportedSymbol> {
        let extractor = TreeSitterExportExtractor::new();
        extractor.extract_from_source(Path::new("test.rs"), source, Language::Rust)
    }

    #[test]
    fn rust_pub_function() {
        let exports = rs_exports("pub fn greet(name: &str) -> String { format!(\"Hi {}\", name) }");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "greet");
        assert_eq!(exports[0].id.kind, "function");
        assert!(exports[0].is_public);
    }

    #[test]
    fn rust_pub_struct() {
        let exports = rs_exports("pub struct Point {\n    pub x: f64,\n    pub y: f64,\n}");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Point");
        assert_eq!(exports[0].id.kind, "struct");
    }

    #[test]
    fn rust_pub_enum() {
        let exports = rs_exports("pub enum Color { Red, Green, Blue }");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Color");
        assert_eq!(exports[0].id.kind, "enum");
    }

    #[test]
    fn rust_pub_trait() {
        let exports = rs_exports("pub trait Drawable { fn draw(&self); }");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Drawable");
        assert_eq!(exports[0].id.kind, "trait");
    }

    #[test]
    fn rust_pub_type_alias() {
        let exports = rs_exports("pub type Result<T> = std::result::Result<T, MyError>;");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Result");
        assert_eq!(exports[0].id.kind, "type");
    }

    #[test]
    fn rust_pub_const() {
        let exports = rs_exports("pub const MAX_SIZE: usize = 1024;");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "MAX_SIZE");
        assert_eq!(exports[0].id.kind, "const");
    }

    #[test]
    fn rust_pub_mod() {
        let exports = rs_exports("pub mod utils { pub fn helper() {} }");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "utils");
        assert_eq!(exports[0].id.kind, "module");
    }

    #[test]
    fn rust_private_items_excluded() {
        let source = "fn private_fn() {}\nstruct InternalStruct {}\npub fn public_fn() {}";
        let exports = rs_exports(source);
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "public_fn");
    }

    #[test]
    fn rust_pub_use_reexport() {
        let exports = rs_exports("pub use crate::error::EngramError;");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "EngramError");
        assert_eq!(exports[0].id.kind, "re-export");
    }

    #[test]
    fn rust_multiple_pub_items() {
        let source = r#"pub fn foo() {}
pub struct Bar {}
fn private() {}
pub enum Baz { A, B }
pub const C: i32 = 0;"#;
        let exports = rs_exports(source);
        assert_eq!(exports.len(), 4);
        let names: Vec<&str> = exports.iter().map(|e| e.id.name.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"Bar"));
        assert!(names.contains(&"Baz"));
        assert!(names.contains(&"C"));
    }

    #[test]
    fn rust_chunk_id_format() {
        let exports = rs_exports("pub fn greet() {}");
        assert_eq!(exports[0].chunk_id, Some("test.rs#greet#function".to_string()));
    }

    // --- Python Tests ---

    fn py_exports(source: &str) -> Vec<ExportedSymbol> {
        let extractor = TreeSitterExportExtractor::new();
        extractor.extract_from_source(Path::new("test.py"), source, Language::Python)
    }

    #[test]
    fn python_top_level_function() {
        let exports = py_exports("def greet(name):\n    return f'Hello, {name}!'");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "greet");
        assert_eq!(exports[0].id.kind, "function");
    }

    #[test]
    fn python_top_level_class() {
        let exports = py_exports("class Calculator:\n    def add(self, n):\n        pass");
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Calculator");
        assert_eq!(exports[0].id.kind, "class");
    }

    #[test]
    fn python_private_excluded() {
        let source = "def public():\n    pass\n\ndef _private():\n    pass";
        let exports = py_exports(source);
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "public");
    }

    #[test]
    fn python_all_takes_precedence() {
        let source = r#"__all__ = ["foo", "bar"]

def foo():
    pass

def bar():
    pass

def baz():
    pass"#;
        let exports = py_exports(source);
        assert_eq!(exports.len(), 2);
        let names: Vec<&str> = exports.iter().map(|e| e.id.name.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"bar"));
        assert!(!names.contains(&"baz"));
    }

    #[test]
    fn python_decorated_function() {
        let source = "@staticmethod\ndef helper():\n    return 42";
        let exports = py_exports(source);
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "helper");
        assert_eq!(exports[0].id.kind, "function");
    }

    #[test]
    fn python_decorated_class() {
        let source = "@dataclass\nclass Point:\n    x: float\n    y: float";
        let exports = py_exports(source);
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "Point");
        assert_eq!(exports[0].id.kind, "class");
    }

    #[test]
    fn python_top_level_assignment() {
        let source = "VERSION = '1.0'\nMAX_SIZE = 100\n\ndef main():\n    pass";
        let exports = py_exports(source);
        assert_eq!(exports.len(), 3);
        let names: Vec<&str> = exports.iter().map(|e| e.id.name.as_str()).collect();
        assert!(names.contains(&"VERSION"));
        assert!(names.contains(&"MAX_SIZE"));
        assert!(names.contains(&"main"));
    }

    #[test]
    fn python_empty_source() {
        let exports = py_exports("");
        assert!(exports.is_empty());
    }

    #[test]
    fn python_underscore_assignment_excluded() {
        let source = "_internal = 42\nPUBLIC = 100";
        let exports = py_exports(source);
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].id.name, "PUBLIC");
    }

    // --- JSONL Tests ---

    #[test]
    fn write_and_read_exports_jsonl() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = tmp.path();
        fs::create_dir_all(store.join("index")).unwrap();

        let symbols = vec![
            ExportedSymbol {
                id: SymbolId {
                    file: PathBuf::from("src/lib.rs"),
                    name: "foo".to_string(),
                    kind: "function".to_string(),
                },
                line: 1,
                is_public: true,
                doc: None,
                chunk_id: Some("repo#src/lib.rs#foo".to_string()),
            },
            ExportedSymbol {
                id: SymbolId {
                    file: PathBuf::from("src/lib.rs"),
                    name: "Bar".to_string(),
                    kind: "struct".to_string(),
                },
                line: 5,
                is_public: true,
                doc: None,
                chunk_id: Some("repo#src/lib.rs#Bar".to_string()),
            },
        ];

        write_exports_jsonl(store, &symbols).unwrap();
        let read_back = read_exports_jsonl(store).unwrap();
        assert_eq!(read_back.len(), 2);
        assert_eq!(read_back[0].id.name, "foo");
        assert_eq!(read_back[1].id.name, "Bar");
    }

    #[test]
    fn read_exports_empty_store() {
        let tmp = tempfile::TempDir::new().unwrap();
        let exports = read_exports_jsonl(tmp.path()).unwrap();
        assert!(exports.is_empty());
    }

    #[test]
    fn exports_jsonl_path_correct() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = tmp.path();
        let symbols = vec![ExportedSymbol {
            id: SymbolId {
                file: PathBuf::from("test.ts"),
                name: "x".to_string(),
                kind: "variable".to_string(),
            },
            line: 1,
            is_public: true,
            doc: None,
            chunk_id: None,
        }];

        write_exports_jsonl(store, &symbols).unwrap();
        assert!(store.join("index/_xrefs/exports.jsonl").exists());
    }

    #[test]
    fn exports_file_and_chunk_id_populated() {
        let exports = ts_exports("export function foo() { return 1; }");
        assert_eq!(exports[0].id.file, PathBuf::from("test.ts"));
        assert_eq!(exports[0].chunk_id, Some("test.ts#foo#function".to_string()));
    }
}
