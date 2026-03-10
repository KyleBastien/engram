use engram_core::ChunkKind;
use std::path::Path;
use tree_sitter::{Node, Parser};

/// A raw chunk extracted from source code by tree-sitter parsing.
#[derive(Debug, Clone, PartialEq)]
pub struct RawChunk {
    pub kind: ChunkKind,
    pub name: String,
    pub signature: Option<String>,
    pub start_line: u32,
    pub end_line: u32,
    pub content: String,
}

/// Supported languages for tree-sitter chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    TypeScript,
    Rust,
}

/// Extracts semantic chunks from source code using tree-sitter AST parsing.
pub struct TreeSitterChunker;

impl TreeSitterChunker {
    pub fn new() -> Self {
        Self
    }

    /// Parse source code and extract semantic chunks.
    pub fn chunk_file(&self, _path: &Path, source: &str, language: Language) -> Vec<RawChunk> {
        let mut parser = Parser::new();
        let ts_language: tree_sitter::Language = match language {
            Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        };
        parser
            .set_language(&ts_language)
            .expect("failed to set language");

        let tree = match parser.parse(source, None) {
            Some(tree) => tree,
            None => return vec![],
        };

        let root = tree.root_node();
        let mut chunks = Vec::new();
        // Track accumulated module-level (non-declaration) content
        let mut mod_start: Option<(usize, u32)> = None; // (start_byte, start_line)
        let mut mod_end: (usize, u32) = (0, 1); // (end_byte, end_line)
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            let extracted = match language {
                Language::TypeScript => self.extract_ts_node(&child, source),
                Language::Rust => self.extract_rust_node(&child, source),
            };
            if let Some(chunk) = extracted {
                // Flush accumulated module content before this declaration
                if let Some((start_byte, start_line)) = mod_start.take() {
                    let content = source[start_byte..child.start_byte()].trim();
                    if !content.is_empty() {
                        chunks.push(RawChunk {
                            kind: ChunkKind::Module,
                            name: "module".to_string(),
                            signature: None,
                            start_line,
                            end_line: mod_end.1,
                            content: content.to_string(),
                        });
                    }
                }
                chunks.push(chunk);
            } else {
                let text = &source[child.start_byte()..child.end_byte()];
                if !text.trim().is_empty() {
                    if mod_start.is_none() {
                        mod_start = Some((child.start_byte(), node_start_line(&child)));
                    }
                    mod_end = (child.end_byte(), node_end_line(&child));
                }
            }
        }

        // Flush remaining module content
        if let Some((start_byte, start_line)) = mod_start {
            let content = source[start_byte..mod_end.0].trim();
            if !content.is_empty() {
                chunks.push(RawChunk {
                    kind: ChunkKind::Module,
                    name: "module".to_string(),
                    signature: None,
                    start_line,
                    end_line: mod_end.1,
                    content: content.to_string(),
                });
            }
        }

        chunks
    }

    fn extract_ts_node(&self, node: &Node, source: &str) -> Option<RawChunk> {
        match node.kind() {
            "function_declaration" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Function, name, node, source))
            }
            "class_declaration" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Class, name, node, source))
            }
            "method_definition" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Method, name, node, source))
            }
            "type_alias_declaration" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Type, name, node, source))
            }
            "interface_declaration" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Type, name, node, source))
            }
            "enum_declaration" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Type, name, node, source))
            }
            "lexical_declaration" | "variable_declaration" => {
                extract_arrow_from_declaration(node, source)
            }
            "export_statement" => self.extract_export(node, source),
            _ => None,
        }
    }

    fn extract_export(&self, node: &Node, source: &str) -> Option<RawChunk> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "function_declaration" => {
                    let name = field_text(&child, "name", source)?;
                    return Some(make_chunk(ChunkKind::Function, name, node, source));
                }
                "class_declaration" => {
                    let name = field_text(&child, "name", source)?;
                    return Some(make_chunk(ChunkKind::Class, name, node, source));
                }
                "type_alias_declaration" => {
                    let name = field_text(&child, "name", source)?;
                    return Some(make_chunk(ChunkKind::Type, name, node, source));
                }
                "interface_declaration" => {
                    let name = field_text(&child, "name", source)?;
                    return Some(make_chunk(ChunkKind::Type, name, node, source));
                }
                "enum_declaration" => {
                    let name = field_text(&child, "name", source)?;
                    return Some(make_chunk(ChunkKind::Type, name, node, source));
                }
                "lexical_declaration" | "variable_declaration" => {
                    if let Some(mut chunk) = extract_arrow_from_declaration(&child, source) {
                        // Use the export_statement's full range
                        chunk.start_line = node_start_line(node);
                        chunk.end_line = node_end_line(node);
                        chunk.content = node_text(node, source);
                        chunk.signature = extract_signature(node, source);
                        return Some(chunk);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn extract_rust_node(&self, node: &Node, source: &str) -> Option<RawChunk> {
        match node.kind() {
            "function_item" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Function, name, node, source))
            }
            "struct_item" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Type, name, node, source))
            }
            "enum_item" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Type, name, node, source))
            }
            "impl_item" => {
                let name = rust_impl_name(node, source)?;
                Some(make_chunk(ChunkKind::Impl, name, node, source))
            }
            "trait_item" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Type, name, node, source))
            }
            "type_item" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Type, name, node, source))
            }
            "mod_item" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Module, name, node, source))
            }
            "const_item" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Other, name, node, source))
            }
            "static_item" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Other, name, node, source))
            }
            "macro_definition" => {
                let name = field_text(node, "name", source)?;
                Some(make_chunk(ChunkKind::Function, name, node, source))
            }
            _ => None,
        }
    }
}

impl Default for TreeSitterChunker {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_arrow_from_declaration(node: &Node, source: &str) -> Option<RawChunk> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "variable_declarator" {
            if let Some(value) = child.child_by_field_name("value") {
                if value.kind() == "arrow_function" {
                    let name = field_text(&child, "name", source)?;
                    return Some(make_chunk(ChunkKind::Function, name, node, source));
                }
            }
        }
    }
    None
}

fn make_chunk(kind: ChunkKind, name: String, node: &Node, source: &str) -> RawChunk {
    RawChunk {
        kind,
        name,
        signature: extract_signature(node, source),
        start_line: node_start_line(node),
        end_line: node_end_line(node),
        content: node_text(node, source),
    }
}

fn field_text(node: &Node, field: &str, source: &str) -> Option<String> {
    let child = node.child_by_field_name(field)?;
    Some(source[child.start_byte()..child.end_byte()].to_string())
}

fn node_text(node: &Node, source: &str) -> String {
    source[node.start_byte()..node.end_byte()].to_string()
}

fn extract_signature(node: &Node, source: &str) -> Option<String> {
    let text = node_text(node, source);
    // Find the first '{' for block-bodied constructs
    if let Some(pos) = text.find('{') {
        let sig = text[..pos].trim();
        if !sig.is_empty() {
            return Some(sig.to_string());
        }
    }
    // Fallback: first line
    Some(text.lines().next().unwrap_or("").trim().to_string())
}

fn rust_impl_name(node: &Node, source: &str) -> Option<String> {
    // For `impl Trait for Type`, extract "Trait for Type"
    // For `impl Type`, extract "Type"
    // Use the type field and optionally the trait field
    let type_node = node.child_by_field_name("type")?;
    let type_name = source[type_node.start_byte()..type_node.end_byte()].to_string();
    if let Some(trait_node) = node.child_by_field_name("trait") {
        let trait_name = source[trait_node.start_byte()..trait_node.end_byte()].to_string();
        Some(format!("{trait_name} for {type_name}"))
    } else {
        Some(type_name)
    }
}

fn node_start_line(node: &Node) -> u32 {
    node.start_position().row as u32 + 1
}

fn node_end_line(node: &Node) -> u32 {
    let end = node.end_position();
    if end.column == 0 && end.row > node.start_position().row {
        end.row as u32
    } else {
        end.row as u32 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(source: &str) -> Vec<RawChunk> {
        let chunker = TreeSitterChunker::new();
        chunker.chunk_file(Path::new("test.ts"), source, Language::TypeScript)
    }

    #[test]
    fn extracts_function_declaration() {
        let source = "function greet(name: string): string {\n  return `Hello, ${name}!`;\n}";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Function);
        assert_eq!(chunks[0].name, "greet");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
        assert!(chunks[0].content.contains("return `Hello"));
    }

    #[test]
    fn extracts_arrow_function() {
        let source = "const add = (a: number, b: number): number => {\n  return a + b;\n};";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Function);
        assert_eq!(chunks[0].name, "add");
    }

    #[test]
    fn extracts_class_with_methods() {
        let source = "class Calculator {\n  add(n: number): void {\n    this.value += n;\n  }\n}";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Class);
        assert_eq!(chunks[0].name, "Calculator");
        assert!(chunks[0].content.contains("add(n: number)"));
    }

    #[test]
    fn extracts_type_alias() {
        let source = "type Point = {\n  x: number;\n  y: number;\n};";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Type);
        assert_eq!(chunks[0].name, "Point");
    }

    #[test]
    fn extracts_interface() {
        let source = "interface Shape {\n  area(): number;\n  perimeter(): number;\n}";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Type);
        assert_eq!(chunks[0].name, "Shape");
    }

    #[test]
    fn extracts_enum() {
        let source = "enum Color {\n  Red,\n  Green,\n  Blue,\n}";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Type);
        assert_eq!(chunks[0].name, "Color");
    }

    #[test]
    fn extracts_exported_function() {
        let source =
            "export function multiply(a: number, b: number): number {\n  return a * b;\n}";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Function);
        assert_eq!(chunks[0].name, "multiply");
        assert!(chunks[0].content.starts_with("export"));
    }

    #[test]
    fn extracts_exported_arrow_function() {
        let source =
            "export const add = (a: number, b: number): number => {\n  return a + b;\n};";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Function);
        assert_eq!(chunks[0].name, "add");
        assert!(chunks[0].content.starts_with("export"));
    }

    #[test]
    fn module_level_code_captured() {
        let source =
            "import { foo } from 'bar';\n\nconst PI = 3.14;\n\nfunction greet() {\n  return 'hello';\n}";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].kind, ChunkKind::Module);
        assert_eq!(chunks[0].name, "module");
        assert!(chunks[0].content.contains("import"));
        assert!(chunks[0].content.contains("PI"));
        assert_eq!(chunks[1].kind, ChunkKind::Function);
        assert_eq!(chunks[1].name, "greet");
    }

    #[test]
    fn chunks_do_not_overlap() {
        let source = "import { x } from 'y';\n\nfunction a() { return 1; }\n\nconst b = 2;\n\nfunction c() { return 3; }";
        let chunks = chunk(source);
        for i in 0..chunks.len() {
            for j in (i + 1)..chunks.len() {
                assert!(
                    chunks[i].end_line <= chunks[j].start_line
                        || chunks[j].end_line <= chunks[i].start_line,
                    "Chunks {} and {} overlap: [{}-{}] vs [{}-{}]",
                    chunks[i].name,
                    chunks[j].name,
                    chunks[i].start_line,
                    chunks[i].end_line,
                    chunks[j].start_line,
                    chunks[j].end_line,
                );
            }
        }
    }

    #[test]
    fn functions_include_full_body() {
        let source =
            "function add(a: number, b: number): number {\n  const result = a + b;\n  return result;\n}";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("const result = a + b;"));
        assert!(chunks[0].content.contains("return result;"));
    }

    #[test]
    fn signature_extraction() {
        let source = "function greet(name: string): string {\n  return name;\n}";
        let chunks = chunk(source);
        assert_eq!(
            chunks[0].signature,
            Some("function greet(name: string): string".to_string())
        );
    }

    #[test]
    fn empty_source_returns_empty() {
        let chunks = chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn mixed_declarations_and_module_code() {
        let source = r#"import { foo } from 'bar';

function hello() {
  return 'world';
}

const VERSION = '1.0';

class App {
  run() {}
}
"#;
        let chunks = chunk(source);
        // Module (import), Function (hello), Module (VERSION), Class (App)
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].kind, ChunkKind::Module);
        assert_eq!(chunks[1].kind, ChunkKind::Function);
        assert_eq!(chunks[1].name, "hello");
        assert_eq!(chunks[2].kind, ChunkKind::Module);
        assert!(chunks[2].content.contains("VERSION"));
        assert_eq!(chunks[3].kind, ChunkKind::Class);
        assert_eq!(chunks[3].name, "App");
    }
}

#[cfg(test)]
mod rust_tests {
    use super::*;

    fn chunk_rs(source: &str) -> Vec<RawChunk> {
        let chunker = TreeSitterChunker::new();
        chunker.chunk_file(Path::new("test.rs"), source, Language::Rust)
    }

    #[test]
    fn extracts_function_item() {
        let source = "fn greet(name: &str) -> String {\n    format!(\"Hello, {}!\", name)\n}";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Function);
        assert_eq!(chunks[0].name, "greet");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
        assert!(chunks[0].content.contains("format!"));
    }

    #[test]
    fn extracts_struct_item() {
        let source = "struct Point {\n    x: f64,\n    y: f64,\n}";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Type);
        assert_eq!(chunks[0].name, "Point");
    }

    #[test]
    fn extracts_enum_item() {
        let source = "enum Color {\n    Red,\n    Green,\n    Blue,\n}";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Type);
        assert_eq!(chunks[0].name, "Color");
    }

    #[test]
    fn extracts_impl_block_as_single_unit() {
        let source = "impl Point {\n    fn new(x: f64, y: f64) -> Self {\n        Self { x, y }\n    }\n\n    fn distance(&self) -> f64 {\n        (self.x.powi(2) + self.y.powi(2)).sqrt()\n    }\n}";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Impl);
        assert_eq!(chunks[0].name, "Point");
        assert!(chunks[0].content.contains("fn new"));
        assert!(chunks[0].content.contains("fn distance"));
    }

    #[test]
    fn extracts_trait_item() {
        let source = "trait Drawable {\n    fn draw(&self);\n    fn bounds(&self) -> Rect;\n}";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Type);
        assert_eq!(chunks[0].name, "Drawable");
    }

    #[test]
    fn extracts_type_alias() {
        let source = "type Result<T> = std::result::Result<T, MyError>;";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Type);
        assert_eq!(chunks[0].name, "Result");
    }

    #[test]
    fn extracts_mod_item() {
        let source = "mod tests {\n    use super::*;\n\n    fn helper() {}\n}";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Module);
        assert_eq!(chunks[0].name, "tests");
        assert!(chunks[0].content.contains("fn helper"));
    }

    #[test]
    fn extracts_const_item() {
        let source = "const MAX_SIZE: usize = 1024;";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Other);
        assert_eq!(chunks[0].name, "MAX_SIZE");
    }

    #[test]
    fn extracts_static_item() {
        let source = "static COUNTER: AtomicUsize = AtomicUsize::new(0);";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Other);
        assert_eq!(chunks[0].name, "COUNTER");
    }

    #[test]
    fn extracts_macro_definition() {
        let source = "macro_rules! vec_of {\n    ($($x:expr),*) => {\n        vec![$($x),*]\n    };\n}";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Function);
        assert_eq!(chunks[0].name, "vec_of");
    }

    #[test]
    fn impl_trait_for_type() {
        let source = "impl Display for Point {\n    fn fmt(&self, f: &mut Formatter) -> fmt::Result {\n        write!(f, \"({}, {})\", self.x, self.y)\n    }\n}";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Impl);
        assert_eq!(chunks[0].name, "Display for Point");
    }

    #[test]
    fn rust_signature_extraction() {
        let source = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}";
        let chunks = chunk_rs(source);
        assert_eq!(
            chunks[0].signature,
            Some("fn add(a: i32, b: i32) -> i32".to_string())
        );
    }

    #[test]
    fn module_level_use_statements() {
        let source = "use std::io;\nuse std::path::Path;\n\nfn main() {\n    println!(\"hello\");\n}";
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].kind, ChunkKind::Module);
        assert!(chunks[0].content.contains("use std::io"));
        assert_eq!(chunks[1].kind, ChunkKind::Function);
        assert_eq!(chunks[1].name, "main");
    }

    #[test]
    fn mixed_rust_declarations() {
        let source = r#"use std::fmt;

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

fn main() {
    let p = Point::new(1.0, 2.0);
}
"#;
        let chunks = chunk_rs(source);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].kind, ChunkKind::Module);
        assert!(chunks[0].content.contains("use std::fmt"));
        assert_eq!(chunks[1].kind, ChunkKind::Type);
        assert_eq!(chunks[1].name, "Point");
        assert_eq!(chunks[2].kind, ChunkKind::Impl);
        assert_eq!(chunks[2].name, "Point");
        assert_eq!(chunks[3].kind, ChunkKind::Function);
        assert_eq!(chunks[3].name, "main");
    }

    #[test]
    fn empty_rust_source() {
        let chunks = chunk_rs("");
        assert!(chunks.is_empty());
    }
}
