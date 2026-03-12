use std::collections::HashMap;
use std::fs;
use std::path::Path;

use engram_core::{Abstraction, KeyAbstractions, PatternInfo, Result};
use tree_sitter::{Node, Parser};

use crate::chunker::Language;
use crate::detect::{detect_language, ChunkerKind};

const MAX_ABSTRACTIONS: usize = 50;

/// Extract key abstractions (types, traits, interfaces, classes) from a repository.
///
/// Walks source files matching the given language, parses them with tree-sitter,
/// identifies exported declarations, counts cross-file references, and returns
/// the top ~50 abstractions prioritized by importance.
pub fn extract_key_abstractions(
    repo_path: &Path,
    language: &str,
) -> Result<KeyAbstractions> {
    let target_lang = match language.to_lowercase().as_str() {
        "rust" => Some(Language::Rust),
        "typescript" | "javascript" => Some(Language::TypeScript),
        "python" => Some(Language::Python),
        "go" | "golang" => Some(Language::Go),
        "java" => Some(Language::Java),
        "c" => Some(Language::C),
        "c++" | "cpp" => Some(Language::Cpp),
        _ => None,
    };

    // Collect source files for the target language
    let mut source_files: Vec<(String, String, Language)> = Vec::new();
    collect_source_files(repo_path, repo_path, &target_lang, &mut source_files)?;

    // Phase 1: Extract all declarations from each file
    let mut all_declarations: Vec<RawDeclaration> = Vec::new();
    for (rel_path, content, lang) in &source_files {
        let decls = extract_declarations(rel_path, content, *lang);
        all_declarations.extend(decls);
    }

    if all_declarations.is_empty() {
        return Ok(KeyAbstractions {
            abstractions: vec![],
            patterns: vec![],
        });
    }

    // Phase 2: Count how many files reference each declaration name
    let mut ref_counts: HashMap<String, usize> = HashMap::new();
    for decl in &all_declarations {
        ref_counts.entry(decl.name.clone()).or_insert(0);
    }
    for (_rel_path, content, _lang) in &source_files {
        for name in ref_counts.keys().cloned().collect::<Vec<_>>() {
            // Count files that reference this name (excluding mere declarations)
            if content.contains(&name) {
                *ref_counts.entry(name).or_insert(0) += 1;
            }
        }
    }

    // Phase 3: Score and sort declarations
    let mut scored: Vec<(f64, RawDeclaration)> = all_declarations
        .into_iter()
        .map(|decl| {
            let mut score = 0.0;

            // Exported items get a boost
            if decl.is_exported {
                score += 10.0;
            }

            // Items from entry-point files get a boost
            if is_entry_point_file(&decl.file) {
                score += 5.0;
            }

            // Items in high-level (shallow) directories score higher
            let depth = decl.file.matches('/').count();
            score += 3.0 / (depth as f64 + 1.0);

            // Reference count boost (subtract 1 for the declaration file itself)
            let refs = ref_counts.get(&decl.name).copied().unwrap_or(0);
            let external_refs = refs.saturating_sub(1);
            score += external_refs as f64 * 2.0;

            // Traits/interfaces get a small boost (they define contracts)
            if matches!(decl.kind.as_str(), "trait" | "interface") {
                score += 3.0;
            }

            (score, decl)
        })
        .collect();

    // Sort by score descending, then by name for stability
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.name.cmp(&b.1.name))
    });

    // Deduplicate by name (keep the highest-scored instance)
    let mut seen = std::collections::HashSet::new();
    scored.retain(|(_, decl)| seen.insert(decl.name.clone()));

    // Take top ~50
    let abstractions: Vec<Abstraction> = scored
        .into_iter()
        .take(MAX_ABSTRACTIONS)
        .map(|(_, decl)| Abstraction {
            name: decl.name,
            kind: decl.kind,
            file: decl.file,
            description: decl.description,
        })
        .collect();

    // Detect patterns from the abstractions
    let patterns = detect_patterns(&abstractions, language);

    Ok(KeyAbstractions {
        abstractions,
        patterns,
    })
}

/// A raw declaration found during AST traversal.
#[derive(Debug)]
struct RawDeclaration {
    name: String,
    kind: String,
    file: String,
    description: String,
    is_exported: bool,
}

pub(crate) fn collect_source_files(
    root: &Path,
    current: &Path,
    target_lang: &Option<Language>,
    files: &mut Vec<(String, String, Language)>,
) -> Result<()> {
    let entries = match fs::read_dir(current) {
        Ok(entries) => entries,
        Err(_) => return Ok(()),
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();

        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') || is_skipped_dir(name) {
                continue;
            }
        }

        if path.is_dir() {
            collect_source_files(root, &path, target_lang, files)?;
        } else if path.is_file() {
            if let ChunkerKind::TreeSitter(lang) = detect_language(&path) {
                // If a target language is specified, only collect matching files
                if let Some(target) = target_lang {
                    if lang != *target {
                        continue;
                    }
                }

                let rel_path = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();

                if let Ok(content) = fs::read_to_string(&path) {
                    files.push((rel_path, content, lang));
                }
            }
        }
    }

    Ok(())
}

fn extract_declarations(rel_path: &str, source: &str, language: Language) -> Vec<RawDeclaration> {
    let mut parser = Parser::new();
    let ts_language: tree_sitter::Language = match language {
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::Java => tree_sitter_java::LANGUAGE.into(),
        Language::C => tree_sitter_c::LANGUAGE.into(),
        Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
    };
    parser
        .set_language(&ts_language)
        .expect("failed to set language");

    let tree = match parser.parse(source, None) {
        Some(tree) => tree,
        None => return vec![],
    };

    let root = tree.root_node();
    let mut declarations = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        match language {
            Language::Rust => extract_rust_declarations(&child, source, rel_path, &mut declarations),
            Language::TypeScript => {
                extract_ts_declarations(&child, source, rel_path, &mut declarations);
            }
            Language::Python => {
                extract_python_declarations(&child, source, rel_path, &mut declarations);
            }
            Language::Go => {
                extract_go_declarations(&child, source, rel_path, &mut declarations);
            }
            Language::Java => {
                extract_java_declarations(&child, source, rel_path, &mut declarations);
            }
            Language::C => {
                extract_c_declarations(&child, source, rel_path, &mut declarations);
            }
            Language::Cpp => {
                extract_cpp_declarations(&child, source, rel_path, &mut declarations);
            }
        }
    }

    declarations
}

fn extract_rust_declarations(
    node: &Node,
    source: &str,
    file: &str,
    decls: &mut Vec<RawDeclaration>,
) {
    match node.kind() {
        "struct_item" | "enum_item" | "trait_item" | "type_item" => {
            if let Some(name) = field_text(node, "name", source) {
                let kind = match node.kind() {
                    "struct_item" => "struct",
                    "enum_item" => "enum",
                    "trait_item" => "trait",
                    "type_item" => "type alias",
                    _ => "type",
                };
                let is_exported = is_rust_pub(node, source);
                let description = extract_doc_comment(node, source);
                decls.push(RawDeclaration {
                    name,
                    kind: kind.to_string(),
                    file: file.to_string(),
                    description,
                    is_exported,
                });
            }
        }
        _ => {}
    }
}

fn extract_ts_declarations(
    node: &Node,
    source: &str,
    file: &str,
    decls: &mut Vec<RawDeclaration>,
) {
    match node.kind() {
        "class_declaration" => {
            if let Some(name) = field_text(node, "name", source) {
                let description = extract_doc_comment(node, source);
                decls.push(RawDeclaration {
                    name,
                    kind: "class".to_string(),
                    file: file.to_string(),
                    description,
                    is_exported: false,
                });
            }
        }
        "interface_declaration" => {
            if let Some(name) = field_text(node, "name", source) {
                let description = extract_doc_comment(node, source);
                decls.push(RawDeclaration {
                    name,
                    kind: "interface".to_string(),
                    file: file.to_string(),
                    description,
                    is_exported: false,
                });
            }
        }
        "type_alias_declaration" => {
            if let Some(name) = field_text(node, "name", source) {
                let description = extract_doc_comment(node, source);
                decls.push(RawDeclaration {
                    name,
                    kind: "type alias".to_string(),
                    file: file.to_string(),
                    description,
                    is_exported: false,
                });
            }
        }
        "enum_declaration" => {
            if let Some(name) = field_text(node, "name", source) {
                let description = extract_doc_comment(node, source);
                decls.push(RawDeclaration {
                    name,
                    kind: "enum".to_string(),
                    file: file.to_string(),
                    description,
                    is_exported: false,
                });
            }
        }
        "export_statement" => {
            // Check for exported declarations inside
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                match child.kind() {
                    "class_declaration" => {
                        if let Some(name) = field_text(&child, "name", source) {
                            let description = extract_doc_comment(node, source);
                            decls.push(RawDeclaration {
                                name,
                                kind: "class".to_string(),
                                file: file.to_string(),
                                description,
                                is_exported: true,
                            });
                        }
                    }
                    "interface_declaration" => {
                        if let Some(name) = field_text(&child, "name", source) {
                            let description = extract_doc_comment(node, source);
                            decls.push(RawDeclaration {
                                name,
                                kind: "interface".to_string(),
                                file: file.to_string(),
                                description,
                                is_exported: true,
                            });
                        }
                    }
                    "type_alias_declaration" => {
                        if let Some(name) = field_text(&child, "name", source) {
                            let description = extract_doc_comment(node, source);
                            decls.push(RawDeclaration {
                                name,
                                kind: "type alias".to_string(),
                                file: file.to_string(),
                                description,
                                is_exported: true,
                            });
                        }
                    }
                    "enum_declaration" => {
                        if let Some(name) = field_text(&child, "name", source) {
                            let description = extract_doc_comment(node, source);
                            decls.push(RawDeclaration {
                                name,
                                kind: "enum".to_string(),
                                file: file.to_string(),
                                description,
                                is_exported: true,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn extract_python_declarations(
    node: &Node,
    source: &str,
    file: &str,
    decls: &mut Vec<RawDeclaration>,
) {
    match node.kind() {
        "class_definition" => {
            if let Some(name) = field_text(node, "name", source) {
                let description = extract_doc_comment(node, source);
                // Python top-level classes are considered exported
                decls.push(RawDeclaration {
                    name,
                    kind: "class".to_string(),
                    file: file.to_string(),
                    description,
                    is_exported: true,
                });
            }
        }
        "decorated_definition" => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "class_definition" {
                    if let Some(name) = field_text(&child, "name", source) {
                        let description = extract_doc_comment(node, source);
                        decls.push(RawDeclaration {
                            name,
                            kind: "class".to_string(),
                            file: file.to_string(),
                            description,
                            is_exported: true,
                        });
                    }
                }
            }
        }
        _ => {}
    }
}

fn extract_go_declarations(
    node: &Node,
    source: &str,
    file: &str,
    decls: &mut Vec<RawDeclaration>,
) {
    if node.kind() == "type_declaration" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "type_spec" {
                if let Some(name) = field_text(&child, "name", source) {
                    // Determine kind from the type field
                    let kind = child
                        .child_by_field_name("type")
                        .map(|t| match t.kind() {
                            "struct_type" => "struct",
                            "interface_type" => "interface",
                            _ => "type",
                        })
                        .unwrap_or("type");
                    let is_exported = name.starts_with(|c: char| c.is_ascii_uppercase());
                    let description = extract_doc_comment(node, source);
                    decls.push(RawDeclaration {
                        name,
                        kind: kind.to_string(),
                        file: file.to_string(),
                        description,
                        is_exported,
                    });
                }
            }
        }
    }
}

fn extract_java_declarations(
    node: &Node,
    source: &str,
    file: &str,
    decls: &mut Vec<RawDeclaration>,
) {
    match node.kind() {
        "class_declaration" | "interface_declaration" | "enum_declaration" => {
            if let Some(name) = field_text(node, "name", source) {
                let kind = match node.kind() {
                    "class_declaration" => "class",
                    "interface_declaration" => "interface",
                    "enum_declaration" => "enum",
                    _ => "type",
                };
                // Java top-level public classes are exported
                let text = &source[node.start_byte()..node.end_byte()];
                let is_exported = text.contains("public ");
                let description = extract_doc_comment(node, source);
                decls.push(RawDeclaration {
                    name,
                    kind: kind.to_string(),
                    file: file.to_string(),
                    description,
                    is_exported,
                });
            }
        }
        _ => {}
    }
}

fn extract_c_declarations(
    node: &Node,
    source: &str,
    file: &str,
    decls: &mut Vec<RawDeclaration>,
) {
    match node.kind() {
        "struct_specifier" | "enum_specifier" => {
            if let Some(name) = field_text(node, "name", source) {
                let kind = match node.kind() {
                    "struct_specifier" => "struct",
                    "enum_specifier" => "enum",
                    _ => "type",
                };
                let description = extract_doc_comment(node, source);
                decls.push(RawDeclaration {
                    name,
                    kind: kind.to_string(),
                    file: file.to_string(),
                    description,
                    is_exported: true, // C has no visibility modifiers; all top-level are "exported"
                });
            }
        }
        _ => {}
    }
}

fn extract_cpp_declarations(
    node: &Node,
    source: &str,
    file: &str,
    decls: &mut Vec<RawDeclaration>,
) {
    match node.kind() {
        "struct_specifier" | "enum_specifier" => {
            if let Some(name) = field_text(node, "name", source) {
                let kind = match node.kind() {
                    "struct_specifier" => "struct",
                    "enum_specifier" => "enum",
                    _ => "type",
                };
                let description = extract_doc_comment(node, source);
                decls.push(RawDeclaration {
                    name,
                    kind: kind.to_string(),
                    file: file.to_string(),
                    description,
                    is_exported: true,
                });
            }
        }
        "class_specifier" => {
            if let Some(name) = field_text(node, "name", source) {
                let description = extract_doc_comment(node, source);
                decls.push(RawDeclaration {
                    name,
                    kind: "class".to_string(),
                    file: file.to_string(),
                    description,
                    is_exported: true,
                });
            }
        }
        "template_declaration" => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                match child.kind() {
                    "class_specifier" => {
                        if let Some(name) = field_text(&child, "name", source) {
                            let description = extract_doc_comment(node, source);
                            decls.push(RawDeclaration {
                                name,
                                kind: "class".to_string(),
                                file: file.to_string(),
                                description,
                                is_exported: true,
                            });
                        }
                    }
                    "struct_specifier" => {
                        if let Some(name) = field_text(&child, "name", source) {
                            let description = extract_doc_comment(node, source);
                            decls.push(RawDeclaration {
                                name,
                                kind: "struct".to_string(),
                                file: file.to_string(),
                                description,
                                is_exported: true,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn field_text(node: &Node, field: &str, source: &str) -> Option<String> {
    let child = node.child_by_field_name(field)?;
    Some(source[child.start_byte()..child.end_byte()].to_string())
}

fn is_rust_pub(node: &Node, source: &str) -> bool {
    // Check if the node starts with "pub" by looking at the first child or the text
    let text = &source[node.start_byte()..node.end_byte()];
    text.starts_with("pub ")
        || text.starts_with("pub(crate)")
        || text.starts_with("pub(super)")
}

fn extract_doc_comment(node: &Node, source: &str) -> String {
    // Look at the sibling node immediately before this one for doc comments
    let mut prev = node.prev_sibling();
    let mut comment_lines: Vec<String> = Vec::new();

    while let Some(sibling) = prev {
        let kind = sibling.kind();
        if kind == "line_comment" || kind == "comment" || kind == "block_comment" {
            let text = &source[sibling.start_byte()..sibling.end_byte()];
            // Rust doc comments: /// or //!
            if text.starts_with("///") {
                let line = text.trim_start_matches("///").trim();
                comment_lines.push(line.to_string());
            }
            // Python/TS single-line comments: # or //
            else if text.starts_with("//") {
                let line = text.trim_start_matches("//").trim();
                comment_lines.push(line.to_string());
            } else if text.starts_with('#') {
                let line = text.trim_start_matches('#').trim();
                comment_lines.push(line.to_string());
            }
            // Block comments: /* ... */ or /** ... */
            else if text.starts_with("/*") {
                let inner = text
                    .trim_start_matches("/**")
                    .trim_start_matches("/*")
                    .trim_end_matches("*/")
                    .trim();
                // Take first meaningful line from block comment
                for line in inner.lines() {
                    let trimmed = line.trim().trim_start_matches('*').trim();
                    if !trimmed.is_empty() {
                        comment_lines.push(trimmed.to_string());
                        break;
                    }
                }
            }
            prev = sibling.prev_sibling();
        } else {
            break;
        }
    }

    comment_lines.reverse();
    if comment_lines.is_empty() {
        String::new()
    } else {
        // Return first sentence/line as a brief description
        let full = comment_lines.join(" ");
        // Truncate at first period+space or 200 chars
        if let Some(pos) = full.find(". ") {
            full[..=pos].to_string()
        } else if full.len() > 200 {
            format!("{}...", &full[..197])
        } else {
            full
        }
    }
}

fn is_entry_point_file(file: &str) -> bool {
    let name = file.rsplit('/').next().unwrap_or(file);
    matches!(
        name,
        "index.ts"
            | "index.tsx"
            | "index.js"
            | "index.jsx"
            | "mod.rs"
            | "lib.rs"
            | "main.rs"
            | "__init__.py"
            | "main.py"
            | "main.go"
            | "app.ts"
            | "app.js"
            | "app.py"
    )
}

fn is_skipped_dir(name: &str) -> bool {
    matches!(
        name,
        "node_modules"
            | "target"
            | "dist"
            | "build"
            | ".git"
            | "__pycache__"
            | ".tox"
            | ".venv"
            | "venv"
            | ".mypy_cache"
            | ".pytest_cache"
            | ".next"
            | "coverage"
            | "vendor"
    )
}

pub(crate) fn detect_patterns(abstractions: &[Abstraction], language: &str) -> Vec<PatternInfo> {
    let mut patterns = Vec::new();

    // Count abstractions by kind
    let mut kind_counts: HashMap<&str, Vec<&str>> = HashMap::new();
    for a in abstractions {
        kind_counts
            .entry(a.kind.as_str())
            .or_default()
            .push(a.name.as_str());
    }

    match language.to_lowercase().as_str() {
        "rust" => {
            if let Some(traits) = kind_counts.get("trait") {
                if traits.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Trait-based abstraction".to_string(),
                        description: "Uses traits to define shared behavior contracts".to_string(),
                        examples: traits.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
            if let Some(enums) = kind_counts.get("enum") {
                if enums.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Enum-based modeling".to_string(),
                        description: "Uses enums for type-safe variant modeling".to_string(),
                        examples: enums.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
        }
        "typescript" | "javascript" => {
            if let Some(interfaces) = kind_counts.get("interface") {
                if interfaces.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Interface-driven design".to_string(),
                        description: "Uses interfaces to define type contracts".to_string(),
                        examples: interfaces.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
            if let Some(classes) = kind_counts.get("class") {
                if classes.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Class-based architecture".to_string(),
                        description: "Uses classes for encapsulation and organization".to_string(),
                        examples: classes.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
        }
        "python" => {
            if let Some(classes) = kind_counts.get("class") {
                if classes.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Class-based design".to_string(),
                        description: "Uses classes for data and behavior organization".to_string(),
                        examples: classes.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
        }
        "go" | "golang" => {
            if let Some(interfaces) = kind_counts.get("interface") {
                if interfaces.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Interface-driven design".to_string(),
                        description: "Uses interfaces to define behavioral contracts".to_string(),
                        examples: interfaces.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
            if let Some(structs) = kind_counts.get("struct") {
                if structs.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Struct-based modeling".to_string(),
                        description: "Uses structs for data modeling".to_string(),
                        examples: structs.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
        }
        "java" => {
            if let Some(interfaces) = kind_counts.get("interface") {
                if interfaces.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Interface-driven design".to_string(),
                        description: "Uses interfaces to define type contracts".to_string(),
                        examples: interfaces.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
            if let Some(classes) = kind_counts.get("class") {
                if classes.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Class-based architecture".to_string(),
                        description: "Uses classes for encapsulation and organization".to_string(),
                        examples: classes.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
        }
        "c" => {
            if let Some(structs) = kind_counts.get("struct") {
                if structs.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Struct-based modeling".to_string(),
                        description: "Uses structs for data modeling".to_string(),
                        examples: structs.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
        }
        "c++" | "cpp" => {
            if let Some(classes) = kind_counts.get("class") {
                if classes.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Class-based architecture".to_string(),
                        description: "Uses classes for encapsulation and organization".to_string(),
                        examples: classes.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
            if let Some(structs) = kind_counts.get("struct") {
                if structs.len() >= 2 {
                    patterns.push(PatternInfo {
                        name: "Struct-based modeling".to_string(),
                        description: "Uses structs for data modeling".to_string(),
                        examples: structs.iter().take(3).map(|s| s.to_string()).collect(),
                    });
                }
            }
        }
        _ => {}
    }

    // Check for one-module-per-type pattern (common in Rust)
    let mut files_with_single_type: Vec<&str> = Vec::new();
    let mut file_type_counts: HashMap<&str, usize> = HashMap::new();
    for a in abstractions {
        *file_type_counts.entry(a.file.as_str()).or_insert(0) += 1;
    }
    for (file, count) in &file_type_counts {
        if *count == 1 {
            files_with_single_type.push(file);
        }
    }
    if files_with_single_type.len() >= 3 && file_type_counts.len() >= 3 {
        let ratio = files_with_single_type.len() as f64 / file_type_counts.len() as f64;
        if ratio >= 0.5 {
            patterns.push(PatternInfo {
                name: "One module per concept".to_string(),
                description: "Each type gets its own dedicated module file".to_string(),
                examples: files_with_single_type
                    .iter()
                    .take(3)
                    .map(|s| s.to_string())
                    .collect(),
            });
        }
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn extracts_rust_structs_and_traits() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/lib.rs"),
            r#"/// The main error type.
pub struct AppError {
    message: String,
}

/// A provider trait for embeddings.
pub trait EmbeddingProvider {
    fn embed(&self, text: &str) -> Vec<f32>;
}

struct InternalHelper {
    data: Vec<u8>,
}
"#,
        )
        .unwrap();

        let result = extract_key_abstractions(root, "Rust").unwrap();
        assert!(!result.abstractions.is_empty());

        let app_error = result.abstractions.iter().find(|a| a.name == "AppError");
        assert!(app_error.is_some());
        let app_error = app_error.unwrap();
        assert_eq!(app_error.kind, "struct");
        assert!(app_error.description.contains("main error type"));

        let provider = result
            .abstractions
            .iter()
            .find(|a| a.name == "EmbeddingProvider");
        assert!(provider.is_some());
        assert_eq!(provider.unwrap().kind, "trait");

        // Internal (non-pub) should still appear but with lower priority
        let internal = result
            .abstractions
            .iter()
            .find(|a| a.name == "InternalHelper");
        assert!(internal.is_some());
    }

    #[test]
    fn extracts_typescript_interfaces_and_classes() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/index.ts"),
            r#"// The user model
export interface User {
  id: string;
  name: string;
}

export class UserService {
  getUser(id: string): User {
    return { id, name: "test" };
  }
}

type UserId = string;
"#,
        )
        .unwrap();

        let result = extract_key_abstractions(root, "TypeScript").unwrap();
        assert!(!result.abstractions.is_empty());

        let user = result.abstractions.iter().find(|a| a.name == "User");
        assert!(user.is_some());
        assert_eq!(user.unwrap().kind, "interface");

        let service = result
            .abstractions
            .iter()
            .find(|a| a.name == "UserService");
        assert!(service.is_some());
        assert_eq!(service.unwrap().kind, "class");
    }

    #[test]
    fn extracts_python_classes() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/models.py"),
            r#"# A data model for users
class User:
    def __init__(self, name):
        self.name = name

class AdminUser(User):
    def __init__(self, name, role):
        super().__init__(name)
        self.role = role
"#,
        )
        .unwrap();

        let result = extract_key_abstractions(root, "Python").unwrap();
        assert!(!result.abstractions.is_empty());

        let user = result.abstractions.iter().find(|a| a.name == "User");
        assert!(user.is_some());
        assert_eq!(user.unwrap().kind, "class");

        let admin = result.abstractions.iter().find(|a| a.name == "AdminUser");
        assert!(admin.is_some());
    }

    #[test]
    fn prioritizes_exported_and_entry_point_items() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/lib.rs"),
            r#"pub struct TopLevel {
    data: String,
}
"#,
        )
        .unwrap();
        fs::write(
            root.join("src/internal.rs"),
            r#"struct DeepInternal {
    data: String,
}
"#,
        )
        .unwrap();

        let result = extract_key_abstractions(root, "Rust").unwrap();
        assert!(result.abstractions.len() >= 2);
        // TopLevel should be first (exported + entry point)
        assert_eq!(result.abstractions[0].name, "TopLevel");
    }

    #[test]
    fn limits_to_50_abstractions() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();

        let mut source = String::new();
        for i in 0..60 {
            source.push_str(&format!("pub struct Type{} {{}}\n", i));
        }
        fs::write(root.join("src/lib.rs"), &source).unwrap();

        let result = extract_key_abstractions(root, "Rust").unwrap();
        assert!(result.abstractions.len() <= MAX_ABSTRACTIONS);
    }

    #[test]
    fn reference_counting_boosts_popular_types() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/types.rs"),
            r#"pub struct Config {
    data: String,
}

pub struct RareType {
    data: String,
}
"#,
        )
        .unwrap();
        // Config is referenced in multiple files
        fs::write(
            root.join("src/a.rs"),
            "use crate::Config;\nfn setup() -> Config { todo!() }\n",
        )
        .unwrap();
        fs::write(
            root.join("src/b.rs"),
            "use crate::Config;\nfn init() -> Config { todo!() }\n",
        )
        .unwrap();

        let result = extract_key_abstractions(root, "Rust").unwrap();
        let config_pos = result
            .abstractions
            .iter()
            .position(|a| a.name == "Config");
        let rare_pos = result
            .abstractions
            .iter()
            .position(|a| a.name == "RareType");
        assert!(config_pos.is_some());
        assert!(rare_pos.is_some());
        assert!(
            config_pos.unwrap() < rare_pos.unwrap(),
            "Config should rank higher due to more references"
        );
    }

    #[test]
    fn detects_trait_pattern() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/lib.rs"),
            r#"pub trait Readable {
    fn read(&self) -> Vec<u8>;
}

pub trait Writable {
    fn write(&self, data: &[u8]);
}
"#,
        )
        .unwrap();

        let result = extract_key_abstractions(root, "Rust").unwrap();
        let trait_pattern = result
            .patterns
            .iter()
            .find(|p| p.name == "Trait-based abstraction");
        assert!(trait_pattern.is_some());
        assert!(trait_pattern.unwrap().examples.contains(&"Readable".to_string()));
    }

    #[test]
    fn empty_repo_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let result = extract_key_abstractions(tmp.path(), "Rust").unwrap();
        assert!(result.abstractions.is_empty());
        assert!(result.patterns.is_empty());
    }

    #[test]
    fn unsupported_language_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.go"), "package main\ntype Foo struct{}").unwrap();

        let result = extract_key_abstractions(root, "Go").unwrap();
        assert!(result.abstractions.is_empty());
    }

    #[test]
    fn skips_hidden_and_build_dirs() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::create_dir_all(root.join("node_modules/dep")).unwrap();

        fs::write(root.join("src/lib.rs"), "pub struct Good {}").unwrap();
        fs::write(root.join("target/debug/build.rs"), "pub struct Bad {}").unwrap();
        fs::write(root.join("node_modules/dep/index.ts"), "export class Bad {}").unwrap();

        let result = extract_key_abstractions(root, "Rust").unwrap();
        assert!(result.abstractions.iter().any(|a| a.name == "Good"));
        assert!(!result.abstractions.iter().any(|a| a.name == "Bad"));
    }

    #[test]
    fn extracts_doc_comments() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/lib.rs"),
            r#"/// A configuration for the application.
/// Contains all runtime settings.
pub struct Config {
    data: String,
}
"#,
        )
        .unwrap();

        let result = extract_key_abstractions(root, "Rust").unwrap();
        let config = result.abstractions.iter().find(|a| a.name == "Config");
        assert!(config.is_some());
        assert!(config.unwrap().description.contains("configuration"));
    }

    #[test]
    fn deduplicates_by_name() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        // Same struct name in two files (re-export scenario)
        fs::write(root.join("src/a.rs"), "pub struct Config {}").unwrap();
        fs::write(root.join("src/b.rs"), "pub struct Config {}").unwrap();

        let result = extract_key_abstractions(root, "Rust").unwrap();
        let config_count = result
            .abstractions
            .iter()
            .filter(|a| a.name == "Config")
            .count();
        assert_eq!(config_count, 1, "should deduplicate by name");
    }

    #[test]
    fn exported_ts_enum() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/types.ts"),
            r#"export enum Status {
  Active = "active",
  Inactive = "inactive",
}
"#,
        )
        .unwrap();

        let result = extract_key_abstractions(root, "TypeScript").unwrap();
        let status = result.abstractions.iter().find(|a| a.name == "Status");
        assert!(status.is_some());
        assert_eq!(status.unwrap().kind, "enum");
    }
}
