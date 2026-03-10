use std::fs;
use std::path::{Path, PathBuf};

use engram_core::{Decision, EngramError, GlossaryEntry, Lesson, Pattern, Result};

use crate::embeddings::write_embeddings_bin;

/// Generate a URL-friendly slug from a title.
///
/// Lowercases the input and replaces spaces with hyphens.
fn slugify(title: &str) -> String {
    title.to_lowercase().replace(' ', "-")
}

/// Write a decision to `knowledge/decisions/{date}_{slug}.yaml`.
///
/// The `date` portion is extracted from the decision's `created_at` field
/// (first 10 characters, e.g. `2026-03-09`).
pub fn write_decision(store_root: &Path, decision: &Decision) -> Result<PathBuf> {
    let date = &decision.created_at[..10];
    let slug = slugify(&decision.title);
    let filename = format!("{date}_{slug}.yaml");
    let path = store_root.join("knowledge/decisions").join(&filename);

    fs::create_dir_all(path.parent().unwrap())?;
    let yaml =
        serde_yaml::to_string(decision).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(&path, yaml)?;

    Ok(path)
}

/// Write a lesson to `knowledge/lessons/{date}_{slug}.yaml`.
///
/// The `date` portion is extracted from the lesson's `created_at` field
/// (first 10 characters, e.g. `2026-03-09`).
pub fn write_lesson(store_root: &Path, lesson: &Lesson) -> Result<PathBuf> {
    let date = &lesson.created_at[..10];
    let slug = slugify(&lesson.title);
    let filename = format!("{date}_{slug}.yaml");
    let path = store_root.join("knowledge/lessons").join(&filename);

    fs::create_dir_all(path.parent().unwrap())?;
    let yaml = serde_yaml::to_string(lesson).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(&path, yaml)?;

    Ok(path)
}

/// Write a pattern to `knowledge/patterns/{slug}.yaml`.
///
/// Patterns use only the slug (no date prefix) since they are living documents.
pub fn write_pattern(store_root: &Path, pattern: &Pattern) -> Result<PathBuf> {
    let slug = slugify(&pattern.name);
    let filename = format!("{slug}.yaml");
    let path = store_root.join("knowledge/patterns").join(&filename);

    fs::create_dir_all(path.parent().unwrap())?;
    let yaml =
        serde_yaml::to_string(pattern).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(&path, yaml)?;

    Ok(path)
}

/// Write a glossary entry to `knowledge/glossary/terms.yaml`.
///
/// If the file already exists, the entry is appended or updated (matching by `term`).
/// If it does not exist, a new file is created with a single entry.
pub fn write_glossary_entry(store_root: &Path, entry: &GlossaryEntry) -> Result<PathBuf> {
    let path = store_root.join("knowledge/glossary/terms.yaml");
    fs::create_dir_all(path.parent().unwrap())?;

    let mut entries: Vec<GlossaryEntry> = if path.exists() {
        let content = fs::read_to_string(&path)?;
        serde_yaml::from_str(&content).map_err(|e| EngramError::Serialize(e.to_string()))?
    } else {
        Vec::new()
    };

    // Update existing entry or append new one
    if let Some(existing) = entries.iter_mut().find(|e| e.term == entry.term) {
        *existing = entry.clone();
    } else {
        entries.push(entry.clone());
    }

    let yaml =
        serde_yaml::to_string(&entries).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(&path, yaml)?;

    Ok(path)
}

/// Compute the `.embedding.bin` path for a knowledge YAML file.
///
/// Replaces the `.yaml` extension with `.embedding.bin`.
pub fn knowledge_embedding_path(yaml_path: &Path) -> PathBuf {
    let stem = yaml_path.file_stem().unwrap().to_string_lossy();
    yaml_path.with_file_name(format!("{stem}.embedding.bin"))
}

/// Write an embedding binary alongside a knowledge YAML file.
///
/// Uses the same EGRM binary format as code chunk embeddings (single vector).
/// Returns the path to the created `.embedding.bin` file.
pub fn write_knowledge_embedding(
    yaml_path: &Path,
    embedding: &[f32],
    dimensions: usize,
) -> Result<PathBuf> {
    let emb_path = knowledge_embedding_path(yaml_path);
    write_embeddings_bin(&emb_path, &[embedding.to_vec()], dimensions)?;
    Ok(emb_path)
}

/// Extract embeddable text from a Decision.
pub fn decision_embed_text(d: &Decision) -> String {
    format!("{}\n{}\n{}", d.title, d.context, d.decision)
}

/// Extract embeddable text from a Lesson.
pub fn lesson_embed_text(l: &Lesson) -> String {
    format!("{}\n{}\n{}", l.title, l.description, l.resolution)
}

/// Extract embeddable text from a Pattern.
pub fn pattern_embed_text(p: &Pattern) -> String {
    format!("{}\n{}", p.name, p.description)
}

/// Write a decision with its embedding. Sets `embedding_ref` in the YAML.
pub fn write_decision_with_embedding(
    store_root: &Path,
    decision: &Decision,
    embedding: &[f32],
    dimensions: usize,
) -> Result<PathBuf> {
    let yaml_path = write_decision(store_root, decision)?;
    let emb_path = write_knowledge_embedding(&yaml_path, embedding, dimensions)?;

    let rel = emb_path
        .strip_prefix(store_root)
        .unwrap_or(&emb_path)
        .to_string_lossy()
        .to_string();

    let mut updated = decision.clone();
    updated.embedding_ref = Some(rel);
    let yaml =
        serde_yaml::to_string(&updated).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(&yaml_path, yaml)?;

    Ok(yaml_path)
}

/// Write a lesson with its embedding. Sets `embedding_ref` in the YAML.
pub fn write_lesson_with_embedding(
    store_root: &Path,
    lesson: &Lesson,
    embedding: &[f32],
    dimensions: usize,
) -> Result<PathBuf> {
    let yaml_path = write_lesson(store_root, lesson)?;
    let emb_path = write_knowledge_embedding(&yaml_path, embedding, dimensions)?;

    let rel = emb_path
        .strip_prefix(store_root)
        .unwrap_or(&emb_path)
        .to_string_lossy()
        .to_string();

    let mut updated = lesson.clone();
    updated.embedding_ref = Some(rel);
    let yaml =
        serde_yaml::to_string(&updated).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(&yaml_path, yaml)?;

    Ok(yaml_path)
}

/// Write a pattern with its embedding. Sets `embedding_ref` in the YAML.
pub fn write_pattern_with_embedding(
    store_root: &Path,
    pattern: &Pattern,
    embedding: &[f32],
    dimensions: usize,
) -> Result<PathBuf> {
    let yaml_path = write_pattern(store_root, pattern)?;
    let emb_path = write_knowledge_embedding(&yaml_path, embedding, dimensions)?;

    let rel = emb_path
        .strip_prefix(store_root)
        .unwrap_or(&emb_path)
        .to_string_lossy()
        .to_string();

    let mut updated = pattern.clone();
    updated.embedding_ref = Some(rel);
    let yaml =
        serde_yaml::to_string(&updated).map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(&yaml_path, yaml)?;

    Ok(yaml_path)
}

/// Metadata about a knowledge item that has an embedding, used during boot indexing.
pub struct KnowledgeEmbeddingInfo {
    /// Knowledge type: "decision", "lesson", or "pattern".
    pub kind: String,
    /// Item identifier (e.g., "DEC-001").
    pub id: String,
    /// Human-readable title/name.
    pub title: String,
    /// Path to the embedding binary file.
    pub embedding_path: PathBuf,
    /// Timestamp from the knowledge item.
    pub created_at: String,
}

/// Scan the knowledge directory for items that have embeddings.
///
/// Returns metadata for each knowledge item that has a corresponding
/// `.embedding.bin` file alongside its YAML.
pub fn scan_knowledge_embeddings(store_root: &Path) -> Result<Vec<KnowledgeEmbeddingInfo>> {
    let mut results = Vec::new();

    // Scan decisions
    let decisions_dir = store_root.join("knowledge/decisions");
    if decisions_dir.is_dir() {
        for entry in fs::read_dir(&decisions_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml") {
                let emb_path = knowledge_embedding_path(&path);
                if emb_path.exists() {
                    let yaml = fs::read_to_string(&path)?;
                    let decision: Decision = serde_yaml::from_str(&yaml)
                        .map_err(|e| EngramError::Serialize(e.to_string()))?;
                    results.push(KnowledgeEmbeddingInfo {
                        kind: "decision".to_string(),
                        id: decision.id,
                        title: decision.title,
                        embedding_path: emb_path,
                        created_at: decision.created_at,
                    });
                }
            }
        }
    }

    // Scan lessons
    let lessons_dir = store_root.join("knowledge/lessons");
    if lessons_dir.is_dir() {
        for entry in fs::read_dir(&lessons_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml") {
                let emb_path = knowledge_embedding_path(&path);
                if emb_path.exists() {
                    let yaml = fs::read_to_string(&path)?;
                    let lesson: Lesson = serde_yaml::from_str(&yaml)
                        .map_err(|e| EngramError::Serialize(e.to_string()))?;
                    results.push(KnowledgeEmbeddingInfo {
                        kind: "lesson".to_string(),
                        id: lesson.id,
                        title: lesson.title,
                        embedding_path: emb_path,
                        created_at: lesson.created_at,
                    });
                }
            }
        }
    }

    // Scan patterns
    let patterns_dir = store_root.join("knowledge/patterns");
    if patterns_dir.is_dir() {
        for entry in fs::read_dir(&patterns_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml") {
                let emb_path = knowledge_embedding_path(&path);
                if emb_path.exists() {
                    let yaml = fs::read_to_string(&path)?;
                    let pattern: Pattern = serde_yaml::from_str(&yaml)
                        .map_err(|e| EngramError::Serialize(e.to_string()))?;
                    results.push(KnowledgeEmbeddingInfo {
                        kind: "pattern".to_string(),
                        id: pattern.id,
                        title: pattern.name,
                        embedding_path: emb_path,
                        created_at: pattern.created_at,
                    });
                }
            }
        }
    }

    // Sort by id for deterministic ordering
    results.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_store_dirs(root: &Path) {
        fs::create_dir_all(root.join("knowledge/decisions")).unwrap();
        fs::create_dir_all(root.join("knowledge/lessons")).unwrap();
        fs::create_dir_all(root.join("knowledge/patterns")).unwrap();
        fs::create_dir_all(root.join("knowledge/glossary")).unwrap();
    }

    fn sample_decision() -> Decision {
        Decision {
            id: "DEC-001".to_string(),
            title: "Use HNSW for vector search".to_string(),
            status: "accepted".to_string(),
            context: "Need fast approximate nearest neighbor search for embeddings".to_string(),
            decision: "Use HNSW index with ef_construction=200".to_string(),
            consequences: vec![
                "Fast query times".to_string(),
                "Higher memory usage".to_string(),
            ],
            related_files: vec!["crates/engram-query/src/hnsw.rs".to_string()],
            contributed_by: "agent-session-1".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
            embedding_ref: None,
        }
    }

    fn sample_lesson() -> Lesson {
        Lesson {
            id: "LES-001".to_string(),
            title: "Always scope git2 borrows".to_string(),
            description: "git2 Repository borrows must be scoped before moving".to_string(),
            trigger: "Borrow checker error when moving Repository".to_string(),
            resolution: "Scope borrows in a block before moving".to_string(),
            related_files: vec!["crates/engram-store/src/lib.rs".to_string()],
            contributed_by: "agent-session-2".to_string(),
            created_at: "2026-03-09T12:00:00Z".to_string(),
            embedding_ref: None,
        }
    }

    fn sample_pattern() -> Pattern {
        Pattern {
            id: "PAT-001".to_string(),
            name: "One module per concept".to_string(),
            description: "Each concept gets its own module file".to_string(),
            examples: vec![
                "error.rs for error types".to_string(),
                "chunk.rs for chunk types".to_string(),
            ],
            anti_patterns: vec!["Putting all types in lib.rs".to_string()],
            contributed_by: "agent-session-3".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
            embedding_ref: None,
        }
    }

    fn sample_glossary_entry() -> GlossaryEntry {
        GlossaryEntry {
            term: "chunk".to_string(),
            definition: "A semantic unit of code extracted by tree-sitter".to_string(),
            context: "Used throughout engram for indexing and search".to_string(),
            contributed_by: "agent-session-4".to_string(),
            created_at: "2026-03-09T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("Use HNSW for vector search"), "use-hnsw-for-vector-search");
        assert_eq!(slugify("One module per concept"), "one-module-per-concept");
        assert_eq!(slugify("simple"), "simple");
    }

    #[test]
    fn test_write_decision_creates_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let decision = sample_decision();
        let path = write_decision(root, &decision).unwrap();

        assert_eq!(
            path,
            root.join("knowledge/decisions/2026-03-09_use-hnsw-for-vector-search.yaml")
        );
        assert!(path.exists());

        let content = fs::read_to_string(&path).unwrap();
        let deserialized: Decision = serde_yaml::from_str(&content).unwrap();
        assert_eq!(deserialized, decision);
    }

    #[test]
    fn test_write_lesson_creates_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let lesson = sample_lesson();
        let path = write_lesson(root, &lesson).unwrap();

        assert_eq!(
            path,
            root.join("knowledge/lessons/2026-03-09_always-scope-git2-borrows.yaml")
        );
        assert!(path.exists());

        let content = fs::read_to_string(&path).unwrap();
        let deserialized: Lesson = serde_yaml::from_str(&content).unwrap();
        assert_eq!(deserialized, lesson);
    }

    #[test]
    fn test_write_pattern_creates_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let pattern = sample_pattern();
        let path = write_pattern(root, &pattern).unwrap();

        assert_eq!(
            path,
            root.join("knowledge/patterns/one-module-per-concept.yaml")
        );
        assert!(path.exists());

        let content = fs::read_to_string(&path).unwrap();
        let deserialized: Pattern = serde_yaml::from_str(&content).unwrap();
        assert_eq!(deserialized, pattern);
    }

    #[test]
    fn test_write_glossary_entry_creates_file() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let entry = sample_glossary_entry();
        let path = write_glossary_entry(root, &entry).unwrap();

        assert_eq!(path, root.join("knowledge/glossary/terms.yaml"));
        assert!(path.exists());

        let content = fs::read_to_string(&path).unwrap();
        let entries: Vec<GlossaryEntry> = serde_yaml::from_str(&content).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn test_write_glossary_entry_appends() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let entry1 = sample_glossary_entry();
        write_glossary_entry(root, &entry1).unwrap();

        let entry2 = GlossaryEntry {
            term: "embedding".to_string(),
            definition: "A dense vector representation of text".to_string(),
            context: "Used for semantic similarity search".to_string(),
            contributed_by: "agent-session-5".to_string(),
            created_at: "2026-03-09T01:00:00Z".to_string(),
        };
        write_glossary_entry(root, &entry2).unwrap();

        let content =
            fs::read_to_string(root.join("knowledge/glossary/terms.yaml")).unwrap();
        let entries: Vec<GlossaryEntry> = serde_yaml::from_str(&content).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], entry1);
        assert_eq!(entries[1], entry2);
    }

    #[test]
    fn test_write_glossary_entry_updates_existing() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let entry = sample_glossary_entry();
        write_glossary_entry(root, &entry).unwrap();

        let updated = GlossaryEntry {
            term: "chunk".to_string(),
            definition: "An updated definition of chunk".to_string(),
            context: "Updated context".to_string(),
            contributed_by: "agent-session-6".to_string(),
            created_at: "2026-03-09T02:00:00Z".to_string(),
        };
        write_glossary_entry(root, &updated).unwrap();

        let content =
            fs::read_to_string(root.join("knowledge/glossary/terms.yaml")).unwrap();
        let entries: Vec<GlossaryEntry> = serde_yaml::from_str(&content).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], updated);
    }

    #[test]
    fn test_write_decision_yaml_is_human_readable() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let decision = sample_decision();
        let path = write_decision(root, &decision).unwrap();
        let content = fs::read_to_string(path).unwrap();

        // YAML should contain human-readable field names, not JSON
        assert!(content.contains("title:"));
        assert!(content.contains("context:"));
        assert!(content.contains("consequences:"));
        assert!(!content.contains("{")); // Not JSON
    }

    #[test]
    fn test_write_pattern_no_date_prefix() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let pattern = sample_pattern();
        let path = write_pattern(root, &pattern).unwrap();
        let filename = path.file_name().unwrap().to_str().unwrap();

        // Pattern files should NOT have a date prefix
        assert_eq!(filename, "one-module-per-concept.yaml");
        assert!(!filename.starts_with("2026"));
    }

    #[test]
    fn test_write_decision_creates_parent_dirs() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("deep/nested/store");
        // Don't call make_store_dirs — let write_decision create parents

        let decision = sample_decision();
        let path = write_decision(&root, &decision).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_knowledge_embedding_path() {
        let yaml = Path::new("/store/knowledge/decisions/2026-03-09_auth-strategy.yaml");
        let emb = knowledge_embedding_path(yaml);
        assert_eq!(
            emb,
            PathBuf::from("/store/knowledge/decisions/2026-03-09_auth-strategy.embedding.bin")
        );
    }

    #[test]
    fn test_write_knowledge_embedding_creates_binary() {
        let tmp = TempDir::new().unwrap();
        let yaml_path = tmp.path().join("test.yaml");
        fs::write(&yaml_path, "dummy").unwrap();

        let embedding = vec![1.0f32, 2.0, 3.0, 4.0];
        let emb_path = write_knowledge_embedding(&yaml_path, &embedding, 4).unwrap();

        assert_eq!(emb_path, tmp.path().join("test.embedding.bin"));
        assert!(emb_path.exists());

        // Verify it's valid EGRM format
        let file = crate::read_embeddings_bin(&emb_path).unwrap();
        assert_eq!(file.count, 1);
        assert_eq!(file.dimensions, 4);
        assert_eq!(file.vectors, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_write_decision_with_embedding() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let decision = sample_decision();
        let embedding = vec![0.1f32, 0.2, 0.3];
        let yaml_path =
            write_decision_with_embedding(root, &decision, &embedding, 3).unwrap();

        // YAML should have embedding_ref set
        let content = fs::read_to_string(&yaml_path).unwrap();
        let loaded: Decision = serde_yaml::from_str(&content).unwrap();
        assert!(loaded.embedding_ref.is_some());
        let emb_ref = loaded.embedding_ref.unwrap();
        assert!(emb_ref.ends_with(".embedding.bin"));
        assert!(emb_ref.starts_with("knowledge/decisions/"));

        // Embedding binary should exist alongside YAML
        let emb_path = knowledge_embedding_path(&yaml_path);
        assert!(emb_path.exists());
    }

    #[test]
    fn test_write_lesson_with_embedding() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let lesson = sample_lesson();
        let embedding = vec![0.5f32, 0.6, 0.7];
        let yaml_path =
            write_lesson_with_embedding(root, &lesson, &embedding, 3).unwrap();

        let content = fs::read_to_string(&yaml_path).unwrap();
        let loaded: Lesson = serde_yaml::from_str(&content).unwrap();
        assert!(loaded.embedding_ref.is_some());
        assert!(loaded.embedding_ref.unwrap().ends_with(".embedding.bin"));

        let emb_path = knowledge_embedding_path(&yaml_path);
        assert!(emb_path.exists());
    }

    #[test]
    fn test_write_pattern_with_embedding() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let pattern = sample_pattern();
        let embedding = vec![0.9f32, 0.8, 0.7];
        let yaml_path =
            write_pattern_with_embedding(root, &pattern, &embedding, 3).unwrap();

        let content = fs::read_to_string(&yaml_path).unwrap();
        let loaded: Pattern = serde_yaml::from_str(&content).unwrap();
        assert!(loaded.embedding_ref.is_some());
        assert!(loaded.embedding_ref.unwrap().ends_with(".embedding.bin"));

        let emb_path = knowledge_embedding_path(&yaml_path);
        assert!(emb_path.exists());
    }

    #[test]
    fn test_decision_embed_text() {
        let d = sample_decision();
        let text = decision_embed_text(&d);
        assert!(text.contains(&d.title));
        assert!(text.contains(&d.context));
        assert!(text.contains(&d.decision));
    }

    #[test]
    fn test_lesson_embed_text() {
        let l = sample_lesson();
        let text = lesson_embed_text(&l);
        assert!(text.contains(&l.title));
        assert!(text.contains(&l.description));
        assert!(text.contains(&l.resolution));
    }

    #[test]
    fn test_pattern_embed_text() {
        let p = sample_pattern();
        let text = pattern_embed_text(&p);
        assert!(text.contains(&p.name));
        assert!(text.contains(&p.description));
    }

    #[test]
    fn test_scan_knowledge_embeddings_empty() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        let results = scan_knowledge_embeddings(root).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_scan_knowledge_embeddings_finds_items() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        make_store_dirs(root);

        // Write a decision with embedding
        let decision = sample_decision();
        write_decision_with_embedding(root, &decision, &[0.1, 0.2, 0.3], 3).unwrap();

        // Write a pattern with embedding
        let pattern = sample_pattern();
        write_pattern_with_embedding(root, &pattern, &[0.4, 0.5, 0.6], 3).unwrap();

        // Write a lesson WITHOUT embedding (should not appear)
        let lesson = sample_lesson();
        write_lesson(root, &lesson).unwrap();

        let results = scan_knowledge_embeddings(root).unwrap();
        assert_eq!(results.len(), 2);

        let kinds: Vec<&str> = results.iter().map(|r| r.kind.as_str()).collect();
        assert!(kinds.contains(&"decision"));
        assert!(kinds.contains(&"pattern"));
    }

    #[test]
    fn test_scan_knowledge_embeddings_no_dirs() {
        let tmp = TempDir::new().unwrap();
        // Don't create knowledge dirs
        let results = scan_knowledge_embeddings(tmp.path()).unwrap();
        assert!(results.is_empty());
    }
}
