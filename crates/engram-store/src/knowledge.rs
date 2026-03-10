use std::fs;
use std::path::{Path, PathBuf};

use engram_core::{Decision, EngramError, GlossaryEntry, Lesson, Pattern, Result};

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
}
