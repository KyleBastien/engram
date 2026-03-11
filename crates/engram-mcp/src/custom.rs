use std::path::Path;

use serde::Deserialize;

use crate::server::ModeBehavior;

/// Custom context definition loaded from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct CustomContextDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tools: CustomContextTools,
    #[serde(default)]
    pub search: CustomContextSearch,
}

/// Tool configuration for custom contexts.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CustomContextTools {
    /// Tool names to exclude from the default set.
    #[serde(default)]
    pub exclude: Vec<String>,
}

/// Search configuration for custom contexts.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CustomContextSearch {
    /// Whether to use compact output by default.
    #[serde(default)]
    pub compact_by_default: bool,
    /// Whether to include knowledge sidecar results.
    #[serde(default)]
    pub knowledge_sidecar: bool,
}

/// Custom mode definition loaded from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct CustomModeDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub search: CustomModeSearch,
}

/// Search configuration for custom modes.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CustomModeSearch {
    #[serde(default)]
    pub knowledge: CustomModeKnowledge,
}

/// Knowledge search parameters for custom modes.
#[derive(Debug, Clone, Deserialize)]
pub struct CustomModeKnowledge {
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_min_relevance")]
    pub min_relevance: f64,
    #[serde(default = "default_boost")]
    pub boost_decisions: f64,
    #[serde(default = "default_boost")]
    pub boost_patterns: f64,
}

fn default_top_k() -> usize {
    5
}
fn default_min_relevance() -> f64 {
    0.6
}
fn default_boost() -> f64 {
    1.0
}

impl Default for CustomModeKnowledge {
    fn default() -> Self {
        Self {
            top_k: default_top_k(),
            min_relevance: default_min_relevance(),
            boost_decisions: default_boost(),
            boost_patterns: default_boost(),
        }
    }
}

impl CustomModeDef {
    /// Convert to ModeBehavior for use in the mode tracker.
    pub fn to_behavior(&self) -> ModeBehavior {
        ModeBehavior {
            knowledge_top_k: self.search.knowledge.top_k,
            min_relevance: self.search.knowledge.min_relevance,
            boost_decisions: self.search.knowledge.boost_decisions,
            boost_patterns: self.search.knowledge.boost_patterns,
        }
    }
}

/// Registry of custom context and mode definitions loaded from YAML files.
#[derive(Debug, Clone, Default)]
pub struct CustomDefinitions {
    pub contexts: Vec<CustomContextDef>,
    pub modes: Vec<CustomModeDef>,
}

impl CustomDefinitions {
    /// Load custom definitions by scanning `.engram/contexts/` and `.engram/modes/`
    /// directories under the given store path.
    pub fn load_from_store(store_path: &Path) -> Self {
        let contexts = load_yaml_dir::<CustomContextDef>(&store_path.join(".engram/contexts"));
        let modes = load_yaml_dir::<CustomModeDef>(&store_path.join(".engram/modes"));
        Self { contexts, modes }
    }

    /// Find a custom context definition by name.
    pub fn find_context(&self, name: &str) -> Option<&CustomContextDef> {
        self.contexts.iter().find(|c| c.name == name)
    }

    /// Find a custom mode definition by name.
    pub fn find_mode(&self, name: &str) -> Option<&CustomModeDef> {
        self.modes.iter().find(|m| m.name == name)
    }
}

/// Scan a directory for YAML files and deserialize each one.
/// Silently skips files that fail to parse.
fn load_yaml_dir<T: serde::de::DeserializeOwned>(dir: &Path) -> Vec<T> {
    let mut results = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return results,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "yaml" && ext != "yml" {
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => match serde_yaml::from_str::<T>(&content) {
                Ok(def) => results.push(def),
                Err(e) => {
                    eprintln!("engram-mcp: failed to parse {}: {e}", path.display());
                }
            },
            Err(e) => {
                eprintln!("engram-mcp: failed to read {}: {e}", path.display());
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_custom_context_yaml_deserialization() {
        let yaml = r#"
name: my-project
description: Custom context for my project
tools:
  exclude:
    - engram_sync
    - engram_snapshot
search:
  compact_by_default: true
  knowledge_sidecar: true
"#;
        let def: CustomContextDef = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(def.name, "my-project");
        assert_eq!(def.description, "Custom context for my project");
        assert_eq!(def.tools.exclude, vec!["engram_sync", "engram_snapshot"]);
        assert!(def.search.compact_by_default);
        assert!(def.search.knowledge_sidecar);
    }

    #[test]
    fn test_custom_context_yaml_minimal() {
        let yaml = "name: minimal\n";
        let def: CustomContextDef = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(def.name, "minimal");
        assert!(def.description.is_empty());
        assert!(def.tools.exclude.is_empty());
        assert!(!def.search.compact_by_default);
        assert!(!def.search.knowledge_sidecar);
    }

    #[test]
    fn test_custom_mode_yaml_deserialization() {
        let yaml = r#"
name: focus
description: Focused editing mode
search:
  knowledge:
    top_k: 3
    min_relevance: 0.8
    boost_decisions: 1.5
    boost_patterns: 1.2
"#;
        let def: CustomModeDef = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(def.name, "focus");
        assert_eq!(def.description, "Focused editing mode");
        assert_eq!(def.search.knowledge.top_k, 3);
        assert!((def.search.knowledge.min_relevance - 0.8).abs() < f64::EPSILON);
        assert!((def.search.knowledge.boost_decisions - 1.5).abs() < f64::EPSILON);
        assert!((def.search.knowledge.boost_patterns - 1.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_mode_yaml_minimal() {
        let yaml = "name: simple\n";
        let def: CustomModeDef = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(def.name, "simple");
        assert_eq!(def.search.knowledge.top_k, 5);
        assert!((def.search.knowledge.min_relevance - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_mode_to_behavior() {
        let yaml = r#"
name: deep
search:
  knowledge:
    top_k: 15
    min_relevance: 0.3
    boost_decisions: 2.0
    boost_patterns: 1.8
"#;
        let def: CustomModeDef = serde_yaml::from_str(yaml).unwrap();
        let behavior = def.to_behavior();
        assert_eq!(behavior.knowledge_top_k, 15);
        assert!((behavior.min_relevance - 0.3).abs() < f64::EPSILON);
        assert!((behavior.boost_decisions - 2.0).abs() < f64::EPSILON);
        assert!((behavior.boost_patterns - 1.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_from_store_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let defs = CustomDefinitions::load_from_store(tmp.path());
        assert!(defs.contexts.is_empty());
        assert!(defs.modes.is_empty());
    }

    #[test]
    fn test_load_from_store_with_files() {
        let tmp = TempDir::new().unwrap();
        let contexts_dir = tmp.path().join(".engram/contexts");
        let modes_dir = tmp.path().join(".engram/modes");
        std::fs::create_dir_all(&contexts_dir).unwrap();
        std::fs::create_dir_all(&modes_dir).unwrap();

        std::fs::write(
            contexts_dir.join("review.yaml"),
            "name: review\ndescription: Code review context\ntools:\n  exclude:\n    - engram_onboard\n",
        )
        .unwrap();

        std::fs::write(
            modes_dir.join("deep.yaml"),
            "name: deep\nsearch:\n  knowledge:\n    top_k: 20\n",
        )
        .unwrap();

        let defs = CustomDefinitions::load_from_store(tmp.path());
        assert_eq!(defs.contexts.len(), 1);
        assert_eq!(defs.contexts[0].name, "review");
        assert_eq!(defs.modes.len(), 1);
        assert_eq!(defs.modes[0].name, "deep");
        assert_eq!(defs.modes[0].search.knowledge.top_k, 20);
    }

    #[test]
    fn test_load_from_store_skips_invalid_yaml() {
        let tmp = TempDir::new().unwrap();
        let contexts_dir = tmp.path().join(".engram/contexts");
        std::fs::create_dir_all(&contexts_dir).unwrap();

        // Valid file
        std::fs::write(
            contexts_dir.join("good.yaml"),
            "name: good\n",
        )
        .unwrap();

        // Invalid YAML (missing name field)
        std::fs::write(
            contexts_dir.join("bad.yaml"),
            "not_a_context: true\n",
        )
        .unwrap();

        // Non-YAML file (should be skipped)
        std::fs::write(
            contexts_dir.join("readme.txt"),
            "not a yaml file\n",
        )
        .unwrap();

        let defs = CustomDefinitions::load_from_store(tmp.path());
        assert_eq!(defs.contexts.len(), 1);
        assert_eq!(defs.contexts[0].name, "good");
    }

    #[test]
    fn test_find_context() {
        let defs = CustomDefinitions {
            contexts: vec![
                serde_yaml::from_str("name: alpha\n").unwrap(),
                serde_yaml::from_str("name: beta\n").unwrap(),
            ],
            modes: vec![],
        };
        assert!(defs.find_context("alpha").is_some());
        assert!(defs.find_context("beta").is_some());
        assert!(defs.find_context("gamma").is_none());
    }

    #[test]
    fn test_find_mode() {
        let defs = CustomDefinitions {
            modes: vec![
                serde_yaml::from_str("name: fast\n").unwrap(),
                serde_yaml::from_str("name: slow\n").unwrap(),
            ],
            contexts: vec![],
        };
        assert!(defs.find_mode("fast").is_some());
        assert!(defs.find_mode("slow").is_some());
        assert!(defs.find_mode("medium").is_none());
    }

    #[test]
    fn test_load_yml_extension() {
        let tmp = TempDir::new().unwrap();
        let modes_dir = tmp.path().join(".engram/modes");
        std::fs::create_dir_all(&modes_dir).unwrap();

        std::fs::write(
            modes_dir.join("custom.yml"),
            "name: yml-mode\n",
        )
        .unwrap();

        let defs = CustomDefinitions::load_from_store(tmp.path());
        assert_eq!(defs.modes.len(), 1);
        assert_eq!(defs.modes[0].name, "yml-mode");
    }
}
