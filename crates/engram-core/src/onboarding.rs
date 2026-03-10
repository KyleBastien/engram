use serde::{Deserialize, Serialize};

/// Information about a single CLI command used during build/test/lint/run.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CommandInfo {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub framework: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
}

/// High-level overview of the project detected during onboarding.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProjectOverview {
    pub name: String,
    pub language: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub framework: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package_manager: Option<String>,
    pub repo_type: String,
    pub description: String,
    #[serde(default)]
    pub entry_points: Vec<String>,
}

/// Build, test, lint, and run commands detected for the project.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BuildTestCommands {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<CommandInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test: Option<CommandInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lint: Option<CommandInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<CommandInfo>,
}

/// Information about a directory in the project's architecture.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DirectoryInfo {
    pub path: String,
    pub purpose: String,
    #[serde(default)]
    pub key_files: Vec<String>,
}

/// A map of the project's directory structure and purpose.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ArchitectureMap {
    #[serde(default)]
    pub directories: Vec<DirectoryInfo>,
}

/// A key abstraction (class, trait, interface, module) in the project.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Abstraction {
    pub name: String,
    pub kind: String,
    pub file: String,
    pub description: String,
}

/// A recognized pattern used across the project.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PatternInfo {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub examples: Vec<String>,
}

/// Key abstractions and patterns discovered during onboarding.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KeyAbstractions {
    #[serde(default)]
    pub abstractions: Vec<Abstraction>,
    #[serde(default)]
    pub patterns: Vec<PatternInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_overview_yaml_round_trip() {
        let overview = ProjectOverview {
            name: "engram".to_string(),
            language: "Rust".to_string(),
            framework: Some("Tokio".to_string()),
            package_manager: Some("cargo".to_string()),
            repo_type: "monorepo".to_string(),
            description: "Git-backed semantic context server".to_string(),
            entry_points: vec![
                "crates/engram-cli/src/main.rs".to_string(),
                "crates/engram-mcp/src/lib.rs".to_string(),
            ],
        };

        let yaml = serde_yaml::to_string(&overview).expect("serialize to YAML");
        assert!(yaml.contains("name: engram"));
        assert!(yaml.contains("language: Rust"));
        let deserialized: ProjectOverview =
            serde_yaml::from_str(&yaml).expect("deserialize from YAML");
        assert_eq!(overview, deserialized);
    }

    #[test]
    fn project_overview_optional_fields() {
        let overview = ProjectOverview {
            name: "simple-app".to_string(),
            language: "Python".to_string(),
            framework: None,
            package_manager: None,
            repo_type: "single".to_string(),
            description: "A simple app".to_string(),
            entry_points: vec![],
        };

        let yaml = serde_yaml::to_string(&overview).expect("serialize");
        assert!(!yaml.contains("framework"));
        assert!(!yaml.contains("package_manager"));
        let deserialized: ProjectOverview = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(overview, deserialized);
    }

    #[test]
    fn command_info_yaml_round_trip() {
        let cmd = CommandInfo {
            command: "cargo build --workspace".to_string(),
            output_dir: Some("target/debug".to_string()),
            framework: None,
            config: Some("Cargo.toml".to_string()),
            port: None,
        };

        let yaml = serde_yaml::to_string(&cmd).expect("serialize");
        let deserialized: CommandInfo = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(cmd, deserialized);
    }

    #[test]
    fn build_test_commands_yaml_round_trip() {
        let commands = BuildTestCommands {
            build: Some(CommandInfo {
                command: "cargo build".to_string(),
                output_dir: Some("target/".to_string()),
                framework: None,
                config: None,
                port: None,
            }),
            test: Some(CommandInfo {
                command: "cargo test".to_string(),
                output_dir: None,
                framework: Some("built-in".to_string()),
                config: None,
                port: None,
            }),
            lint: Some(CommandInfo {
                command: "cargo clippy".to_string(),
                output_dir: None,
                framework: None,
                config: None,
                port: None,
            }),
            run: Some(CommandInfo {
                command: "cargo run -p engram-cli -- serve".to_string(),
                output_dir: None,
                framework: None,
                config: None,
                port: Some(3100),
            }),
        };

        let yaml = serde_yaml::to_string(&commands).expect("serialize");
        assert!(yaml.contains("cargo build"));
        assert!(yaml.contains("port: 3100"));
        let deserialized: BuildTestCommands = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(commands, deserialized);
    }

    #[test]
    fn build_test_commands_partial() {
        let commands = BuildTestCommands {
            build: Some(CommandInfo {
                command: "make".to_string(),
                output_dir: None,
                framework: None,
                config: None,
                port: None,
            }),
            test: None,
            lint: None,
            run: None,
        };

        let yaml = serde_yaml::to_string(&commands).expect("serialize");
        assert!(!yaml.contains("test:"));
        assert!(!yaml.contains("lint:"));
        let deserialized: BuildTestCommands = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(commands, deserialized);
    }

    #[test]
    fn architecture_map_yaml_round_trip() {
        let arch = ArchitectureMap {
            directories: vec![
                DirectoryInfo {
                    path: "crates/engram-core".to_string(),
                    purpose: "Shared types and traits".to_string(),
                    key_files: vec!["src/lib.rs".to_string(), "src/error.rs".to_string()],
                },
                DirectoryInfo {
                    path: "crates/engram-ingest".to_string(),
                    purpose: "Code parsing and chunking pipeline".to_string(),
                    key_files: vec!["src/chunker.rs".to_string()],
                },
            ],
        };

        let yaml = serde_yaml::to_string(&arch).expect("serialize");
        assert!(yaml.contains("Shared types and traits"));
        let deserialized: ArchitectureMap = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(arch, deserialized);
    }

    #[test]
    fn key_abstractions_yaml_round_trip() {
        let ka = KeyAbstractions {
            abstractions: vec![
                Abstraction {
                    name: "EmbeddingProvider".to_string(),
                    kind: "trait".to_string(),
                    file: "crates/engram-core/src/embedding.rs".to_string(),
                    description: "Async trait for embedding text into vectors".to_string(),
                },
                Abstraction {
                    name: "ChunkMetadata".to_string(),
                    kind: "struct".to_string(),
                    file: "crates/engram-core/src/chunk.rs".to_string(),
                    description: "Metadata for a code chunk".to_string(),
                },
            ],
            patterns: vec![PatternInfo {
                name: "One module per concept".to_string(),
                description: "Each type family gets its own module file".to_string(),
                examples: vec![
                    "error.rs for EngramError".to_string(),
                    "chunk.rs for ChunkMetadata".to_string(),
                ],
            }],
        };

        let yaml = serde_yaml::to_string(&ka).expect("serialize");
        assert!(yaml.contains("EmbeddingProvider"));
        assert!(yaml.contains("One module per concept"));
        let deserialized: KeyAbstractions = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(ka, deserialized);
    }

    #[test]
    fn key_abstractions_empty() {
        let ka = KeyAbstractions {
            abstractions: vec![],
            patterns: vec![],
        };

        let yaml = serde_yaml::to_string(&ka).expect("serialize");
        let deserialized: KeyAbstractions = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(ka, deserialized);
    }
}
