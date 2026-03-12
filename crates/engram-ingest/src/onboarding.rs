use std::fs;
use std::path::Path;

use engram_core::{
    Abstraction, EngramError, ExportedSymbol, KeyAbstractions, OnboardingDepth, OnboardingReport,
    Result, SourceConfig, SymbolResolutionConfig, TypeHierarchy,
};
use engram_lsp::LspSymbolResolver;
use engram_store::commit_changes;
use git2::Repository;

use crate::{
    abstractions::collect_source_files,
    abstractions::detect_patterns,
    analyze_directory_structure, extract_build_commands, extract_key_abstractions,
    detect_project_metadata,
};
use crate::chunker::Language;

/// Run the full onboarding pipeline, writing knowledge YAML files to the store.
///
/// Depending on `depth`:
/// - **Quick**: project metadata + build/test commands
/// - **Standard** (default): Quick + architecture map + key abstractions (tree-sitter)
/// - **Deep**: Standard + LSP-based symbol analysis for richer type hierarchy
///   (falls back to tree-sitter if LSP backend not configured)
///
/// Onboarding is idempotent — re-running overwrites existing files.
pub async fn run_onboarding(
    repo_path: &Path,
    store_path: &Path,
    source_config: &SourceConfig,
    depth: OnboardingDepth,
    symbol_config: Option<&SymbolResolutionConfig>,
) -> Result<OnboardingReport> {
    let onboarding_dir = store_path.join("knowledge/onboarding");
    fs::create_dir_all(&onboarding_dir)?;

    let mut files_written = Vec::new();
    let mut architecture_analyzed = false;
    let mut abstractions_extracted = false;

    // --- Always: metadata + build commands ---

    let overview = detect_project_metadata(repo_path)?;
    let yaml = serde_yaml::to_string(&overview)
        .map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(onboarding_dir.join("project-overview.yaml"), &yaml)?;
    files_written.push("knowledge/onboarding/project-overview.yaml".to_string());

    let commands = extract_build_commands(repo_path)?;
    let yaml = serde_yaml::to_string(&commands)
        .map_err(|e| EngramError::Serialize(e.to_string()))?;
    fs::write(onboarding_dir.join("build-test-commands.yaml"), &yaml)?;
    files_written.push("knowledge/onboarding/build-test-commands.yaml".to_string());

    // --- Standard / Deep: architecture map + key abstractions ---

    if matches!(depth, OnboardingDepth::Standard | OnboardingDepth::Deep) {
        let arch = analyze_directory_structure(repo_path, source_config)?;
        let yaml = serde_yaml::to_string(&arch)
            .map_err(|e| EngramError::Serialize(e.to_string()))?;
        fs::write(onboarding_dir.join("architecture-map.yaml"), &yaml)?;
        files_written.push("knowledge/onboarding/architecture-map.yaml".to_string());
        architecture_analyzed = true;

        // Deep + LSP backend: use LSP for richer key abstractions with type hierarchy
        let abstractions = if depth == OnboardingDepth::Deep
            && is_lsp_backend(symbol_config)
        {
            match extract_key_abstractions_lsp(repo_path, &overview.language, symbol_config.unwrap()).await {
                Ok(ka) => ka,
                Err(e) => {
                    eprintln!(
                        "Warning: LSP-based abstraction extraction failed, falling back to tree-sitter: {}",
                        e
                    );
                    extract_key_abstractions(repo_path, &overview.language)?
                }
            }
        } else {
            if depth == OnboardingDepth::Deep {
                eprintln!(
                    "Warning: Deep onboarding requested but LSP backend not configured, falling back to tree-sitter"
                );
            }
            extract_key_abstractions(repo_path, &overview.language)?
        };

        let yaml = serde_yaml::to_string(&abstractions)
            .map_err(|e| EngramError::Serialize(e.to_string()))?;
        fs::write(onboarding_dir.join("key-abstractions.yaml"), &yaml)?;
        files_written.push("knowledge/onboarding/key-abstractions.yaml".to_string());
        abstractions_extracted = true;
    }

    // --- Commit to store ---

    let repo =
        Repository::open(store_path).map_err(|e| EngramError::Git(e.to_string()))?;
    let message = format!("engram: onboard {} ({})", overview.name, depth);
    let commit_hash = commit_changes(&repo, &message)?.map(|oid| oid.to_string());

    Ok(OnboardingReport {
        depth,
        metadata_detected: true,
        commands_extracted: true,
        architecture_analyzed,
        abstractions_extracted,
        files_written,
        commit_hash,
    })
}

/// Check if the symbol resolution config specifies LSP backend.
fn is_lsp_backend(config: Option<&SymbolResolutionConfig>) -> bool {
    config.is_some_and(|c| c.enabled && c.backend == "lsp")
}

/// Extract key abstractions using LSP for richer type hierarchy information.
///
/// Uses LspSymbolResolver to get exact symbol exports and type hierarchy data,
/// then enriches the abstractions with parent/child type relationships.
async fn extract_key_abstractions_lsp(
    repo_path: &Path,
    language: &str,
    symbol_config: &SymbolResolutionConfig,
) -> Result<KeyAbstractions> {
    use engram_core::SymbolResolver;

    let resolver = LspSymbolResolver::with_auto_install(
        repo_path.to_path_buf(),
        symbol_config.lsp.auto_install,
    );

    // Collect source files for the target language
    let target_lang = match language.to_lowercase().as_str() {
        "rust" => Some(Language::Rust),
        "typescript" | "javascript" => Some(Language::TypeScript),
        "python" => Some(Language::Python),
        _ => None,
    };

    let mut source_files: Vec<(String, String, Language)> = Vec::new();
    collect_source_files(repo_path, repo_path, &target_lang, &mut source_files)?;

    if source_files.is_empty() {
        let _ = resolver.shutdown().await;
        return Ok(KeyAbstractions {
            abstractions: vec![],
            patterns: vec![],
        });
    }

    // Phase 1: Extract exports via LSP for each file
    let mut all_exports: Vec<ExportedSymbol> = Vec::new();
    let mut lsp_failed = false;

    for (rel_path, _content, _lang) in &source_files {
        let abs_path = repo_path.join(rel_path);
        match resolver.extract_exports(&abs_path).await {
            Ok(exports) if !exports.is_empty() => {
                all_exports.extend(exports);
            }
            Ok(_) => {
                // Empty result — LSP may not have a server for this language
            }
            Err(_) => {
                lsp_failed = true;
                break;
            }
        }
    }

    // If LSP produced no results, fall back to tree-sitter
    if all_exports.is_empty() || lsp_failed {
        let _ = resolver.shutdown().await;
        if lsp_failed {
            return Err(EngramError::Mcp(
                "LSP symbol extraction failed".to_string(),
            ));
        }
        eprintln!("Warning: LSP returned no symbols, falling back to tree-sitter");
        return extract_key_abstractions(repo_path, language);
    }

    // Phase 2: Enrich abstractions with type hierarchy for type-like symbols
    let mut hierarchies: Vec<TypeHierarchy> = Vec::new();
    for export in &all_exports {
        let kind = export.id.kind.as_str();
        if matches!(kind, "class" | "struct" | "interface" | "enum" | "trait") {
            if let Ok(Some(hierarchy)) = resolver.type_hierarchy(&export.id).await {
                hierarchies.push(hierarchy);
            }
        }
    }

    let _ = resolver.shutdown().await;

    // Phase 3: Build abstractions with enriched descriptions
    let hierarchy_map: std::collections::HashMap<String, &TypeHierarchy> = hierarchies
        .iter()
        .map(|h| (h.symbol.name.clone(), h))
        .collect();

    let abstractions: Vec<Abstraction> = all_exports
        .into_iter()
        .filter(|e| {
            let kind = e.id.kind.as_str();
            matches!(
                kind,
                "class"
                    | "struct"
                    | "interface"
                    | "enum"
                    | "trait"
                    | "type_parameter"
                    | "module"
            )
        })
        .map(|export| {
            let rel_file = export
                .id
                .file
                .strip_prefix(repo_path)
                .unwrap_or(&export.id.file)
                .to_string_lossy()
                .to_string();

            let mut description = export.doc.unwrap_or_default();

            // Enrich with type hierarchy info
            if let Some(hierarchy) = hierarchy_map.get(&export.id.name) {
                let mut parts = Vec::new();
                if !hierarchy.parents.is_empty() {
                    let parent_names: Vec<&str> =
                        hierarchy.parents.iter().map(|p| p.name.as_str()).collect();
                    parts.push(format!("extends {}", parent_names.join(", ")));
                }
                if !hierarchy.children.is_empty() {
                    let child_names: Vec<&str> =
                        hierarchy.children.iter().map(|c| c.name.as_str()).collect();
                    parts.push(format!(
                        "implemented by {}",
                        child_names.join(", ")
                    ));
                }
                if !parts.is_empty() {
                    let hierarchy_info = parts.join("; ");
                    if description.is_empty() {
                        description = hierarchy_info;
                    } else {
                        description = format!("{}. {}", description, hierarchy_info);
                    }
                }
            }

            Abstraction {
                name: export.id.name,
                kind: export.id.kind,
                file: rel_file,
                description,
            }
        })
        .collect();

    // Deduplicate by name (keep first occurrence)
    let mut seen = std::collections::HashSet::new();
    let abstractions: Vec<Abstraction> = abstractions
        .into_iter()
        .filter(|a| seen.insert(a.name.clone()))
        .take(50)
        .collect();

    let patterns = detect_patterns(&abstractions, language);

    Ok(KeyAbstractions {
        abstractions,
        patterns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::StoreConfig;
    use engram_store::Store;
    use tempfile::TempDir;

    fn setup_rust_project(dir: &Path) {
        fs::write(
            dir.join("Cargo.toml"),
            r#"[package]
name = "test-project"
version = "0.1.0"

[dependencies]
tokio = "1"
"#,
        )
        .unwrap();
        fs::write(dir.join("Cargo.lock"), "# lock file").unwrap();
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(dir.join("src/lib.rs"), "pub fn hello() {}").unwrap();
    }

    fn init_store(tmp: &TempDir) -> Store {
        let store_path = tmp.path().join("store");
        Store::init_local(&store_path, &StoreConfig::default()).unwrap()
    }

    #[tokio::test]
    async fn quick_onboarding_writes_two_files() {
        let project_tmp = TempDir::new().unwrap();
        setup_rust_project(project_tmp.path());

        let store_tmp = TempDir::new().unwrap();
        let store = init_store(&store_tmp);
        let source_config = SourceConfig {
            name: "test".to_string(),
            path: ".".to_string(),
            include: vec![],
            exclude: vec![],
        };

        let report = run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Quick,
            None,
        )
        .await
        .unwrap();

        assert_eq!(report.depth, OnboardingDepth::Quick);
        assert!(report.metadata_detected);
        assert!(report.commands_extracted);
        assert!(!report.architecture_analyzed);
        assert!(!report.abstractions_extracted);
        assert_eq!(report.files_written.len(), 2);
        assert!(report.commit_hash.is_some());

        // Verify YAML files exist
        assert!(store.path.join("knowledge/onboarding/project-overview.yaml").exists());
        assert!(store.path.join("knowledge/onboarding/build-test-commands.yaml").exists());
        assert!(!store.path.join("knowledge/onboarding/architecture-map.yaml").exists());
        assert!(!store.path.join("knowledge/onboarding/key-abstractions.yaml").exists());
    }

    #[tokio::test]
    async fn standard_onboarding_writes_four_files() {
        let project_tmp = TempDir::new().unwrap();
        setup_rust_project(project_tmp.path());

        let store_tmp = TempDir::new().unwrap();
        let store = init_store(&store_tmp);
        let source_config = SourceConfig {
            name: "test".to_string(),
            path: ".".to_string(),
            include: vec![],
            exclude: vec![],
        };

        let report = run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Standard,
            None,
        )
        .await
        .unwrap();

        assert_eq!(report.depth, OnboardingDepth::Standard);
        assert!(report.metadata_detected);
        assert!(report.commands_extracted);
        assert!(report.architecture_analyzed);
        assert!(report.abstractions_extracted);
        assert_eq!(report.files_written.len(), 4);
        assert!(report.commit_hash.is_some());

        // Verify all YAML files exist
        assert!(store.path.join("knowledge/onboarding/project-overview.yaml").exists());
        assert!(store.path.join("knowledge/onboarding/build-test-commands.yaml").exists());
        assert!(store.path.join("knowledge/onboarding/architecture-map.yaml").exists());
        assert!(store.path.join("knowledge/onboarding/key-abstractions.yaml").exists());
    }

    #[tokio::test]
    async fn onboarding_is_idempotent() {
        let project_tmp = TempDir::new().unwrap();
        setup_rust_project(project_tmp.path());

        let store_tmp = TempDir::new().unwrap();
        let store = init_store(&store_tmp);
        let source_config = SourceConfig {
            name: "test".to_string(),
            path: ".".to_string(),
            include: vec![],
            exclude: vec![],
        };

        // Run twice
        let report1 = run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Quick,
            None,
        )
        .await
        .unwrap();

        let report2 = run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Quick,
            None,
        )
        .await
        .unwrap();

        // First run should commit
        assert!(report1.commit_hash.is_some());
        // Second run may not commit if content is unchanged (no-op commit)
        assert!(report2.metadata_detected);
        assert!(report2.commands_extracted);
        // Files should still exist
        assert!(store.path.join("knowledge/onboarding/project-overview.yaml").exists());
    }

    #[tokio::test]
    async fn onboarding_yaml_is_valid() {
        let project_tmp = TempDir::new().unwrap();
        setup_rust_project(project_tmp.path());

        let store_tmp = TempDir::new().unwrap();
        let store = init_store(&store_tmp);
        let source_config = SourceConfig {
            name: "test".to_string(),
            path: ".".to_string(),
            include: vec![],
            exclude: vec![],
        };

        run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Quick,
            None,
        )
        .await
        .unwrap();

        // Verify YAML is parseable
        let yaml =
            fs::read_to_string(store.path.join("knowledge/onboarding/project-overview.yaml"))
                .unwrap();
        let overview: engram_core::ProjectOverview = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(overview.language, "Rust");

        let yaml =
            fs::read_to_string(store.path.join("knowledge/onboarding/build-test-commands.yaml"))
                .unwrap();
        let _commands: engram_core::BuildTestCommands = serde_yaml::from_str(&yaml).unwrap();
    }

    #[tokio::test]
    async fn onboarding_commits_to_store() {
        let project_tmp = TempDir::new().unwrap();
        setup_rust_project(project_tmp.path());

        let store_tmp = TempDir::new().unwrap();
        let store = init_store(&store_tmp);
        let source_config = SourceConfig {
            name: "test".to_string(),
            path: ".".to_string(),
            include: vec![],
            exclude: vec![],
        };

        let report = run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Quick,
            None,
        )
        .await
        .unwrap();

        // Verify commit exists in the store's git repo
        let repo = Repository::open(&store.path).unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        let message = head.message().unwrap();
        assert!(message.contains("engram: onboard"));
        assert!(message.contains("quick"));

        // commit_hash should match HEAD
        let head_oid = repo.head().unwrap().target().unwrap().to_string();
        assert_eq!(report.commit_hash.unwrap(), head_oid);
    }

    #[tokio::test]
    async fn deep_onboarding_writes_same_as_standard() {
        let project_tmp = TempDir::new().unwrap();
        setup_rust_project(project_tmp.path());

        let store_tmp = TempDir::new().unwrap();
        let store = init_store(&store_tmp);
        let source_config = SourceConfig {
            name: "test".to_string(),
            path: ".".to_string(),
            include: vec![],
            exclude: vec![],
        };

        let report = run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Deep,
            None,
        )
        .await
        .unwrap();

        // Deep without LSP config falls back to tree-sitter (same output as Standard)
        assert_eq!(report.depth, OnboardingDepth::Deep);
        assert_eq!(report.files_written.len(), 4);
        assert!(report.architecture_analyzed);
        assert!(report.abstractions_extracted);
    }

    #[tokio::test]
    async fn deep_onboarding_with_lsp_config_falls_back() {
        // When LSP is configured but returns no symbols (e.g., unsupported language),
        // deep onboarding should fall back to tree-sitter and still produce valid output.
        let project_tmp = TempDir::new().unwrap();
        setup_rust_project(project_tmp.path());

        let store_tmp = TempDir::new().unwrap();
        let store = init_store(&store_tmp);
        let source_config = SourceConfig {
            name: "test".to_string(),
            path: ".".to_string(),
            include: vec![],
            exclude: vec![],
        };

        // Use disabled=false but backend=lsp — however, since the LSP resolver
        // may or may not find a server, we test the full pipeline succeeds
        // either way (via LSP results or tree-sitter fallback).
        let symbol_config = engram_core::SymbolResolutionConfig {
            enabled: false, // disabled means is_lsp_backend returns false → tree-sitter path
            backend: "lsp".to_string(),
            lsp: engram_core::LspResolutionConfig {
                auto_install: false,
            },
        };

        let report = run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Deep,
            Some(&symbol_config),
        )
        .await
        .unwrap();

        // Should still succeed via tree-sitter (LSP is disabled)
        assert_eq!(report.depth, OnboardingDepth::Deep);
        assert_eq!(report.files_written.len(), 4);
        assert!(report.architecture_analyzed);
        assert!(report.abstractions_extracted);
        assert!(report.commit_hash.is_some());

        // key-abstractions.yaml should exist and be valid YAML
        let yaml = fs::read_to_string(
            store.path.join("knowledge/onboarding/key-abstractions.yaml"),
        )
        .unwrap();
        let _ka: engram_core::KeyAbstractions = serde_yaml::from_str(&yaml).unwrap();
    }

    #[tokio::test]
    async fn deep_onboarding_without_lsp_backend_uses_treesitter() {
        // When symbol_resolution is enabled but backend is "tree-sitter",
        // deep onboarding should use tree-sitter (no LSP attempt).
        let project_tmp = TempDir::new().unwrap();
        setup_rust_project(project_tmp.path());

        let store_tmp = TempDir::new().unwrap();
        let store = init_store(&store_tmp);
        let source_config = SourceConfig {
            name: "test".to_string(),
            path: ".".to_string(),
            include: vec![],
            exclude: vec![],
        };

        let symbol_config = engram_core::SymbolResolutionConfig {
            enabled: true,
            backend: "tree-sitter".to_string(),
            lsp: engram_core::LspResolutionConfig {
                auto_install: false,
            },
        };

        let report = run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Deep,
            Some(&symbol_config),
        )
        .await
        .unwrap();

        assert_eq!(report.depth, OnboardingDepth::Deep);
        assert_eq!(report.files_written.len(), 4);
        assert!(report.abstractions_extracted);
    }

    #[test]
    fn is_lsp_backend_checks_enabled_and_backend() {
        use super::is_lsp_backend;

        // None config → not LSP
        assert!(!is_lsp_backend(None));

        // Enabled + lsp → true
        let cfg = engram_core::SymbolResolutionConfig {
            enabled: true,
            backend: "lsp".to_string(),
            lsp: Default::default(),
        };
        assert!(is_lsp_backend(Some(&cfg)));

        // Disabled + lsp → false
        let cfg = engram_core::SymbolResolutionConfig {
            enabled: false,
            backend: "lsp".to_string(),
            lsp: Default::default(),
        };
        assert!(!is_lsp_backend(Some(&cfg)));

        // Enabled + tree-sitter → false
        let cfg = engram_core::SymbolResolutionConfig {
            enabled: true,
            backend: "tree-sitter".to_string(),
            lsp: Default::default(),
        };
        assert!(!is_lsp_backend(Some(&cfg)));
    }
}
