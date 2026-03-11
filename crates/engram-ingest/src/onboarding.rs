use std::fs;
use std::path::Path;

use engram_core::{EngramError, OnboardingDepth, OnboardingReport, Result, SourceConfig};
use engram_store::commit_changes;
use git2::Repository;

use crate::{
    analyze_directory_structure, extract_build_commands, extract_key_abstractions,
    detect_project_metadata,
};

/// Run the full onboarding pipeline, writing knowledge YAML files to the store.
///
/// Depending on `depth`:
/// - **Quick**: project metadata + build/test commands
/// - **Standard** (default): Quick + architecture map + key abstractions
/// - **Deep**: same as Standard (reserved for future full symbol export analysis)
///
/// Onboarding is idempotent — re-running overwrites existing files.
pub async fn run_onboarding(
    repo_path: &Path,
    store_path: &Path,
    source_config: &SourceConfig,
    depth: OnboardingDepth,
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

        let abstractions = extract_key_abstractions(repo_path, &overview.language)?;
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
        )
        .await
        .unwrap();

        let report2 = run_onboarding(
            project_tmp.path(),
            &store.path,
            &source_config,
            OnboardingDepth::Quick,
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
        )
        .await
        .unwrap();

        assert_eq!(report.depth, OnboardingDepth::Deep);
        assert_eq!(report.files_written.len(), 4);
        assert!(report.architecture_analyzed);
        assert!(report.abstractions_extracted);
    }
}
