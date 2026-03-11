use std::collections::HashMap;
use std::fs;
use std::path::Path;

use engram_core::{ArchitectureMap, DirectoryInfo, Result, SourceConfig};
use globset::{Glob, GlobSet, GlobSetBuilder};

/// Analyze a repository's directory structure and map directories to purposes.
///
/// Walks the repo tree, identifies key directories by naming conventions,
/// lists top 5 most important files per directory, and respects
/// include/exclude globs from source config.
pub fn analyze_directory_structure(
    repo_path: &Path,
    source_config: &SourceConfig,
) -> Result<ArchitectureMap> {
    let include_set = build_glob_set(&source_config.include)?;
    let exclude_set = build_glob_set(&source_config.exclude)?;

    let mut dir_files: HashMap<String, Vec<(String, u64)>> = HashMap::new();

    collect_files(repo_path, repo_path, &include_set, &exclude_set, &mut dir_files)?;

    let mut directories: Vec<DirectoryInfo> = dir_files
        .into_iter()
        .filter_map(|(dir_path, mut files)| {
            let purpose = infer_purpose(&dir_path)?;

            // Sort by file size descending, then by name for stability
            files.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            // Prioritize entry-point files to the top
            files.sort_by(|a, b| {
                let a_entry = is_entry_point(&a.0);
                let b_entry = is_entry_point(&b.0);
                b_entry.cmp(&a_entry).then_with(|| b.1.cmp(&a.1)).then_with(|| a.0.cmp(&b.0))
            });

            let key_files: Vec<String> = files.into_iter().map(|(name, _)| name).take(5).collect();

            Some(DirectoryInfo {
                path: dir_path,
                purpose,
                key_files,
            })
        })
        .collect();

    // Sort directories by path for deterministic output
    directories.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(ArchitectureMap { directories })
}

fn build_glob_set(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = Glob::new(pattern)
            .map_err(|e| engram_core::EngramError::Config(format!("invalid glob '{}': {}", pattern, e)))?;
        builder.add(glob);
    }
    builder
        .build()
        .map_err(|e| engram_core::EngramError::Config(format!("glob set build error: {}", e)))
}

fn collect_files(
    root: &Path,
    current: &Path,
    include_set: &GlobSet,
    exclude_set: &GlobSet,
    dir_files: &mut HashMap<String, Vec<(String, u64)>>,
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
        let rel_path = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        // Skip hidden directories/files and common non-source dirs
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') || is_skipped_dir(name) {
                continue;
            }
        }

        // Check exclude globs
        if !exclude_set.is_empty() && exclude_set.is_match(&rel_path) {
            continue;
        }

        if path.is_dir() {
            collect_files(root, &path, include_set, exclude_set, dir_files)?;
        } else if path.is_file() {
            // Check include globs (empty means include all)
            if !include_set.is_empty() && !include_set.is_match(&rel_path) {
                continue;
            }

            let dir_rel = path
                .parent()
                .and_then(|p| p.strip_prefix(root).ok())
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            // Skip root-level files (we only care about directories)
            if dir_rel.is_empty() {
                continue;
            }

            let file_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);

            dir_files
                .entry(dir_rel)
                .or_default()
                .push((file_name, size));
        }
    }

    Ok(())
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

fn is_entry_point(file_name: &str) -> bool {
    matches!(
        file_name,
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

/// Infer the purpose of a directory from its name or path components.
/// Returns None if the directory has no recognized purpose.
fn infer_purpose(dir_path: &str) -> Option<String> {
    // Normalize: take the last path component for matching
    let parts: Vec<&str> = dir_path.split('/').collect();
    let leaf = parts.last().copied().unwrap_or("");

    // Check leaf name first
    if let Some(purpose) = match_directory_name(leaf) {
        return Some(purpose);
    }

    // Check full path for nested patterns like "src/auth"
    for (i, part) in parts.iter().enumerate() {
        if let Some(purpose) = match_directory_name(part) {
            // If it's a deep directory, provide context from parent
            if i > 0 && parts.len() > 2 {
                return Some(purpose);
            }
            return Some(purpose);
        }
    }

    // Top-level well-known directories
    if parts.len() == 1 {
        return match_top_level_dir(leaf);
    }

    None
}

fn match_directory_name(name: &str) -> Option<String> {
    let lower = name.to_lowercase();
    match lower.as_str() {
        "api" | "apis" | "routes" | "endpoints" => Some("API routes and endpoints".to_string()),
        "auth" | "authentication" | "authn" => Some("Authentication".to_string()),
        "authz" | "authorization" | "permissions" | "rbac" => {
            Some("Authorization and permissions".to_string())
        }
        "components" | "ui" | "widgets" => Some("UI components".to_string()),
        "pages" | "views" | "screens" => Some("Page/view definitions".to_string()),
        "hooks" => Some("React hooks".to_string()),
        "utils" | "util" | "helpers" | "common" => Some("Utility functions".to_string()),
        "lib" | "libs" | "pkg" | "packages" => Some("Shared libraries".to_string()),
        "models" | "entities" | "domain" | "schema" => {
            Some("Data models and domain types".to_string())
        }
        "services" | "service" => Some("Business logic / service layer".to_string()),
        "controllers" | "controller" | "handlers" => {
            Some("Request handlers / controllers".to_string())
        }
        "middleware" | "middlewares" => Some("Middleware".to_string()),
        "config" | "configs" | "configuration" => Some("Configuration".to_string()),
        "types" | "typings" | "interfaces" => Some("Type definitions".to_string()),
        "tests" | "test" | "__tests__" | "spec" | "specs" => Some("Tests".to_string()),
        "fixtures" | "testdata" | "test-data" | "mocks" | "stubs" => {
            Some("Test fixtures and mock data".to_string())
        }
        "migrations" | "migrate" => Some("Database migrations".to_string()),
        "db" | "database" | "repositories" | "repository" | "repo" => {
            Some("Database access layer".to_string())
        }
        "store" | "stores" | "state" | "redux" | "zustand" => {
            Some("State management".to_string())
        }
        "assets" | "static" | "public" | "resources" => {
            Some("Static assets and resources".to_string())
        }
        "styles" | "css" | "scss" | "themes" => Some("Styles and theming".to_string()),
        "i18n" | "locales" | "locale" | "translations" | "l10n" => {
            Some("Internationalization".to_string())
        }
        "docs" | "documentation" | "doc" => Some("Documentation".to_string()),
        "scripts" | "tools" | "bin" => Some("Build/dev scripts and tools".to_string()),
        "cli" => Some("Command-line interface".to_string()),
        "core" => Some("Core library / shared types".to_string()),
        "server" => Some("Server-side code".to_string()),
        "client" => Some("Client-side code".to_string()),
        "shared" => Some("Shared code between client/server".to_string()),
        "internal" => Some("Internal implementation details".to_string()),
        "cmd" => Some("CLI command definitions".to_string()),
        "proto" | "protos" | "protobuf" | "grpc" => {
            Some("Protocol buffer / gRPC definitions".to_string())
        }
        "graphql" | "gql" => Some("GraphQL schema and resolvers".to_string()),
        "providers" => Some("Provider implementations".to_string()),
        "adapters" | "adapter" => Some("Adapter implementations".to_string()),
        "crates" => Some("Rust workspace crates".to_string()),
        "plugins" | "extensions" => Some("Plugin / extension modules".to_string()),
        "jobs" | "workers" | "tasks" | "queue" => {
            Some("Background jobs and task processing".to_string())
        }
        "events" | "event" | "listeners" => Some("Event handling".to_string()),
        "errors" | "exceptions" => Some("Error types and handling".to_string()),
        "logging" | "logger" | "telemetry" | "monitoring" => {
            Some("Logging and observability".to_string())
        }
        "cache" | "caching" => Some("Caching layer".to_string()),
        "email" | "mailer" | "notifications" | "notification" => {
            Some("Email / notification delivery".to_string())
        }
        "images" | "img" | "icons" | "fonts" | "media" => {
            Some("Media assets".to_string())
        }
        "layout" | "layouts" => Some("Layout components".to_string()),
        "context" | "contexts" => Some("React context providers".to_string()),
        "guards" => Some("Route/auth guards".to_string()),
        "pipes" | "validators" | "validation" => {
            Some("Data validation and transformation".to_string())
        }
        "decorators" => Some("Decorators".to_string()),
        "dto" | "dtos" => Some("Data transfer objects".to_string()),
        "resolvers" => Some("GraphQL resolvers".to_string()),
        "modules" => Some("Feature modules".to_string()),
        _ => None,
    }
}

fn match_top_level_dir(name: &str) -> Option<String> {
    let lower = name.to_lowercase();
    match lower.as_str() {
        "src" => Some("Source code".to_string()),
        "app" => Some("Application entry point".to_string()),
        "e2e" | "integration" => Some("End-to-end / integration tests".to_string()),
        "examples" | "example" | "samples" => Some("Example code and usage".to_string()),
        "benches" | "benchmarks" => Some("Performance benchmarks".to_string()),
        "ci" | ".github" | ".circleci" => Some("CI/CD configuration".to_string()),
        "deploy" | "infra" | "terraform" | "k8s" | "kubernetes" | "helm" => {
            Some("Infrastructure and deployment".to_string())
        }
        "docker" => Some("Docker configuration".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn default_source_config() -> SourceConfig {
        SourceConfig {
            name: "test".to_string(),
            path: "/tmp/test".to_string(),
            include: vec![],
            exclude: vec![],
        }
    }

    #[test]
    fn basic_directory_structure() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Create a typical project structure
        fs::create_dir_all(root.join("src/api")).unwrap();
        fs::create_dir_all(root.join("src/auth")).unwrap();
        fs::create_dir_all(root.join("src/components")).unwrap();
        fs::create_dir_all(root.join("tests")).unwrap();

        fs::write(root.join("src/api/routes.ts"), "export const routes = {};").unwrap();
        fs::write(root.join("src/api/index.ts"), "export * from './routes';").unwrap();
        fs::write(root.join("src/auth/login.ts"), "export function login() {}").unwrap();
        fs::write(root.join("src/components/Button.tsx"), "<button/>").unwrap();
        fs::write(root.join("tests/api.test.ts"), "test('api', () => {});").unwrap();

        let config = default_source_config();
        let result = analyze_directory_structure(root, &config).unwrap();

        assert!(!result.directories.is_empty());

        let api = result.directories.iter().find(|d| d.path == "src/api");
        assert!(api.is_some());
        let api = api.unwrap();
        assert_eq!(api.purpose, "API routes and endpoints");
        assert!(api.key_files.contains(&"index.ts".to_string()));
        assert!(api.key_files.contains(&"routes.ts".to_string()));

        let auth = result.directories.iter().find(|d| d.path == "src/auth");
        assert!(auth.is_some());
        assert_eq!(auth.unwrap().purpose, "Authentication");

        let components = result.directories.iter().find(|d| d.path == "src/components");
        assert!(components.is_some());
        assert_eq!(components.unwrap().purpose, "UI components");

        let tests = result.directories.iter().find(|d| d.path == "tests");
        assert!(tests.is_some());
        assert_eq!(tests.unwrap().purpose, "Tests");
    }

    #[test]
    fn respects_include_globs() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src/api")).unwrap();
        fs::create_dir_all(root.join("src/utils")).unwrap();

        fs::write(root.join("src/api/routes.ts"), "routes").unwrap();
        fs::write(root.join("src/api/readme.md"), "# API").unwrap();
        fs::write(root.join("src/utils/helpers.ts"), "helpers").unwrap();

        let config = SourceConfig {
            name: "test".to_string(),
            path: "/tmp/test".to_string(),
            include: vec!["**/*.ts".to_string()],
            exclude: vec![],
        };

        let result = analyze_directory_structure(root, &config).unwrap();

        let api = result.directories.iter().find(|d| d.path == "src/api");
        assert!(api.is_some());
        // .md file should be excluded by include glob
        assert!(!api.unwrap().key_files.contains(&"readme.md".to_string()));
    }

    #[test]
    fn respects_exclude_globs() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src/api")).unwrap();
        fs::create_dir_all(root.join("src/tests")).unwrap();

        fs::write(root.join("src/api/routes.ts"), "routes").unwrap();
        fs::write(root.join("src/tests/api.test.ts"), "test").unwrap();

        let config = SourceConfig {
            name: "test".to_string(),
            path: "/tmp/test".to_string(),
            include: vec![],
            exclude: vec!["**/tests/**".to_string()],
        };

        let result = analyze_directory_structure(root, &config).unwrap();

        let tests = result.directories.iter().find(|d| d.path.contains("tests"));
        assert!(tests.is_none(), "tests directory should be excluded");

        let api = result.directories.iter().find(|d| d.path == "src/api");
        assert!(api.is_some());
    }

    #[test]
    fn limits_to_top_5_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src/utils")).unwrap();

        for i in 0..10 {
            let content = "x".repeat((i + 1) * 100);
            fs::write(root.join(format!("src/utils/helper_{}.ts", i)), content).unwrap();
        }

        let config = default_source_config();
        let result = analyze_directory_structure(root, &config).unwrap();

        let utils = result.directories.iter().find(|d| d.path == "src/utils").unwrap();
        assert_eq!(utils.key_files.len(), 5);
    }

    #[test]
    fn entry_point_files_prioritized() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src/api")).unwrap();

        // Small entry point file
        fs::write(root.join("src/api/index.ts"), "export").unwrap();
        // Large non-entry file
        fs::write(root.join("src/api/big_file.ts"), "x".repeat(10000)).unwrap();

        let config = default_source_config();
        let result = analyze_directory_structure(root, &config).unwrap();

        let api = result.directories.iter().find(|d| d.path == "src/api").unwrap();
        assert_eq!(api.key_files[0], "index.ts", "entry point should be first");
    }

    #[test]
    fn skips_hidden_and_build_dirs() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join(".git/objects")).unwrap();
        fs::create_dir_all(root.join("node_modules/lodash")).unwrap();
        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::create_dir_all(root.join("src/api")).unwrap();

        fs::write(root.join(".git/objects/abc"), "git obj").unwrap();
        fs::write(root.join("node_modules/lodash/index.js"), "lodash").unwrap();
        fs::write(root.join("target/debug/app"), "binary").unwrap();
        fs::write(root.join("src/api/routes.ts"), "routes").unwrap();

        let config = default_source_config();
        let result = analyze_directory_structure(root, &config).unwrap();

        for dir in &result.directories {
            assert!(!dir.path.contains(".git"));
            assert!(!dir.path.contains("node_modules"));
            assert!(!dir.path.contains("target"));
        }

        let api = result.directories.iter().find(|d| d.path == "src/api");
        assert!(api.is_some());
    }

    #[test]
    fn empty_directory_returns_empty_map() {
        let tmp = TempDir::new().unwrap();
        let config = default_source_config();
        let result = analyze_directory_structure(tmp.path(), &config).unwrap();
        assert!(result.directories.is_empty());
    }

    #[test]
    fn directories_sorted_by_path() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src/utils")).unwrap();
        fs::create_dir_all(root.join("src/api")).unwrap();
        fs::create_dir_all(root.join("docs")).unwrap();

        fs::write(root.join("src/utils/helpers.ts"), "helpers").unwrap();
        fs::write(root.join("src/api/routes.ts"), "routes").unwrap();
        fs::write(root.join("docs/readme.md"), "readme").unwrap();

        let config = default_source_config();
        let result = analyze_directory_structure(root, &config).unwrap();

        let paths: Vec<&str> = result.directories.iter().map(|d| d.path.as_str()).collect();
        let mut sorted = paths.clone();
        sorted.sort();
        assert_eq!(paths, sorted);
    }

    #[test]
    fn rust_workspace_directories() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("crates/core/src")).unwrap();
        fs::create_dir_all(root.join("crates/cli/src")).unwrap();

        fs::write(root.join("crates/core/src/lib.rs"), "pub mod error;").unwrap();
        fs::write(root.join("crates/cli/src/main.rs"), "fn main() {}").unwrap();

        let config = default_source_config();
        let result = analyze_directory_structure(root, &config).unwrap();

        let crates = result.directories.iter().find(|d| d.path == "crates");
        assert!(crates.is_none(), "crates has no direct files");

        let core_src = result.directories.iter().find(|d| d.path == "crates/core/src");
        assert!(core_src.is_some());
    }

    #[test]
    fn naming_convention_detection() {
        // Test various naming conventions
        assert_eq!(infer_purpose("src/api"), Some("API routes and endpoints".to_string()));
        assert_eq!(infer_purpose("src/auth"), Some("Authentication".to_string()));
        assert_eq!(infer_purpose("models"), Some("Data models and domain types".to_string()));
        assert_eq!(infer_purpose("middleware"), Some("Middleware".to_string()));
        assert_eq!(infer_purpose("services"), Some("Business logic / service layer".to_string()));
        assert_eq!(infer_purpose("src"), Some("Source code".to_string()));
        assert_eq!(infer_purpose("e2e"), Some("End-to-end / integration tests".to_string()));

        // Unrecognized directories should return None
        assert_eq!(infer_purpose("foobar"), None);
        assert_eq!(infer_purpose("my-random-dir"), None);
    }
}
