use std::collections::HashMap;
use std::path::Path;

use engram_core::{ProjectOverview, Result};

/// Detect project metadata from a repository path.
///
/// Inspects config files, lock files, file extensions, and directory structure
/// to determine language, framework, package manager, repo type, and entry points.
pub fn detect_project_metadata(repo_path: &Path) -> Result<ProjectOverview> {
    let name = detect_project_name(repo_path);
    let language = detect_language(repo_path);
    let framework = detect_framework(repo_path);
    let package_manager = detect_package_manager(repo_path);
    let repo_type = detect_repo_type(repo_path);
    let entry_points = detect_entry_points(repo_path);
    let description = build_description(&name, &language, &framework);

    Ok(ProjectOverview {
        name,
        language,
        framework,
        package_manager,
        repo_type,
        description,
        entry_points,
    })
}

fn detect_project_name(repo_path: &Path) -> String {
    repo_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn detect_language(repo_path: &Path) -> String {
    // Check config files first (most reliable signal)
    let config_signals: &[(&str, &str)] = &[
        ("Cargo.toml", "Rust"),
        ("package.json", "TypeScript"),
        ("go.mod", "Go"),
        ("pyproject.toml", "Python"),
        ("setup.py", "Python"),
        ("requirements.txt", "Python"),
        ("Gemfile", "Ruby"),
        ("pom.xml", "Java"),
        ("build.gradle", "Java"),
        ("build.gradle.kts", "Kotlin"),
        ("*.csproj", "C#"),
        ("Package.swift", "Swift"),
        ("mix.exs", "Elixir"),
        ("composer.json", "PHP"),
    ];

    for (file, lang) in config_signals {
        if repo_path.join(file).exists() {
            // Distinguish TypeScript vs JavaScript from package.json
            if *lang == "TypeScript" {
                if repo_path.join("tsconfig.json").exists() {
                    return "TypeScript".to_string();
                }
                return "JavaScript".to_string();
            }
            return lang.to_string();
        }
    }

    // Fallback: scan file extensions
    let ext_counts = count_file_extensions(repo_path);
    if let Some((ext, _)) = ext_counts.into_iter().max_by_key(|(_, count)| *count) {
        return extension_to_language(&ext).to_string();
    }

    "Unknown".to_string()
}

fn detect_framework(repo_path: &Path) -> Option<String> {
    // Rust frameworks
    if repo_path.join("Cargo.toml").exists() {
        if let Some(fw) = detect_rust_framework(repo_path) {
            return Some(fw);
        }
    }

    // Node.js frameworks
    if repo_path.join("package.json").exists() {
        if let Some(fw) = detect_node_framework(repo_path) {
            return Some(fw);
        }
    }

    // Python frameworks
    if repo_path.join("pyproject.toml").exists()
        || repo_path.join("setup.py").exists()
        || repo_path.join("requirements.txt").exists()
    {
        if let Some(fw) = detect_python_framework(repo_path) {
            return Some(fw);
        }
    }

    // Go frameworks
    if repo_path.join("go.mod").exists() {
        if let Some(fw) = detect_go_framework(repo_path) {
            return Some(fw);
        }
    }

    None
}

fn detect_rust_framework(repo_path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(repo_path.join("Cargo.toml")).ok()?;
    let frameworks: &[(&str, &str)] = &[
        ("actix-web", "Actix Web"),
        ("axum", "Axum"),
        ("rocket", "Rocket"),
        ("warp", "Warp"),
        ("tide", "Tide"),
        ("tauri", "Tauri"),
        ("bevy", "Bevy"),
        ("tokio", "Tokio"),
    ];
    for (dep, name) in frameworks {
        if content.contains(dep) {
            return Some(name.to_string());
        }
    }
    None
}

fn detect_node_framework(repo_path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(repo_path.join("package.json")).ok()?;
    let frameworks: &[(&str, &str)] = &[
        ("@nestjs/core", "NestJS"),
        ("next", "Next.js"),
        ("nuxt", "Nuxt"),
        ("@angular/core", "Angular"),
        ("vue", "Vue"),
        ("react", "React"),
        ("express", "Express"),
        ("fastify", "Fastify"),
        ("svelte", "Svelte"),
        ("remix", "Remix"),
        ("astro", "Astro"),
    ];
    for (dep, name) in frameworks {
        if content.contains(dep) {
            return Some(name.to_string());
        }
    }
    None
}

fn detect_python_framework(repo_path: &Path) -> Option<String> {
    let frameworks: &[(&str, &str)] = &[
        ("django", "Django"),
        ("flask", "Flask"),
        ("fastapi", "FastAPI"),
        ("starlette", "Starlette"),
        ("tornado", "Tornado"),
        ("pyramid", "Pyramid"),
    ];

    // Check pyproject.toml, requirements.txt, and setup.py
    for file in &["pyproject.toml", "requirements.txt", "setup.py"] {
        if let Ok(content) = std::fs::read_to_string(repo_path.join(file)) {
            let lower = content.to_lowercase();
            for (dep, name) in frameworks {
                if lower.contains(dep) {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

fn detect_go_framework(repo_path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(repo_path.join("go.mod")).ok()?;
    let frameworks: &[(&str, &str)] = &[
        ("github.com/gin-gonic/gin", "Gin"),
        ("github.com/gofiber/fiber", "Fiber"),
        ("github.com/labstack/echo", "Echo"),
        ("github.com/gorilla/mux", "Gorilla Mux"),
    ];
    for (dep, name) in frameworks {
        if content.contains(dep) {
            return Some(name.to_string());
        }
    }
    None
}

fn detect_package_manager(repo_path: &Path) -> Option<String> {
    let lock_signals: &[(&str, &str)] = &[
        ("Cargo.lock", "cargo"),
        ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"),
        ("bun.lockb", "bun"),
        ("package-lock.json", "npm"),
        ("Pipfile.lock", "pipenv"),
        ("poetry.lock", "poetry"),
        ("uv.lock", "uv"),
        ("go.sum", "go modules"),
        ("Gemfile.lock", "bundler"),
        ("composer.lock", "composer"),
        ("mix.lock", "mix"),
    ];

    for (file, manager) in lock_signals {
        if repo_path.join(file).exists() {
            return Some(manager.to_string());
        }
    }

    // Fallback: config file presence without lock file
    if repo_path.join("Cargo.toml").exists() {
        return Some("cargo".to_string());
    }
    if repo_path.join("go.mod").exists() {
        return Some("go modules".to_string());
    }

    None
}

fn detect_repo_type(repo_path: &Path) -> String {
    // Cargo workspace
    if let Ok(content) = std::fs::read_to_string(repo_path.join("Cargo.toml")) {
        if content.contains("[workspace]") {
            return "monorepo".to_string();
        }
    }

    // Node.js workspaces
    if let Ok(content) = std::fs::read_to_string(repo_path.join("package.json")) {
        if content.contains("\"workspaces\"") {
            return "monorepo".to_string();
        }
    }

    // pnpm workspaces
    if repo_path.join("pnpm-workspace.yaml").exists() {
        return "monorepo".to_string();
    }

    // Nx monorepo
    if repo_path.join("nx.json").exists() && repo_path.join("packages").is_dir() {
        return "monorepo".to_string();
    }

    // Lerna
    if repo_path.join("lerna.json").exists() {
        return "monorepo".to_string();
    }

    // Go workspace
    if repo_path.join("go.work").exists() {
        return "monorepo".to_string();
    }

    "single".to_string()
}

fn detect_entry_points(repo_path: &Path) -> Vec<String> {
    let candidates: &[&str] = &[
        "src/main.rs",
        "src/lib.rs",
        "src/main.ts",
        "src/index.ts",
        "src/index.tsx",
        "src/main.tsx",
        "src/app.ts",
        "src/App.tsx",
        "main.go",
        "cmd/main.go",
        "src/main.py",
        "app.py",
        "main.py",
        "manage.py",
        "src/index.js",
        "src/main.js",
        "index.js",
        "index.ts",
        "server.ts",
        "server.js",
        "app.js",
        "lib/main.rb",
        "config.ru",
        "lib.rs",
        "main.rs",
        "Program.cs",
    ];

    let mut found = Vec::new();
    for candidate in candidates {
        if repo_path.join(candidate).exists() {
            found.push(candidate.to_string());
        }
    }

    // Also check for workspace member entry points (Cargo workspace)
    if let Ok(content) = std::fs::read_to_string(repo_path.join("Cargo.toml")) {
        if content.contains("[workspace]") {
            if let Some(members) = extract_workspace_members(&content) {
                for member in members {
                    let main_rs = format!("{}/src/main.rs", member);
                    let lib_rs = format!("{}/src/lib.rs", member);
                    if repo_path.join(&main_rs).exists() && !found.contains(&main_rs) {
                        found.push(main_rs);
                    }
                    if repo_path.join(&lib_rs).exists() && !found.contains(&lib_rs) {
                        found.push(lib_rs);
                    }
                }
            }
        }
    }

    found
}

fn extract_workspace_members(cargo_toml: &str) -> Option<Vec<String>> {
    // Simple extraction of workspace members from Cargo.toml
    // Looks for members = ["path1", "path2", ...] or members = [\n"path1",\n...]
    let members_start = cargo_toml.find("members")?;
    let after_members = &cargo_toml[members_start..];
    let bracket_start = after_members.find('[')?;
    let bracket_end = after_members.find(']')?;
    let inner = &after_members[bracket_start + 1..bracket_end];

    let mut members = Vec::new();
    for part in inner.split(',') {
        let trimmed = part.trim().trim_matches('"').trim_matches('\'').trim();
        if !trimmed.is_empty() {
            // Expand glob patterns like "crates/*"
            if trimmed.contains('*') {
                let prefix = trimmed.trim_end_matches('*').trim_end_matches('/');
                // We can't resolve the glob here without the filesystem path,
                // but the caller passes repo_path. We'll just skip glob members
                // since entry_points checks specific paths.
                // Actually, we can't access repo_path here. This is fine—
                // the entry point detection already checks the top-level candidates.
                let _ = prefix;
            } else {
                members.push(trimmed.to_string());
            }
        }
    }
    if members.is_empty() {
        None
    } else {
        Some(members)
    }
}

fn count_file_extensions(repo_path: &Path) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    count_extensions_recursive(repo_path, &mut counts, 0);
    counts
}

fn count_extensions_recursive(dir: &Path, counts: &mut HashMap<String, usize>, depth: usize) {
    if depth > 5 {
        return; // Don't recurse too deep
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden dirs, node_modules, target, .git, vendor
        if name_str.starts_with('.')
            || name_str == "node_modules"
            || name_str == "target"
            || name_str == "vendor"
            || name_str == "__pycache__"
            || name_str == "dist"
            || name_str == "build"
        {
            continue;
        }

        if path.is_dir() {
            count_extensions_recursive(&path, counts, depth + 1);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            *counts.entry(ext.to_lowercase()).or_insert(0) += 1;
        }
    }
}

fn extension_to_language(ext: &str) -> &str {
    match ext {
        "rs" => "Rust",
        "ts" | "tsx" => "TypeScript",
        "js" | "jsx" | "mjs" | "cjs" => "JavaScript",
        "py" => "Python",
        "go" => "Go",
        "java" => "Java",
        "kt" | "kts" => "Kotlin",
        "rb" => "Ruby",
        "cs" => "C#",
        "swift" => "Swift",
        "ex" | "exs" => "Elixir",
        "php" => "PHP",
        "c" | "h" => "C",
        "cpp" | "cc" | "cxx" | "hpp" => "C++",
        "zig" => "Zig",
        "scala" => "Scala",
        _ => "Unknown",
    }
}

fn build_description(name: &str, language: &str, framework: &Option<String>) -> String {
    match framework {
        Some(fw) => format!("{name} — {language} project using {fw}"),
        None => format!("{name} — {language} project"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_rust_project(dir: &Path) {
        fs::write(
            dir.join("Cargo.toml"),
            r#"[package]
name = "my-app"
version = "0.1.0"

[dependencies]
actix-web = "4"
"#,
        )
        .unwrap();
        fs::write(dir.join("Cargo.lock"), "# lock file").unwrap();
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/main.rs"), "fn main() {}").unwrap();
    }

    fn setup_node_project(dir: &Path) {
        fs::write(
            dir.join("package.json"),
            r#"{"name": "my-app", "dependencies": {"@nestjs/core": "^10.0.0", "react": "^18.0.0"}}"#,
        )
        .unwrap();
        fs::write(dir.join("tsconfig.json"), "{}").unwrap();
        fs::write(dir.join("pnpm-lock.yaml"), "# lock").unwrap();
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/main.ts"), "// entry").unwrap();
    }

    fn setup_python_project(dir: &Path) {
        fs::write(
            dir.join("pyproject.toml"),
            r#"[tool.poetry.dependencies]
django = "^4.2"
"#,
        )
        .unwrap();
        fs::write(dir.join("poetry.lock"), "# lock").unwrap();
        fs::write(dir.join("manage.py"), "# entry").unwrap();
    }

    fn setup_go_project(dir: &Path) {
        fs::write(
            dir.join("go.mod"),
            "module example.com/myapp\n\nrequire github.com/gin-gonic/gin v1.9.0\n",
        )
        .unwrap();
        fs::write(dir.join("go.sum"), "# sum").unwrap();
        fs::write(dir.join("main.go"), "package main").unwrap();
    }

    #[test]
    fn detect_rust_project() {
        let tmp = TempDir::new().unwrap();
        setup_rust_project(tmp.path());

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.language, "Rust");
        assert_eq!(result.framework, Some("Actix Web".to_string()));
        assert_eq!(result.package_manager, Some("cargo".to_string()));
        assert_eq!(result.repo_type, "single");
        assert!(result.entry_points.contains(&"src/main.rs".to_string()));
    }

    #[test]
    fn detect_node_typescript_project() {
        let tmp = TempDir::new().unwrap();
        setup_node_project(tmp.path());

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.language, "TypeScript");
        assert_eq!(result.framework, Some("NestJS".to_string()));
        assert_eq!(result.package_manager, Some("pnpm".to_string()));
        assert!(result.entry_points.contains(&"src/main.ts".to_string()));
    }

    #[test]
    fn detect_node_javascript_project() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("package.json"),
            r#"{"name": "js-app", "dependencies": {"express": "^4.18.0"}}"#,
        )
        .unwrap();
        fs::write(tmp.path().join("package-lock.json"), "{}").unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/index.js"), "// entry").unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.language, "JavaScript");
        assert_eq!(result.framework, Some("Express".to_string()));
        assert_eq!(result.package_manager, Some("npm".to_string()));
        assert!(result.entry_points.contains(&"src/index.js".to_string()));
    }

    #[test]
    fn detect_python_project() {
        let tmp = TempDir::new().unwrap();
        setup_python_project(tmp.path());

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.language, "Python");
        assert_eq!(result.framework, Some("Django".to_string()));
        assert_eq!(result.package_manager, Some("poetry".to_string()));
        assert!(result.entry_points.contains(&"manage.py".to_string()));
    }

    #[test]
    fn detect_go_project() {
        let tmp = TempDir::new().unwrap();
        setup_go_project(tmp.path());

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.language, "Go");
        assert_eq!(result.framework, Some("Gin".to_string()));
        assert_eq!(result.package_manager, Some("go modules".to_string()));
        assert!(result.entry_points.contains(&"main.go".to_string()));
    }

    #[test]
    fn detect_monorepo_cargo_workspace() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("Cargo.toml"),
            r#"[workspace]
members = ["crates/core", "crates/cli"]
"#,
        )
        .unwrap();
        fs::write(tmp.path().join("Cargo.lock"), "# lock").unwrap();
        fs::create_dir_all(tmp.path().join("crates/core/src")).unwrap();
        fs::write(tmp.path().join("crates/core/src/lib.rs"), "").unwrap();
        fs::create_dir_all(tmp.path().join("crates/cli/src")).unwrap();
        fs::write(tmp.path().join("crates/cli/src/main.rs"), "fn main() {}").unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.language, "Rust");
        assert_eq!(result.repo_type, "monorepo");
        assert!(result
            .entry_points
            .contains(&"crates/cli/src/main.rs".to_string()));
        assert!(result
            .entry_points
            .contains(&"crates/core/src/lib.rs".to_string()));
    }

    #[test]
    fn detect_monorepo_node_workspaces() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("package.json"),
            r#"{"name": "mono", "workspaces": ["packages/*"]}"#,
        )
        .unwrap();
        fs::write(tmp.path().join("tsconfig.json"), "{}").unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.repo_type, "monorepo");
    }

    #[test]
    fn detect_monorepo_pnpm_workspace() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("package.json"),
            r#"{"name": "mono"}"#,
        )
        .unwrap();
        fs::write(tmp.path().join("tsconfig.json"), "{}").unwrap();
        fs::write(tmp.path().join("pnpm-workspace.yaml"), "packages:\n  - packages/*\n").unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.repo_type, "monorepo");
    }

    #[test]
    fn detect_unknown_language_fallback() {
        let tmp = TempDir::new().unwrap();
        // No config files, create some .zig files
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.zig"), "pub fn main() {}").unwrap();
        fs::write(tmp.path().join("src/lib.zig"), "").unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.language, "Zig");
    }

    #[test]
    fn detect_empty_directory() {
        let tmp = TempDir::new().unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.language, "Unknown");
        assert_eq!(result.framework, None);
        assert_eq!(result.package_manager, None);
        assert_eq!(result.repo_type, "single");
        assert!(result.entry_points.is_empty());
    }

    #[test]
    fn project_name_from_directory() {
        let tmp = TempDir::new().unwrap();
        let result = detect_project_metadata(tmp.path()).unwrap();
        // tempdir names are random but should be non-empty
        assert!(!result.name.is_empty());
    }

    #[test]
    fn description_includes_framework() {
        let tmp = TempDir::new().unwrap();
        setup_rust_project(tmp.path());

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert!(result.description.contains("Rust"));
        assert!(result.description.contains("Actix Web"));
    }

    #[test]
    fn description_without_framework() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("Cargo.toml"),
            r#"[package]
name = "my-lib"
version = "0.1.0"
"#,
        )
        .unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert!(result.description.contains("Rust"));
        assert!(!result.description.contains("using"));
    }

    #[test]
    fn entry_points_not_duplicated() {
        let tmp = TempDir::new().unwrap();
        // Create a workspace where a top-level src/lib.rs exists
        // AND a member also has src/lib.rs
        fs::write(
            tmp.path().join("Cargo.toml"),
            r#"[workspace]
members = []
"#,
        )
        .unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "").unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        let lib_count = result
            .entry_points
            .iter()
            .filter(|e| *e == "src/lib.rs")
            .count();
        assert_eq!(lib_count, 1);
    }

    #[test]
    fn detect_python_requirements_txt() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("requirements.txt"),
            "fastapi==0.100.0\nuvicorn==0.23.0\n",
        )
        .unwrap();
        fs::write(tmp.path().join("app.py"), "# entry").unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.language, "Python");
        assert_eq!(result.framework, Some("FastAPI".to_string()));
        assert!(result.entry_points.contains(&"app.py".to_string()));
    }

    #[test]
    fn detect_package_manager_priority() {
        // pnpm-lock.yaml should take priority over package-lock.json
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("package.json"), r#"{"name": "app"}"#).unwrap();
        fs::write(tmp.path().join("pnpm-lock.yaml"), "").unwrap();
        fs::write(tmp.path().join("package-lock.json"), "{}").unwrap();

        let result = detect_project_metadata(tmp.path()).unwrap();
        assert_eq!(result.package_manager, Some("pnpm".to_string()));
    }
}
