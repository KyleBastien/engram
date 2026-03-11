use std::path::Path;

use engram_core::{BuildTestCommands, CommandInfo, Result};

/// Extract build, test, lint, and run commands from a repository.
///
/// Inspects config files (package.json, Cargo.toml, pyproject.toml, etc.)
/// and config-file presence (jest.config.ts, .eslintrc, etc.) to determine
/// the project's build/test/lint/run commands.
pub fn extract_build_commands(repo_path: &Path) -> Result<BuildTestCommands> {
    // Rust project
    if repo_path.join("Cargo.toml").exists() {
        return Ok(extract_rust_commands(repo_path));
    }

    // Node.js project
    if repo_path.join("package.json").exists() {
        return extract_node_commands(repo_path);
    }

    // Python project
    if repo_path.join("pyproject.toml").exists()
        || repo_path.join("setup.py").exists()
        || repo_path.join("requirements.txt").exists()
    {
        return Ok(extract_python_commands(repo_path));
    }

    // Go project
    if repo_path.join("go.mod").exists() {
        return Ok(extract_go_commands());
    }

    Ok(BuildTestCommands {
        build: None,
        test: None,
        lint: None,
        run: None,
    })
}

fn extract_rust_commands(repo_path: &Path) -> BuildTestCommands {
    let is_workspace = std::fs::read_to_string(repo_path.join("Cargo.toml"))
        .map(|c| c.contains("[workspace]"))
        .unwrap_or(false);

    let build_cmd = if is_workspace {
        "cargo build --workspace"
    } else {
        "cargo build"
    };

    let test_cmd = if is_workspace {
        "cargo test --workspace"
    } else {
        "cargo test"
    };

    BuildTestCommands {
        build: Some(CommandInfo {
            command: build_cmd.to_string(),
            output_dir: Some("target/debug".to_string()),
            framework: None,
            config: Some("Cargo.toml".to_string()),
            port: None,
        }),
        test: Some(CommandInfo {
            command: test_cmd.to_string(),
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
            command: "cargo run".to_string(),
            output_dir: None,
            framework: None,
            config: None,
            port: None,
        }),
    }
}

fn extract_node_commands(repo_path: &Path) -> Result<BuildTestCommands> {
    let content = std::fs::read_to_string(repo_path.join("package.json"))?;
    let pkg: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| engram_core::EngramError::Config(e.to_string()))?;

    let scripts = pkg.get("scripts").and_then(|s| s.as_object());

    let build = scripts
        .and_then(|s| s.get("build"))
        .and_then(|v| v.as_str())
        .map(|cmd| CommandInfo {
            command: cmd.to_string(),
            output_dir: detect_node_output_dir(cmd),
            framework: detect_node_build_framework(cmd),
            config: None,
            port: None,
        });

    let test = scripts
        .and_then(|s| s.get("test"))
        .and_then(|v| v.as_str())
        .map(|cmd| CommandInfo {
            command: cmd.to_string(),
            output_dir: None,
            framework: detect_test_framework_from_cmd(cmd)
                .or_else(|| detect_test_framework_from_config(repo_path)),
            config: detect_test_config_file(repo_path),
            port: None,
        })
        .or_else(|| {
            // No test script — check for test framework config files
            detect_test_framework_from_config(repo_path).map(|fw| CommandInfo {
                command: format!("npx {}", fw.to_lowercase()),
                output_dir: None,
                framework: Some(fw),
                config: detect_test_config_file(repo_path),
                port: None,
            })
        });

    let lint = scripts
        .and_then(|s| s.get("lint"))
        .and_then(|v| v.as_str())
        .map(|cmd| CommandInfo {
            command: cmd.to_string(),
            output_dir: None,
            framework: detect_lint_framework(cmd, repo_path),
            config: detect_lint_config_file(repo_path),
            port: None,
        });

    let run = scripts
        .and_then(|s| {
            s.get("dev")
                .or_else(|| s.get("start"))
        })
        .and_then(|v| v.as_str())
        .map(|cmd| CommandInfo {
            command: cmd.to_string(),
            output_dir: None,
            framework: None,
            config: None,
            port: detect_port_from_cmd(cmd),
        });

    Ok(BuildTestCommands {
        build,
        test,
        lint,
        run,
    })
}

fn extract_python_commands(repo_path: &Path) -> BuildTestCommands {
    let test = detect_python_test(repo_path);
    let lint = detect_python_lint(repo_path);

    BuildTestCommands {
        build: None,
        test,
        lint,
        run: Some(CommandInfo {
            command: "python -m app".to_string(),
            output_dir: None,
            framework: None,
            config: None,
            port: None,
        }),
    }
}

fn detect_python_test(repo_path: &Path) -> Option<CommandInfo> {
    // Check for pytest
    if repo_path.join("pytest.ini").exists()
        || repo_path.join("conftest.py").exists()
        || repo_path.join("setup.cfg").exists()
        || has_python_dep(repo_path, "pytest")
    {
        return Some(CommandInfo {
            command: "pytest".to_string(),
            output_dir: None,
            framework: Some("pytest".to_string()),
            config: detect_pytest_config(repo_path),
            port: None,
        });
    }

    // Check for tox
    if repo_path.join("tox.ini").exists() {
        return Some(CommandInfo {
            command: "tox".to_string(),
            output_dir: None,
            framework: Some("tox".to_string()),
            config: Some("tox.ini".to_string()),
            port: None,
        });
    }

    // Fallback to unittest
    Some(CommandInfo {
        command: "python -m unittest discover".to_string(),
        output_dir: None,
        framework: Some("unittest".to_string()),
        config: None,
        port: None,
    })
}

fn detect_python_lint(repo_path: &Path) -> Option<CommandInfo> {
    // Check for ruff
    if repo_path.join("ruff.toml").exists()
        || repo_path.join(".ruff.toml").exists()
        || has_python_dep(repo_path, "ruff")
    {
        return Some(CommandInfo {
            command: "ruff check .".to_string(),
            output_dir: None,
            framework: Some("ruff".to_string()),
            config: if repo_path.join("ruff.toml").exists() {
                Some("ruff.toml".to_string())
            } else if repo_path.join(".ruff.toml").exists() {
                Some(".ruff.toml".to_string())
            } else {
                None
            },
            port: None,
        });
    }

    // Check for flake8
    if repo_path.join(".flake8").exists()
        || repo_path.join("setup.cfg").exists()
        || has_python_dep(repo_path, "flake8")
    {
        return Some(CommandInfo {
            command: "flake8 .".to_string(),
            output_dir: None,
            framework: Some("flake8".to_string()),
            config: if repo_path.join(".flake8").exists() {
                Some(".flake8".to_string())
            } else {
                None
            },
            port: None,
        });
    }

    None
}

fn has_python_dep(repo_path: &Path, dep: &str) -> bool {
    for file in &["pyproject.toml", "requirements.txt", "setup.py", "Pipfile"] {
        if let Ok(content) = std::fs::read_to_string(repo_path.join(file)) {
            if content.to_lowercase().contains(dep) {
                return true;
            }
        }
    }
    false
}

fn detect_pytest_config(repo_path: &Path) -> Option<String> {
    if repo_path.join("pytest.ini").exists() {
        return Some("pytest.ini".to_string());
    }
    if repo_path.join("pyproject.toml").exists() {
        if let Ok(content) = std::fs::read_to_string(repo_path.join("pyproject.toml")) {
            if content.contains("[tool.pytest") {
                return Some("pyproject.toml".to_string());
            }
        }
    }
    if repo_path.join("setup.cfg").exists() {
        if let Ok(content) = std::fs::read_to_string(repo_path.join("setup.cfg")) {
            if content.contains("[tool:pytest]") {
                return Some("setup.cfg".to_string());
            }
        }
    }
    None
}

fn extract_go_commands() -> BuildTestCommands {
    BuildTestCommands {
        build: Some(CommandInfo {
            command: "go build ./...".to_string(),
            output_dir: None,
            framework: None,
            config: Some("go.mod".to_string()),
            port: None,
        }),
        test: Some(CommandInfo {
            command: "go test ./...".to_string(),
            output_dir: None,
            framework: Some("built-in".to_string()),
            config: None,
            port: None,
        }),
        lint: Some(CommandInfo {
            command: "golangci-lint run".to_string(),
            output_dir: None,
            framework: None,
            config: None,
            port: None,
        }),
        run: Some(CommandInfo {
            command: "go run .".to_string(),
            output_dir: None,
            framework: None,
            config: None,
            port: None,
        }),
    }
}

// --- Node.js helpers ---

fn detect_node_output_dir(cmd: &str) -> Option<String> {
    if cmd.contains("tsc") {
        return Some("dist".to_string());
    }
    if cmd.contains("next build") {
        return Some(".next".to_string());
    }
    if cmd.contains("vite build") || cmd.contains("webpack") {
        return Some("dist".to_string());
    }
    None
}

fn detect_node_build_framework(cmd: &str) -> Option<String> {
    if cmd.contains("next") {
        return Some("Next.js".to_string());
    }
    if cmd.contains("vite") {
        return Some("Vite".to_string());
    }
    if cmd.contains("webpack") {
        return Some("Webpack".to_string());
    }
    if cmd.contains("tsc") {
        return Some("TypeScript".to_string());
    }
    if cmd.contains("esbuild") {
        return Some("esbuild".to_string());
    }
    None
}

fn detect_test_framework_from_cmd(cmd: &str) -> Option<String> {
    if cmd.contains("jest") {
        return Some("Jest".to_string());
    }
    if cmd.contains("vitest") {
        return Some("Vitest".to_string());
    }
    if cmd.contains("mocha") {
        return Some("Mocha".to_string());
    }
    if cmd.contains("ava") {
        return Some("Ava".to_string());
    }
    if cmd.contains("playwright") {
        return Some("Playwright".to_string());
    }
    if cmd.contains("cypress") {
        return Some("Cypress".to_string());
    }
    None
}

fn detect_test_framework_from_config(repo_path: &Path) -> Option<String> {
    let config_signals: &[(&str, &str)] = &[
        ("jest.config.ts", "Jest"),
        ("jest.config.js", "Jest"),
        ("jest.config.mjs", "Jest"),
        ("vitest.config.ts", "Vitest"),
        ("vitest.config.js", "Vitest"),
        (".mocharc.yml", "Mocha"),
        (".mocharc.json", "Mocha"),
        ("playwright.config.ts", "Playwright"),
        ("playwright.config.js", "Playwright"),
        ("cypress.config.ts", "Cypress"),
        ("cypress.config.js", "Cypress"),
    ];

    for (file, framework) in config_signals {
        if repo_path.join(file).exists() {
            return Some(framework.to_string());
        }
    }
    None
}

fn detect_test_config_file(repo_path: &Path) -> Option<String> {
    let configs: &[&str] = &[
        "jest.config.ts",
        "jest.config.js",
        "jest.config.mjs",
        "vitest.config.ts",
        "vitest.config.js",
        ".mocharc.yml",
        ".mocharc.json",
        "playwright.config.ts",
        "playwright.config.js",
        "cypress.config.ts",
        "cypress.config.js",
    ];

    for file in configs {
        if repo_path.join(file).exists() {
            return Some(file.to_string());
        }
    }
    None
}

fn detect_lint_framework(cmd: &str, repo_path: &Path) -> Option<String> {
    if cmd.contains("eslint") {
        return Some("ESLint".to_string());
    }
    if cmd.contains("biome") {
        return Some("Biome".to_string());
    }
    if cmd.contains("prettier") {
        return Some("Prettier".to_string());
    }
    // Fall back to config file detection
    detect_lint_framework_from_config(repo_path)
}

fn detect_lint_framework_from_config(repo_path: &Path) -> Option<String> {
    let configs: &[(&str, &str)] = &[
        (".eslintrc.js", "ESLint"),
        (".eslintrc.json", "ESLint"),
        (".eslintrc.yml", "ESLint"),
        ("eslint.config.js", "ESLint"),
        ("eslint.config.mjs", "ESLint"),
        ("biome.json", "Biome"),
        (".prettierrc", "Prettier"),
        (".prettierrc.json", "Prettier"),
    ];

    for (file, framework) in configs {
        if repo_path.join(file).exists() {
            return Some(framework.to_string());
        }
    }
    None
}

fn detect_lint_config_file(repo_path: &Path) -> Option<String> {
    let configs: &[&str] = &[
        ".eslintrc.js",
        ".eslintrc.json",
        ".eslintrc.yml",
        "eslint.config.js",
        "eslint.config.mjs",
        "biome.json",
    ];

    for file in configs {
        if repo_path.join(file).exists() {
            return Some(file.to_string());
        }
    }
    None
}

fn detect_port_from_cmd(cmd: &str) -> Option<u16> {
    // Look for --port or -p followed by a number
    let patterns = ["--port ", "-p "];
    for pat in patterns {
        if let Some(idx) = cmd.find(pat) {
            let after = &cmd[idx + pat.len()..];
            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(port) = num_str.parse::<u16>() {
                return Some(port);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn rust_single_project() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("Cargo.toml"),
            r#"[package]
name = "my-app"
version = "0.1.0"
"#,
        )
        .unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();
        assert_eq!(
            result.build.as_ref().unwrap().command,
            "cargo build"
        );
        assert_eq!(
            result.test.as_ref().unwrap().command,
            "cargo test"
        );
        assert_eq!(
            result.lint.as_ref().unwrap().command,
            "cargo clippy"
        );
        assert_eq!(
            result.test.as_ref().unwrap().framework,
            Some("built-in".to_string())
        );
    }

    #[test]
    fn rust_workspace_project() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("Cargo.toml"),
            r#"[workspace]
members = ["crates/core", "crates/cli"]
"#,
        )
        .unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();
        assert_eq!(
            result.build.as_ref().unwrap().command,
            "cargo build --workspace"
        );
        assert_eq!(
            result.test.as_ref().unwrap().command,
            "cargo test --workspace"
        );
        assert_eq!(
            result.build.as_ref().unwrap().output_dir,
            Some("target/debug".to_string())
        );
    }

    #[test]
    fn node_project_with_scripts() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("package.json"),
            r#"{
  "name": "my-app",
  "scripts": {
    "build": "next build",
    "test": "jest --coverage",
    "lint": "eslint src/",
    "dev": "next dev --port 3000"
  }
}"#,
        )
        .unwrap();
        fs::write(tmp.path().join("jest.config.ts"), "{}").unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();

        let build = result.build.unwrap();
        assert_eq!(build.command, "next build");
        assert_eq!(build.framework, Some("Next.js".to_string()));
        assert_eq!(build.output_dir, Some(".next".to_string()));

        let test = result.test.unwrap();
        assert_eq!(test.command, "jest --coverage");
        assert_eq!(test.framework, Some("Jest".to_string()));
        assert_eq!(test.config, Some("jest.config.ts".to_string()));

        let lint = result.lint.unwrap();
        assert_eq!(lint.command, "eslint src/");
        assert_eq!(lint.framework, Some("ESLint".to_string()));

        let run = result.run.unwrap();
        assert_eq!(run.command, "next dev --port 3000");
        assert_eq!(run.port, Some(3000));
    }

    #[test]
    fn node_project_start_fallback() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("package.json"),
            r#"{
  "name": "my-app",
  "scripts": {
    "start": "node dist/index.js"
  }
}"#,
        )
        .unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();
        assert!(result.build.is_none());
        let run = result.run.unwrap();
        assert_eq!(run.command, "node dist/index.js");
    }

    #[test]
    fn node_test_framework_from_config_file() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("package.json"),
            r#"{"name": "app"}"#,
        )
        .unwrap();
        fs::write(tmp.path().join("vitest.config.ts"), "{}").unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();
        let test = result.test.unwrap();
        assert_eq!(test.framework, Some("Vitest".to_string()));
        assert_eq!(test.config, Some("vitest.config.ts".to_string()));
    }

    #[test]
    fn python_pytest_project() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("pyproject.toml"),
            r#"[tool.poetry.dependencies]
python = "^3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
"#,
        )
        .unwrap();
        fs::write(
            tmp.path().join("requirements.txt"),
            "pytest==7.4.0\nruff==0.1.0\n",
        )
        .unwrap();
        fs::write(tmp.path().join("ruff.toml"), "").unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();
        assert!(result.build.is_none());

        let test = result.test.unwrap();
        assert_eq!(test.command, "pytest");
        assert_eq!(test.framework, Some("pytest".to_string()));
        assert_eq!(test.config, Some("pyproject.toml".to_string()));

        let lint = result.lint.unwrap();
        assert_eq!(lint.command, "ruff check .");
        assert_eq!(lint.framework, Some("ruff".to_string()));
        assert_eq!(lint.config, Some("ruff.toml".to_string()));
    }

    #[test]
    fn python_tox_project() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("setup.py"), "# setup").unwrap();
        fs::write(tmp.path().join("tox.ini"), "[tox]\nenvlist = py311\n").unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();
        let test = result.test.unwrap();
        assert_eq!(test.command, "tox");
        assert_eq!(test.framework, Some("tox".to_string()));
        assert_eq!(test.config, Some("tox.ini".to_string()));
    }

    #[test]
    fn python_flake8_lint() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("requirements.txt"), "flake8==6.0.0\n").unwrap();
        fs::write(tmp.path().join(".flake8"), "[flake8]\nmax-line-length = 120\n").unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();
        let lint = result.lint.unwrap();
        assert_eq!(lint.command, "flake8 .");
        assert_eq!(lint.framework, Some("flake8".to_string()));
        assert_eq!(lint.config, Some(".flake8".to_string()));
    }

    #[test]
    fn go_project() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("go.mod"), "module example.com/app\n\ngo 1.21\n").unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();
        assert_eq!(result.build.as_ref().unwrap().command, "go build ./...");
        assert_eq!(result.test.as_ref().unwrap().command, "go test ./...");
        assert_eq!(result.lint.as_ref().unwrap().command, "golangci-lint run");
        assert_eq!(result.run.as_ref().unwrap().command, "go run .");
    }

    #[test]
    fn empty_directory_returns_empty_commands() {
        let tmp = TempDir::new().unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();
        assert!(result.build.is_none());
        assert!(result.test.is_none());
        assert!(result.lint.is_none());
        assert!(result.run.is_none());
    }

    #[test]
    fn detect_test_framework_jest_config() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(detect_test_framework_from_config(tmp.path()), None);

        fs::write(tmp.path().join("jest.config.ts"), "export default {}").unwrap();
        assert_eq!(
            detect_test_framework_from_config(tmp.path()),
            Some("Jest".to_string())
        );
    }

    #[test]
    fn detect_test_framework_vitest_config() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("vitest.config.ts"), "export default {}").unwrap();
        assert_eq!(
            detect_test_framework_from_config(tmp.path()),
            Some("Vitest".to_string())
        );
    }

    #[test]
    fn detect_test_framework_playwright_config() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("playwright.config.ts"), "export default {}").unwrap();
        assert_eq!(
            detect_test_framework_from_config(tmp.path()),
            Some("Playwright".to_string())
        );
    }

    #[test]
    fn detect_port_parsing() {
        assert_eq!(detect_port_from_cmd("next dev --port 3000"), Some(3000));
        assert_eq!(detect_port_from_cmd("vite dev -p 5173"), Some(5173));
        assert_eq!(detect_port_from_cmd("node server.js"), None);
    }

    #[test]
    fn node_project_with_vite_and_vitest() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("package.json"),
            r#"{
  "name": "vite-app",
  "scripts": {
    "build": "vite build",
    "test": "vitest run",
    "lint": "biome check .",
    "dev": "vite dev -p 5173"
  }
}"#,
        )
        .unwrap();
        fs::write(tmp.path().join("biome.json"), "{}").unwrap();

        let result = extract_build_commands(tmp.path()).unwrap();

        let build = result.build.unwrap();
        assert_eq!(build.framework, Some("Vite".to_string()));
        assert_eq!(build.output_dir, Some("dist".to_string()));

        let test = result.test.unwrap();
        assert_eq!(test.framework, Some("Vitest".to_string()));

        let lint = result.lint.unwrap();
        assert_eq!(lint.framework, Some("Biome".to_string()));
        assert_eq!(lint.config, Some("biome.json".to_string()));

        let run = result.run.unwrap();
        assert_eq!(run.port, Some(5173));
    }
}
