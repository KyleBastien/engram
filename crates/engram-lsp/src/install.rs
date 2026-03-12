use engram_core::EngramError;
use std::process::Stdio;

/// Check if a command exists on PATH.
pub async fn command_exists(command: &str) -> bool {
    tokio::process::Command::new("which")
        .arg(command)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|s| s.success())
}

/// Install command and arguments for each language server.
pub struct InstallSpec {
    pub command: &'static str,
    pub args: &'static [&'static str],
    pub description: &'static str,
}

/// Returns the install spec for a given language server command, if known.
pub fn install_spec_for(server_command: &str) -> Option<InstallSpec> {
    match server_command {
        "typescript-language-server" => Some(InstallSpec {
            command: "npm",
            args: &["install", "-g", "typescript-language-server", "typescript"],
            description: "TypeScript language server",
        }),
        "rust-analyzer" => Some(InstallSpec {
            command: "rust-analyzer",
            args: &[],
            description: "Rust Analyzer",
        }),
        "pylsp" => Some(InstallSpec {
            command: "pip",
            args: &["install", "python-lsp-server"],
            description: "Python language server",
        }),
        "gopls" => Some(InstallSpec {
            command: "go",
            args: &["install", "golang.org/x/tools/gopls@latest"],
            description: "Go language server",
        }),
        _ => None,
    }
}

/// Attempt to install a language server. Returns Ok(true) if install succeeded,
/// Ok(false) if install was skipped (no spec known), or Err on failure.
pub async fn try_install(server_command: &str) -> engram_core::Result<bool> {
    let spec = match install_spec_for(server_command) {
        Some(s) => s,
        None => return Ok(false),
    };

    // rust-analyzer: just check if it exists, we don't install it
    if server_command == "rust-analyzer" {
        if command_exists("rust-analyzer").await {
            return Ok(true);
        }
        tracing::warn!(
            "rust-analyzer not found on PATH; install it via rustup or your package manager"
        );
        return Err(EngramError::Mcp(
            "rust-analyzer not found on PATH".to_string(),
        ));
    }

    // Check if the installer tool exists
    if !command_exists(spec.command).await {
        return Err(EngramError::Mcp(format!(
            "Cannot install {}: '{}' not found on PATH",
            spec.description, spec.command
        )));
    }

    tracing::info!(
        "Auto-installing {} via `{} {}`",
        spec.description,
        spec.command,
        spec.args.join(" ")
    );

    let status = tokio::process::Command::new(spec.command)
        .args(spec.args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .status()
        .await
        .map_err(|e| {
            EngramError::Mcp(format!("Failed to run {} install: {}", spec.description, e))
        })?;

    if status.success() {
        tracing::info!("Successfully installed {}", spec.description);
        Ok(true)
    } else {
        Err(EngramError::Mcp(format!(
            "Failed to install {} (exit code: {:?})",
            spec.description,
            status.code()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_spec_typescript() {
        let spec = install_spec_for("typescript-language-server").unwrap();
        assert_eq!(spec.command, "npm");
        assert!(spec.args.contains(&"typescript-language-server"));
        assert!(spec.args.contains(&"typescript"));
    }

    #[test]
    fn install_spec_rust_analyzer() {
        let spec = install_spec_for("rust-analyzer").unwrap();
        assert_eq!(spec.command, "rust-analyzer");
    }

    #[test]
    fn install_spec_python() {
        let spec = install_spec_for("pylsp").unwrap();
        assert_eq!(spec.command, "pip");
        assert!(spec.args.contains(&"python-lsp-server"));
    }

    #[test]
    fn install_spec_go() {
        let spec = install_spec_for("gopls").unwrap();
        assert_eq!(spec.command, "go");
        assert!(spec.args.contains(&"golang.org/x/tools/gopls@latest"));
    }

    #[test]
    fn install_spec_unknown() {
        assert!(install_spec_for("unknown-server").is_none());
    }

    #[tokio::test]
    async fn command_exists_finds_known_command() {
        // 'which' itself should always exist on unix-like systems
        assert!(command_exists("which").await);
    }

    #[tokio::test]
    async fn command_exists_returns_false_for_nonexistent() {
        assert!(!command_exists("this-command-does-not-exist-xyz123").await);
    }

    #[tokio::test]
    async fn try_install_unknown_returns_false() {
        let result = try_install("unknown-server").await.unwrap();
        assert!(!result);
    }
}
