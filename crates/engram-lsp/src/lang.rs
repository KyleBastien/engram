use std::path::Path;

/// Configuration for a language server.
#[derive(Debug, Clone)]
pub struct LangServerConfig {
    /// The command to start the language server.
    pub command: &'static str,
    /// Arguments to pass to the language server.
    pub args: &'static [&'static str],
    /// The LSP language identifier.
    pub language_id: &'static str,
}

/// Known language server configurations.
pub const TYPESCRIPT_SERVER: LangServerConfig = LangServerConfig {
    command: "typescript-language-server",
    args: &["--stdio"],
    language_id: "typescript",
};

pub const RUST_SERVER: LangServerConfig = LangServerConfig {
    command: "rust-analyzer",
    args: &[],
    language_id: "rust",
};

pub const PYTHON_SERVER: LangServerConfig = LangServerConfig {
    command: "pylsp",
    args: &[],
    language_id: "python",
};

pub const GO_SERVER: LangServerConfig = LangServerConfig {
    command: "gopls",
    args: &["serve"],
    language_id: "go",
};

/// Returns the appropriate language server config for a file extension.
pub fn config_for_extension(ext: &str) -> Option<&'static LangServerConfig> {
    match ext {
        "ts" | "tsx" | "js" | "jsx" => Some(&TYPESCRIPT_SERVER),
        "rs" => Some(&RUST_SERVER),
        "py" => Some(&PYTHON_SERVER),
        "go" => Some(&GO_SERVER),
        _ => None,
    }
}

/// Returns the appropriate language server config for a file path.
pub fn config_for_path(path: &Path) -> Option<&'static LangServerConfig> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .and_then(config_for_extension)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn typescript_extensions() {
        assert_eq!(
            config_for_extension("ts").unwrap().language_id,
            "typescript"
        );
        assert_eq!(
            config_for_extension("tsx").unwrap().language_id,
            "typescript"
        );
        assert_eq!(
            config_for_extension("js").unwrap().language_id,
            "typescript"
        );
        assert_eq!(
            config_for_extension("jsx").unwrap().language_id,
            "typescript"
        );
    }

    #[test]
    fn rust_extension() {
        let cfg = config_for_extension("rs").unwrap();
        assert_eq!(cfg.language_id, "rust");
        assert_eq!(cfg.command, "rust-analyzer");
    }

    #[test]
    fn python_extension() {
        let cfg = config_for_extension("py").unwrap();
        assert_eq!(cfg.language_id, "python");
        assert_eq!(cfg.command, "pylsp");
    }

    #[test]
    fn go_extension() {
        let cfg = config_for_extension("go").unwrap();
        assert_eq!(cfg.language_id, "go");
        assert_eq!(cfg.command, "gopls");
    }

    #[test]
    fn unknown_extension() {
        assert!(config_for_extension("txt").is_none());
        assert!(config_for_extension("html").is_none());
    }

    #[test]
    fn config_for_path_works() {
        let path = PathBuf::from("src/main.rs");
        assert_eq!(config_for_path(&path).unwrap().language_id, "rust");

        let path = PathBuf::from("index.ts");
        assert_eq!(config_for_path(&path).unwrap().language_id, "typescript");

        let path = PathBuf::from("README.md");
        assert!(config_for_path(&path).is_none());
    }
}
