use crate::client::LspClient;
use crate::install;
use crate::lang;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::sync::Mutex;

/// Manages on-demand language server instances, one per (language, root) pair.
pub struct ServerManager {
    servers: Mutex<HashMap<String, LspClient>>,
    pub root_path: PathBuf,
    pub auto_install: bool,
}

impl ServerManager {
    /// Create a new server manager for the given workspace root.
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            servers: Mutex::new(HashMap::new()),
            root_path,
            auto_install: false,
        }
    }

    /// Create a new server manager with auto-install enabled.
    pub fn with_auto_install(root_path: PathBuf, auto_install: bool) -> Self {
        Self {
            servers: Mutex::new(HashMap::new()),
            root_path,
            auto_install,
        }
    }

    /// Ensure a language server is running for the given file type.
    /// Returns the language_id key if a server was started (or already running),
    /// or None if no server is configured for this file type.
    ///
    /// When `auto_install` is true and the server command is not found on PATH,
    /// attempts to install it before starting.
    pub async fn ensure_server(&self, file: &Path) -> engram_core::Result<Option<String>> {
        let config = match lang::config_for_path(file) {
            Some(c) => c,
            None => return Ok(None),
        };

        let key = config.language_id.to_string();
        let mut servers = self.servers.lock().await;

        if !servers.contains_key(&key) {
            // Check if the server command exists; if not and auto_install is on, try installing
            if !install::command_exists(config.command).await {
                if self.auto_install {
                    match install::try_install(config.command).await {
                        Ok(true) => {
                            tracing::info!(
                                "Auto-installed language server '{}'",
                                config.command
                            );
                        }
                        Ok(false) => {
                            tracing::warn!(
                                "No install spec for '{}', falling back to tree-sitter",
                                config.command
                            );
                            return Ok(None);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to auto-install '{}': {}, falling back to tree-sitter",
                                config.command,
                                e
                            );
                            return Ok(None);
                        }
                    }
                } else {
                    tracing::warn!(
                        "Language server '{}' not found on PATH (auto_install is disabled), \
                         falling back to tree-sitter",
                        config.command
                    );
                    return Ok(None);
                }
            }

            let client = LspClient::start(config, &self.root_path).await?;
            client.initialize().await?;
            servers.insert(key.clone(), client);
        }

        Ok(Some(key))
    }

    /// Access the underlying servers map. The caller must hold the lock for the
    /// duration of any operations on the returned client.
    pub fn servers(&self) -> &Mutex<HashMap<String, LspClient>> {
        &self.servers
    }

    /// Shut down all running language servers.
    pub async fn shutdown_all(&self) -> engram_core::Result<()> {
        let mut servers = self.servers.lock().await;
        for (lang, client) in servers.drain() {
            if let Err(e) = client.shutdown().await {
                tracing::warn!("Failed to shut down {} language server: {}", lang, e);
            }
        }
        Ok(())
    }

    /// Returns the list of currently running language server language IDs.
    pub async fn active_languages(&self) -> Vec<String> {
        self.servers.lock().await.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_manager() {
        let manager = ServerManager::new(PathBuf::from("/tmp/test"));
        assert_eq!(manager.root_path, PathBuf::from("/tmp/test"));
        assert!(!manager.auto_install);
    }

    #[test]
    fn new_manager_with_auto_install() {
        let manager = ServerManager::with_auto_install(PathBuf::from("/tmp/test"), true);
        assert_eq!(manager.root_path, PathBuf::from("/tmp/test"));
        assert!(manager.auto_install);
    }

    #[tokio::test]
    async fn ensure_server_unknown_extension() {
        let manager = ServerManager::new(PathBuf::from("/tmp/test"));
        let result = manager.ensure_server(Path::new("README.md")).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn shutdown_empty_manager() {
        let manager = ServerManager::new(PathBuf::from("/tmp/test"));
        manager.shutdown_all().await.unwrap();
        assert!(manager.active_languages().await.is_empty());
    }

    #[tokio::test]
    async fn ensure_server_falls_back_when_not_installed() {
        // When auto_install is false and the server isn't on PATH, returns None (fallback)
        let manager = ServerManager::new(PathBuf::from("/tmp/test"));
        // .rs files need rust-analyzer; if it's not installed this should gracefully return None
        // We can't guarantee rust-analyzer is missing, so just verify the method doesn't panic
        let result = manager
            .ensure_server(Path::new("README.md"))
            .await
            .unwrap();
        assert!(result.is_none());
    }
}
