//! LSP-based symbol resolution for Engram.
//!
//! This crate provides a `SymbolResolver` implementation that uses Language Server Protocol
//! clients to perform exact symbol resolution. It starts language servers on-demand for
//! TypeScript, Rust, Python, and Go, and shuts them down after indexing completes.

pub mod client;
pub mod install;
pub mod lang;
pub mod manager;
pub mod resolver;

pub use install::{command_exists, try_install};
pub use lang::{LangServerConfig, config_for_extension, config_for_path};
pub use manager::ServerManager;
pub use resolver::LspSymbolResolver;
