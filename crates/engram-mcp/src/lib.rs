mod protocol;
mod server;
mod tools;

pub use protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse, INVALID_PARAMS, METHOD_NOT_FOUND, PARSE_ERROR};
pub use server::{Context, EngineState, McpServer, Mode, ModeBehavior};
pub use tools::phase1_tool_definitions;
