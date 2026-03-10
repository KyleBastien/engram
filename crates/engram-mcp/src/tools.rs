use serde_json::json;

/// Returns tool definitions for Phase 1 MCP tools.
pub fn phase1_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        json!({
            "name": "engram_search",
            "description": "Hybrid semantic + keyword search over indexed code and documentation. Returns ranked results combining vector similarity and BM25 keyword matching.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["code", "docs", "all"],
                        "description": "Search scope: code, docs, or all (default: all)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)"
                    },
                    "compact": {
                        "type": "boolean",
                        "description": "If true, omit signatures and truncate long names"
                    }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "engram_lookup",
            "description": "Look up chunks directly by chunk ID, file path, or symbol name. Returns matching chunks without performing search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Chunk ID (contains #), file path (contains / or .), or symbol name"
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "If true, include raw source text (default: false)"
                    }
                },
                "required": ["identifier"]
            }
        }),
        json!({
            "name": "engram_status",
            "description": "Check index health, staleness, and configuration. Returns store metadata and index statistics.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase1_has_three_tools() {
        let tools = phase1_tool_definitions();
        assert_eq!(tools.len(), 3);
    }

    #[test]
    fn test_tool_names() {
        let tools = phase1_tool_definitions();
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"engram_search"));
        assert!(names.contains(&"engram_lookup"));
        assert!(names.contains(&"engram_status"));
    }

    #[test]
    fn test_engram_search_has_required_query() {
        let tools = phase1_tool_definitions();
        let search = &tools[0];
        let required = search["inputSchema"]["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str().unwrap(), "query");
    }

    #[test]
    fn test_engram_lookup_has_required_identifier() {
        let tools = phase1_tool_definitions();
        let lookup = &tools[1];
        let required = lookup["inputSchema"]["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str().unwrap(), "identifier");
    }

    #[test]
    fn test_engram_status_no_required_params() {
        let tools = phase1_tool_definitions();
        let status = &tools[2];
        assert!(status["inputSchema"]["required"].is_null());
    }

    #[test]
    fn test_all_tools_have_description() {
        let tools = phase1_tool_definitions();
        for tool in &tools {
            assert!(tool["description"].as_str().unwrap().len() > 10);
        }
    }

    #[test]
    fn test_all_tools_have_input_schema() {
        let tools = phase1_tool_definitions();
        for tool in &tools {
            assert_eq!(tool["inputSchema"]["type"].as_str().unwrap(), "object");
        }
    }
}
