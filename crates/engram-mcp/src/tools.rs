use serde_json::json;

/// Returns tool definitions for Phase 1 MCP tools.
pub fn phase1_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        json!({
            "name": "engram_search",
            "description": "Hybrid semantic + keyword search over indexed code, documentation, and knowledge. Returns ranked code_results and knowledge_results combining vector similarity and BM25 keyword matching.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["code", "docs", "all", "knowledge"],
                        "description": "Search scope: code (code only), docs (docs only), knowledge (knowledge only), or all (code+docs+knowledge sidecar) (default: all)"
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

/// Returns tool definitions for knowledge recording MCP tools.
pub fn knowledge_tool_definitions() -> Vec<serde_json::Value> {
    vec![json!({
        "name": "engram_record_decision",
        "description": "Record an architectural or design decision to the knowledge base. Auto-generates id, contributed_by, and created_at. Writes YAML, embeds it, and commits to the store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for the decision"
                },
                "context": {
                    "type": "string",
                    "description": "Background context explaining why the decision is needed"
                },
                "decision": {
                    "type": "string",
                    "description": "The decision that was made"
                },
                "consequences": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of consequences of this decision"
                },
                "related_files": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of related file paths"
                },
                "status": {
                    "type": "string",
                    "description": "Decision status (default: 'accepted')"
                }
            },
            "required": ["title", "context", "decision"]
        }
    }),
    json!({
        "name": "engram_record_lesson",
        "description": "Record a lesson learned to the knowledge base. Auto-generates id, contributed_by, and created_at. Writes YAML, embeds it, and commits to the store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for the lesson"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of what was learned"
                },
                "trigger": {
                    "type": "string",
                    "description": "What situation or event triggered this lesson"
                },
                "resolution": {
                    "type": "string",
                    "description": "How the issue was resolved or what to do differently"
                },
                "related_files": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of related file paths"
                }
            },
            "required": ["title", "description", "trigger"]
        }
    }),
    json!({
        "name": "engram_record_pattern",
        "description": "Record a code or architecture pattern to the knowledge base. Auto-generates id, contributed_by, and created_at. Writes YAML, embeds it, and commits to the store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the pattern"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the pattern"
                },
                "examples": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of example code or usages"
                },
                "anti_patterns": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of anti-patterns to avoid"
                }
            },
            "required": ["name", "description"]
        }
    }),
    json!({
        "name": "engram_record_glossary",
        "description": "Add or update a glossary term in the knowledge base. If the term already exists in terms.yaml, updates it in place; if new, appends it. Commits to the store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The term to define"
                },
                "definition": {
                    "type": "string",
                    "description": "The definition of the term"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about where/how the term is used"
                }
            },
            "required": ["term", "definition"]
        }
    }),
    json!({
        "name": "engram_snapshot",
        "description": "Save a full conversation snapshot for session resumability. Writes to active tier, embeds summary + key_context for searchability, and commits to the store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Unique identifier for the session"
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of the conversation or session"
                },
                "key_context": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Key context items for searchability"
                },
                "full_transcript": {
                    "type": "string",
                    "description": "Full conversation transcript"
                }
            },
            "required": ["session_id", "summary", "key_context", "full_transcript"]
        }
    })]
}

/// Returns tool definitions for onboarding MCP tools.
pub fn onboarding_tool_definitions() -> Vec<serde_json::Value> {
    vec![json!({
        "name": "engram_onboard",
        "description": "Trigger the onboarding pipeline to analyze a source repository and write knowledge YAML files (project overview, build commands, architecture map, key abstractions) to the store. Idempotent — re-running overwrites previous onboarding data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Source repo name from configured source_roots (uses first available if omitted)"
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "deep"],
                    "description": "How deep the analysis should go: quick (metadata + commands), standard (+ architecture + abstractions), deep (reserved for full symbol export). Default: standard"
                }
            }
        }
    })]
}

/// Returns tool definitions for context assessment MCP tools.
pub fn assessment_tool_definitions() -> Vec<serde_json::Value> {
    vec![json!({
        "name": "engram_assess_context",
        "description": "Self-evaluate whether enough context has been gathered for the current task. Returns a summary of searches performed, tokens consumed, knowledge items surfaced, and a structured evaluation prompt. Does NOT perform any search — it is a reflection tool.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the task being worked on, used to frame the evaluation question"
                }
            }
        }
    })]
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

    #[test]
    fn test_knowledge_tool_definitions_count() {
        let tools = knowledge_tool_definitions();
        assert_eq!(tools.len(), 5);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"engram_record_decision"));
        assert!(names.contains(&"engram_record_lesson"));
        assert!(names.contains(&"engram_record_pattern"));
        assert!(names.contains(&"engram_record_glossary"));
        assert!(names.contains(&"engram_snapshot"));
    }

    #[test]
    fn test_record_decision_has_required_params() {
        let tools = knowledge_tool_definitions();
        let decision_tool = &tools[0];
        let required = decision_tool["inputSchema"]["required"].as_array().unwrap();
        let required_names: Vec<&str> = required.iter().map(|r| r.as_str().unwrap()).collect();
        assert!(required_names.contains(&"title"));
        assert!(required_names.contains(&"context"));
        assert!(required_names.contains(&"decision"));
        assert_eq!(required_names.len(), 3);
    }

    #[test]
    fn test_record_decision_has_optional_params() {
        let tools = knowledge_tool_definitions();
        let decision_tool = &tools[0];
        let props = decision_tool["inputSchema"]["properties"].as_object().unwrap();
        assert!(props.contains_key("consequences"));
        assert!(props.contains_key("related_files"));
        assert!(props.contains_key("status"));
    }

    #[test]
    fn test_record_lesson_has_required_params() {
        let tools = knowledge_tool_definitions();
        let lesson_tool = &tools[1];
        let required = lesson_tool["inputSchema"]["required"].as_array().unwrap();
        let required_names: Vec<&str> = required.iter().map(|r| r.as_str().unwrap()).collect();
        assert!(required_names.contains(&"title"));
        assert!(required_names.contains(&"description"));
        assert!(required_names.contains(&"trigger"));
        assert_eq!(required_names.len(), 3);
    }

    #[test]
    fn test_record_lesson_has_optional_params() {
        let tools = knowledge_tool_definitions();
        let lesson_tool = &tools[1];
        let props = lesson_tool["inputSchema"]["properties"].as_object().unwrap();
        assert!(props.contains_key("resolution"));
        assert!(props.contains_key("related_files"));
    }

    #[test]
    fn test_record_pattern_has_required_params() {
        let tools = knowledge_tool_definitions();
        let pattern_tool = &tools[2];
        let required = pattern_tool["inputSchema"]["required"].as_array().unwrap();
        let required_names: Vec<&str> = required.iter().map(|r| r.as_str().unwrap()).collect();
        assert!(required_names.contains(&"name"));
        assert!(required_names.contains(&"description"));
        assert_eq!(required_names.len(), 2);
    }

    #[test]
    fn test_record_pattern_has_optional_params() {
        let tools = knowledge_tool_definitions();
        let pattern_tool = &tools[2];
        let props = pattern_tool["inputSchema"]["properties"].as_object().unwrap();
        assert!(props.contains_key("examples"));
        assert!(props.contains_key("anti_patterns"));
    }

    #[test]
    fn test_record_glossary_has_required_params() {
        let tools = knowledge_tool_definitions();
        let glossary_tool = &tools[3];
        let required = glossary_tool["inputSchema"]["required"].as_array().unwrap();
        let required_names: Vec<&str> = required.iter().map(|r| r.as_str().unwrap()).collect();
        assert!(required_names.contains(&"term"));
        assert!(required_names.contains(&"definition"));
        assert_eq!(required_names.len(), 2);
    }

    #[test]
    fn test_record_glossary_has_optional_params() {
        let tools = knowledge_tool_definitions();
        let glossary_tool = &tools[3];
        let props = glossary_tool["inputSchema"]["properties"].as_object().unwrap();
        assert!(props.contains_key("context"));
    }

    #[test]
    fn test_snapshot_has_required_params() {
        let tools = knowledge_tool_definitions();
        let snapshot_tool = &tools[4];
        assert_eq!(snapshot_tool["name"].as_str().unwrap(), "engram_snapshot");
        let required = snapshot_tool["inputSchema"]["required"].as_array().unwrap();
        let required_names: Vec<&str> = required.iter().map(|r| r.as_str().unwrap()).collect();
        assert!(required_names.contains(&"session_id"));
        assert!(required_names.contains(&"summary"));
        assert!(required_names.contains(&"key_context"));
        assert!(required_names.contains(&"full_transcript"));
        assert_eq!(required_names.len(), 4);
    }

    #[test]
    fn test_snapshot_has_all_properties() {
        let tools = knowledge_tool_definitions();
        let snapshot_tool = &tools[4];
        let props = snapshot_tool["inputSchema"]["properties"].as_object().unwrap();
        assert!(props.contains_key("session_id"));
        assert!(props.contains_key("summary"));
        assert!(props.contains_key("key_context"));
        assert!(props.contains_key("full_transcript"));
    }

    #[test]
    fn test_onboarding_tool_definitions_count() {
        let tools = onboarding_tool_definitions();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"].as_str().unwrap(), "engram_onboard");
    }

    #[test]
    fn test_onboard_has_no_required_params() {
        let tools = onboarding_tool_definitions();
        let onboard = &tools[0];
        assert!(onboard["inputSchema"]["required"].is_null());
    }

    #[test]
    fn test_onboard_has_optional_params() {
        let tools = onboarding_tool_definitions();
        let onboard = &tools[0];
        let props = onboard["inputSchema"]["properties"].as_object().unwrap();
        assert!(props.contains_key("repo"));
        assert!(props.contains_key("depth"));
    }

    #[test]
    fn test_assessment_tool_definitions_count() {
        let tools = assessment_tool_definitions();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"].as_str().unwrap(), "engram_assess_context");
    }

    #[test]
    fn test_assess_context_has_no_required_params() {
        let tools = assessment_tool_definitions();
        let tool = &tools[0];
        assert!(tool["inputSchema"]["required"].is_null());
    }

    #[test]
    fn test_assess_context_has_optional_task_description() {
        let tools = assessment_tool_definitions();
        let tool = &tools[0];
        let props = tool["inputSchema"]["properties"].as_object().unwrap();
        assert!(props.contains_key("task_description"));
        assert_eq!(props["task_description"]["type"].as_str().unwrap(), "string");
    }
}
