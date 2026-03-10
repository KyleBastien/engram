use engram_core::ChunkKind;

use crate::RawChunk;

/// Chunker that splits Markdown files at heading boundaries (#, ##, ###).
pub struct MarkdownChunker;

impl MarkdownChunker {
    pub fn new() -> Self {
        Self
    }

    /// Split a Markdown source into chunks at heading boundaries.
    ///
    /// - Headings (#, ##, ###) start new chunks with kind `DocSection`
    /// - Content before the first heading is captured as a "preamble" chunk
    /// - Each chunk includes the heading line and all content until the next heading
    pub fn chunk_file(&self, source: &str) -> Vec<RawChunk> {
        if source.trim().is_empty() {
            return vec![];
        }

        let lines: Vec<&str> = source.lines().collect();
        let mut chunks = Vec::new();
        let mut current_name: Option<String> = None;
        let mut current_start: usize = 0; // 0-based line index
        let mut current_lines: Vec<&str> = Vec::new();

        for (idx, line) in lines.iter().enumerate() {
            if let Some(heading) = parse_heading(line) {
                // Flush the previous section
                if !current_lines.is_empty() {
                    let name = current_name.unwrap_or_else(|| "preamble".to_string());
                    chunks.push(RawChunk {
                        kind: ChunkKind::DocSection,
                        name,
                        signature: None,
                        start_line: current_start as u32 + 1,
                        end_line: (current_start + current_lines.len()) as u32,
                        content: current_lines.join("\n"),
                    });
                }

                // Start a new section
                current_name = Some(heading);
                current_start = idx;
                current_lines = vec![line];
            } else {
                current_lines.push(line);
            }
        }

        // Flush the last section
        if !current_lines.is_empty() {
            let name = current_name.unwrap_or_else(|| "preamble".to_string());
            chunks.push(RawChunk {
                kind: ChunkKind::DocSection,
                name,
                signature: None,
                start_line: current_start as u32 + 1,
                end_line: (current_start + current_lines.len()) as u32,
                content: current_lines.join("\n"),
            });
        }

        chunks
    }
}

impl Default for MarkdownChunker {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a line as a Markdown heading (#, ##, or ###).
/// Returns the heading text (without the # prefix) if it's a heading, None otherwise.
fn parse_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    // Match #, ##, or ### followed by a space (check ### first to avoid matching ## or #)
    trimmed
        .strip_prefix("### ")
        .or_else(|| trimmed.strip_prefix("## "))
        .or_else(|| trimmed.strip_prefix("# "))
        .map(|text| text.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(source: &str) -> Vec<RawChunk> {
        MarkdownChunker::new().chunk_file(source)
    }

    #[test]
    fn empty_source_returns_empty() {
        let chunks = chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn whitespace_only_returns_empty() {
        let chunks = chunk("   \n  \n  ");
        assert!(chunks.is_empty());
    }

    #[test]
    fn preamble_before_first_heading() {
        let source = "Some intro text\nMore intro\n\n# First Section\nContent here";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].name, "preamble");
        assert_eq!(chunks[0].kind, ChunkKind::DocSection);
        assert_eq!(chunks[0].content, "Some intro text\nMore intro\n");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
    }

    #[test]
    fn splits_at_heading_boundaries() {
        let source = "# Heading One\nContent one\n## Heading Two\nContent two\n### Heading Three\nContent three";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].name, "Heading One");
        assert_eq!(chunks[1].name, "Heading Two");
        assert_eq!(chunks[2].name, "Heading Three");
    }

    #[test]
    fn chunk_kind_is_doc_section() {
        let source = "# Title\nBody text";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::DocSection);
    }

    #[test]
    fn heading_line_included_in_content() {
        let source = "# My Heading\nParagraph text\nMore text";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "# My Heading\nParagraph text\nMore text");
    }

    #[test]
    fn chunk_name_is_heading_text() {
        let source = "## API Reference\nDetails here";
        let chunks = chunk(source);
        assert_eq!(chunks[0].name, "API Reference");
    }

    #[test]
    fn line_ranges_are_correct() {
        let source = "# First\nLine 2\nLine 3\n## Second\nLine 5\nLine 6";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
        assert_eq!(chunks[1].start_line, 4);
        assert_eq!(chunks[1].end_line, 6);
    }

    #[test]
    fn no_headings_returns_single_preamble() {
        let source = "Just some text\nwithout any headings\nat all";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].name, "preamble");
        assert_eq!(chunks[0].content, source);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
    }

    #[test]
    fn signature_is_none() {
        let source = "# Heading\nContent";
        let chunks = chunk(source);
        assert_eq!(chunks[0].signature, None);
    }

    #[test]
    fn preamble_only_when_content_before_heading() {
        let source = "# First Heading\nContent";
        let chunks = chunk(source);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].name, "First Heading");
    }

    #[test]
    fn headings_deeper_than_h3_not_split() {
        let source = "# Title\nContent\n#### Deep Heading\nMore content";
        let chunks = chunk(source);
        // #### is not a split point, so only one chunk
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].name, "Title");
        assert!(chunks[0].content.contains("#### Deep Heading"));
    }

    #[test]
    fn heading_text_is_trimmed() {
        let source = "#   Spaced Heading  \nContent";
        let chunks = chunk(source);
        assert_eq!(chunks[0].name, "Spaced Heading");
    }
}
