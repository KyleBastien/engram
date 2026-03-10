use engram_core::ChunkKind;

use crate::RawChunk;

/// Fallback chunker that splits text into overlapping sliding windows.
/// Used for files in unsupported languages where tree-sitter parsing is not available.
pub struct SlidingWindowChunker;

impl SlidingWindowChunker {
    pub fn new() -> Self {
        Self
    }

    /// Split source text into overlapping windows based on whitespace-approximated token counts.
    ///
    /// - `max_tokens`: maximum number of whitespace-separated tokens per chunk
    /// - `overlap_tokens`: number of tokens to overlap between consecutive chunks
    pub fn chunk_file(&self, source: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<RawChunk> {
        if source.trim().is_empty() {
            return vec![];
        }

        let lines: Vec<&str> = source.lines().collect();

        // Count tokens per line (whitespace-split)
        let line_tokens: Vec<usize> = lines.iter().map(|l| token_count(l)).collect();
        let total_tokens: usize = line_tokens.iter().sum();

        // If the entire file fits in one chunk, return it as a single chunk
        if total_tokens <= max_tokens {
            return vec![RawChunk {
                kind: ChunkKind::Other,
                name: "chunk_0".to_string(),
                signature: None,
                start_line: 1,
                end_line: lines.len() as u32,
                content: source.to_string(),
            }];
        }

        let mut chunks = Vec::new();
        let mut chunk_idx = 0usize;
        let mut start_line_idx = 0usize; // 0-based index into lines

        while start_line_idx < lines.len() {
            // Accumulate lines until we hit max_tokens
            let mut tokens_so_far = 0usize;
            let mut end_line_idx = start_line_idx;

            while end_line_idx < lines.len() {
                let lt = line_tokens[end_line_idx];
                if tokens_so_far + lt > max_tokens && end_line_idx > start_line_idx {
                    break;
                }
                tokens_so_far += lt;
                end_line_idx += 1;
            }

            let content = lines[start_line_idx..end_line_idx].join("\n");
            chunks.push(RawChunk {
                kind: ChunkKind::Other,
                name: format!("chunk_{chunk_idx}"),
                signature: None,
                start_line: start_line_idx as u32 + 1,
                end_line: end_line_idx as u32,
                content,
            });

            chunk_idx += 1;

            if end_line_idx >= lines.len() {
                break;
            }

            // Step back by overlap_tokens worth of lines from end_line_idx
            let mut overlap_counted = 0usize;
            let mut next_start = end_line_idx;
            while next_start > start_line_idx && overlap_counted < overlap_tokens {
                next_start -= 1;
                overlap_counted += line_tokens[next_start];
            }

            // Ensure we always advance at least one line to avoid infinite loops
            if next_start <= start_line_idx {
                next_start = end_line_idx;
            }

            start_line_idx = next_start;
        }

        chunks
    }
}

impl Default for SlidingWindowChunker {
    fn default() -> Self {
        Self::new()
    }
}

/// Approximate token count by splitting on whitespace.
fn token_count(line: &str) -> usize {
    let count = line.split_whitespace().count();
    // Empty/whitespace-only lines still count as at least 1 token
    // to ensure progress through the file
    if count == 0 { 1 } else { count }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(source: &str, max_tokens: usize, overlap: usize) -> Vec<RawChunk> {
        SlidingWindowChunker::new().chunk_file(source, max_tokens, overlap)
    }

    #[test]
    fn empty_source_returns_empty() {
        let chunks = chunk("", 100, 10);
        assert!(chunks.is_empty());
    }

    #[test]
    fn whitespace_only_returns_empty() {
        let chunks = chunk("   \n  \n  ", 100, 10);
        assert!(chunks.is_empty());
    }

    #[test]
    fn small_file_single_chunk() {
        let source = "hello world\nfoo bar baz";
        let chunks = chunk(source, 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Other);
        assert_eq!(chunks[0].name, "chunk_0");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 2);
        assert_eq!(chunks[0].content, source);
    }

    #[test]
    fn splits_into_multiple_chunks() {
        // 5 tokens per line, 4 lines = 20 tokens total
        let source = "a b c d e\nf g h i j\nk l m n o\np q r s t";
        // max 10 tokens, no overlap -> should produce 2 chunks
        let chunks = chunk(source, 10, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].name, "chunk_0");
        assert_eq!(chunks[1].name, "chunk_1");
        // First chunk: lines 1-2, second chunk: lines 3-4
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 2);
        assert_eq!(chunks[1].start_line, 3);
        assert_eq!(chunks[1].end_line, 4);
    }

    #[test]
    fn chunks_have_correct_kind() {
        let source = "a b c d e\nf g h i j\nk l m n o";
        let chunks = chunk(source, 10, 0);
        for c in &chunks {
            assert_eq!(c.kind, ChunkKind::Other);
        }
    }

    #[test]
    fn overlap_produces_overlapping_content() {
        // 5 tokens per line, 4 lines = 20 tokens
        let source = "a b c d e\nf g h i j\nk l m n o\np q r s t";
        // max 10 tokens, overlap 5 tokens -> overlap should include 1 line
        let chunks = chunk(source, 10, 5);
        assert!(chunks.len() >= 2);
        // Second chunk should start before line 3 due to overlap
        assert!(chunks[1].start_line <= 2);
    }

    #[test]
    fn generated_names_sequential() {
        let source = "a b c d e\nf g h i j\nk l m n o\np q r s t\nu v w x y\nz 1 2 3 4";
        let chunks = chunk(source, 10, 0);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.name, format!("chunk_{i}"));
        }
    }

    #[test]
    fn signature_is_none() {
        let source = "hello world\nfoo bar baz";
        let chunks = chunk(source, 100, 10);
        assert_eq!(chunks[0].signature, None);
    }

    #[test]
    fn line_ranges_are_one_based() {
        let source = "first line\nsecond line\nthird line";
        let chunks = chunk(source, 100, 0);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
    }

    #[test]
    fn single_line_file() {
        let source = "just one line of tokens here";
        let chunks = chunk(source, 100, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 1);
        assert_eq!(chunks[0].content, source);
    }
}
