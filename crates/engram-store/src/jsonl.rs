use std::fs;
use std::path::Path;

use engram_core::{ChunkMetadata, EngramError, Result};

/// Write chunk metadata as JSONL (one JSON object per line).
pub fn write_chunks_jsonl(path: &Path, chunks: &[ChunkMetadata]) -> Result<()> {
    let lines: Vec<String> = chunks
        .iter()
        .map(|c| serde_json::to_string(c).map_err(|e| EngramError::Serialize(e.to_string())))
        .collect::<Result<Vec<_>>>()?;
    let content = if lines.is_empty() {
        String::new()
    } else {
        let mut s = lines.join("\n");
        s.push('\n');
        s
    };
    fs::write(path, content)?;
    Ok(())
}

/// Read chunk metadata from a JSONL file (one JSON object per line).
///
/// Returns an empty vec for empty files. Lines with only whitespace are skipped.
pub fn read_chunks_jsonl(path: &Path) -> Result<Vec<ChunkMetadata>> {
    let content = fs::read_to_string(path)?;
    if content.trim().is_empty() {
        return Ok(Vec::new());
    }
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line.trim())
                .map_err(|e| EngramError::Serialize(e.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::{ChunkKind, ChunkMetadata};
    use tempfile::TempDir;

    fn sample_chunk(name: &str, kind: ChunkKind) -> ChunkMetadata {
        ChunkMetadata {
            chunk_id: format!("chunk-{name}"),
            kind,
            name: name.to_string(),
            signature: Some(format!("fn {name}()")),
            start_line: 1,
            end_line: 10,
            content_hash: "abc123".to_string(),
            tags: vec!["test".to_string()],
            indexed_at: "2026-03-09T00:00:00Z".to_string(),
            source_commit: "deadbeef".to_string(),
            embedding_offset: 0,
        }
    }

    #[test]
    fn round_trip_multiple_chunks() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("chunks.jsonl");

        let chunks = vec![
            sample_chunk("foo", ChunkKind::Function),
            sample_chunk("Bar", ChunkKind::Class),
            sample_chunk("baz", ChunkKind::Method),
        ];

        write_chunks_jsonl(&path, &chunks).unwrap();
        let read_back = read_chunks_jsonl(&path).unwrap();

        assert_eq!(chunks, read_back);
    }

    #[test]
    fn empty_file_returns_empty_vec() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("empty.jsonl");

        // Write empty chunks
        write_chunks_jsonl(&path, &[]).unwrap();
        let result = read_chunks_jsonl(&path).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn handles_trailing_newlines_and_whitespace() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("whitespace.jsonl");

        let chunk = sample_chunk("foo", ChunkKind::Function);
        let json = serde_json::to_string(&chunk).unwrap();

        // Write with extra trailing newlines and blank lines
        let content = format!("{json}\n\n  \n{json}\n\n");
        fs::write(&path, content).unwrap();

        let result = read_chunks_jsonl(&path).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], chunk);
        assert_eq!(result[1], chunk);
    }

    #[test]
    fn one_json_object_per_line() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("format.jsonl");

        let chunks = vec![
            sample_chunk("a", ChunkKind::Function),
            sample_chunk("b", ChunkKind::Class),
        ];

        write_chunks_jsonl(&path, &chunks).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        // Each line should parse independently
        for line in &lines {
            let _: ChunkMetadata = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn single_chunk_round_trip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("single.jsonl");

        let chunks = vec![sample_chunk("only", ChunkKind::Other)];

        write_chunks_jsonl(&path, &chunks).unwrap();
        let read_back = read_chunks_jsonl(&path).unwrap();

        assert_eq!(chunks, read_back);
    }
}
