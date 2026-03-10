use std::fs;
use std::io::Cursor;
use std::path::Path;

use engram_core::{EngramError, Result};

const MAGIC: &[u8; 4] = b"EGRM";
const VERSION: u16 = 1;
const PRECISION_F32: u16 = 0;

/// Parsed embedding file returned by [`read_embeddings_bin`].
#[derive(Debug, Clone)]
pub struct EmbeddingFile {
    /// Number of dimensions per vector.
    pub dimensions: usize,
    /// Number of vectors stored.
    pub count: usize,
    /// Precision indicator (0 = f32).
    pub precision: u16,
    /// Flattened vector data (`count * dimensions` floats).
    pub vectors: Vec<f32>,
}

/// Write embedding vectors in compact binary format.
///
/// Format: 16-byte header followed by contiguous little-endian f32 values.
pub fn write_embeddings_bin(
    path: &Path,
    vectors: &[Vec<f32>],
    dimensions: usize,
) -> Result<()> {
    let count = vectors.len() as u32;
    let dims = dimensions as u16;

    // Build header (16 bytes)
    let mut buf: Vec<u8> = Vec::with_capacity(16 + vectors.len() * dimensions * 4);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&dims.to_le_bytes());
    buf.extend_from_slice(&count.to_le_bytes());
    buf.extend_from_slice(&PRECISION_F32.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // reserved

    // Write vector data
    for vec in vectors {
        for &val in vec {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }

    fs::write(path, buf)?;
    Ok(())
}

/// Read embedding vectors from the binary format.
///
/// Validates magic bytes and version. Returns an [`EmbeddingFile`] with the
/// parsed header fields and raw vector data.
pub fn read_embeddings_bin(path: &Path) -> Result<EmbeddingFile> {
    let data = fs::read(path)?;

    if data.len() < 16 {
        return Err(EngramError::Store(
            "embedding file too short for header".into(),
        ));
    }

    // Validate magic
    if &data[0..4] != MAGIC {
        return Err(EngramError::Store(format!(
            "invalid magic bytes: expected EGRM, got {:?}",
            &data[0..4]
        )));
    }

    let mut cursor = Cursor::new(&data);
    use std::io::Read;
    let mut skip = [0u8; 4];
    cursor.read_exact(&mut skip).unwrap(); // magic

    let version = read_u16(&data[4..6]);
    if version != VERSION {
        return Err(EngramError::Store(format!(
            "unsupported embedding file version: {version}"
        )));
    }

    let dimensions = read_u16(&data[6..8]) as usize;
    let count = read_u32(&data[8..12]) as usize;
    let precision = read_u16(&data[12..14]);
    // bytes 14..16 are reserved

    let expected_data_len = count * dimensions * 4;
    let actual_data_len = data.len() - 16;
    if actual_data_len < expected_data_len {
        return Err(EngramError::Store(format!(
            "embedding file truncated: expected {expected_data_len} bytes of vector data, got {actual_data_len}"
        )));
    }

    let mut vectors = Vec::with_capacity(count * dimensions);
    let vector_data = &data[16..];
    for i in 0..(count * dimensions) {
        let offset = i * 4;
        vectors.push(read_f32(&vector_data[offset..offset + 4]));
    }

    Ok(EmbeddingFile {
        dimensions,
        count,
        precision,
        vectors,
    })
}

fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn read_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_zero_vectors() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("empty.bin");

        write_embeddings_bin(&path, &[], 3).unwrap();
        let result = read_embeddings_bin(&path).unwrap();

        assert_eq!(result.count, 0);
        assert_eq!(result.dimensions, 3);
        assert_eq!(result.precision, PRECISION_F32);
        assert!(result.vectors.is_empty());
    }

    #[test]
    fn round_trip_single_vector() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("single.bin");

        let vectors = vec![vec![1.0f32, 2.0, 3.0]];
        write_embeddings_bin(&path, &vectors, 3).unwrap();
        let result = read_embeddings_bin(&path).unwrap();

        assert_eq!(result.count, 1);
        assert_eq!(result.dimensions, 3);
        assert_eq!(result.vectors, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn round_trip_100_vectors() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hundred.bin");

        let dims = 768;
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..dims).map(|d| (i * dims + d) as f32 * 0.001).collect())
            .collect();

        write_embeddings_bin(&path, &vectors, dims).unwrap();
        let result = read_embeddings_bin(&path).unwrap();

        assert_eq!(result.count, 100);
        assert_eq!(result.dimensions, dims);
        // Verify all values round-trip exactly
        for (i, vec) in vectors.iter().enumerate() {
            let start = i * dims;
            assert_eq!(&result.vectors[start..start + dims], vec.as_slice());
        }
    }

    #[test]
    fn invalid_magic_returns_error() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bad_magic.bin");

        // Write a file with wrong magic bytes
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"XXXX");
        fs::write(&path, data).unwrap();

        let err = read_embeddings_bin(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid magic"), "error was: {msg}");
    }

    #[test]
    fn unsupported_version_returns_error() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bad_version.bin");

        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(MAGIC);
        data[4..6].copy_from_slice(&99u16.to_le_bytes()); // bad version
        fs::write(&path, data).unwrap();

        let err = read_embeddings_bin(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unsupported"), "error was: {msg}");
    }

    #[test]
    fn header_size_is_16_bytes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("header.bin");

        write_embeddings_bin(&path, &[], 4).unwrap();
        let data = fs::read(&path).unwrap();

        assert_eq!(data.len(), 16);
        // Verify header fields
        assert_eq!(&data[0..4], MAGIC);
        assert_eq!(read_u16(&data[4..6]), VERSION);
        assert_eq!(read_u16(&data[6..8]), 4); // dimensions
        assert_eq!(read_u32(&data[8..12]), 0); // count
        assert_eq!(read_u16(&data[12..14]), PRECISION_F32);
        assert_eq!(read_u16(&data[14..16]), 0); // reserved
    }
}
