use std::fs;
use std::io::Cursor;
use std::path::Path;

use engram_core::{EngramError, Result};
use half::f16;

const MAGIC: &[u8; 4] = b"EGRM";
const VERSION: u16 = 1;
/// f32 precision (4 bytes per component).
pub const PRECISION_F32: u16 = 0;
/// f16 precision (2 bytes per component, ~50% storage reduction).
pub const PRECISION_F16: u16 = 1;

/// Parsed embedding file returned by [`read_embeddings_bin`].
#[derive(Debug, Clone)]
pub struct EmbeddingFile {
    /// Number of dimensions per vector.
    pub dimensions: usize,
    /// Number of vectors stored.
    pub count: usize,
    /// Precision indicator (0 = f32, 1 = f16).
    pub precision: u16,
    /// Flattened vector data (`count * dimensions` floats), always returned as f32.
    pub vectors: Vec<f32>,
}

/// Write embedding vectors in compact binary format.
///
/// Format: 16-byte header followed by contiguous little-endian values.
/// When `precision` is [`PRECISION_F16`], f32 values are converted to f16 before writing.
pub fn write_embeddings_bin(
    path: &Path,
    vectors: &[Vec<f32>],
    dimensions: usize,
    precision: u16,
) -> Result<()> {
    let count = vectors.len() as u32;
    let dims = dimensions as u16;
    let bytes_per_component: usize = match precision {
        PRECISION_F32 => 4,
        PRECISION_F16 => 2,
        _ => {
            return Err(EngramError::Store(format!(
                "unsupported embedding precision: {precision}"
            )))
        }
    };

    // Build header (16 bytes)
    let mut buf: Vec<u8> =
        Vec::with_capacity(16 + vectors.len() * dimensions * bytes_per_component);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&dims.to_le_bytes());
    buf.extend_from_slice(&count.to_le_bytes());
    buf.extend_from_slice(&precision.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // reserved

    // Write vector data
    match precision {
        PRECISION_F32 => {
            for vec in vectors {
                for &val in vec {
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
        PRECISION_F16 => {
            for vec in vectors {
                for &val in vec {
                    let half_val = f16::from_f32(val);
                    buf.extend_from_slice(&half_val.to_le_bytes());
                }
            }
        }
        _ => unreachable!(),
    }

    fs::write(path, buf)?;
    Ok(())
}

/// Read embedding vectors from the binary format.
///
/// Validates magic bytes and version. When the file uses f16 precision,
/// values are automatically converted back to f32. Returns an [`EmbeddingFile`]
/// with the parsed header fields and f32 vector data.
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

    let bytes_per_component: usize = match precision {
        PRECISION_F32 => 4,
        PRECISION_F16 => 2,
        _ => {
            return Err(EngramError::Store(format!(
                "unsupported embedding precision in file: {precision}"
            )))
        }
    };

    let expected_data_len = count * dimensions * bytes_per_component;
    let actual_data_len = data.len() - 16;
    if actual_data_len < expected_data_len {
        return Err(EngramError::Store(format!(
            "embedding file truncated: expected {expected_data_len} bytes of vector data, got {actual_data_len}"
        )));
    }

    let vector_data = &data[16..];
    let total_elements = count * dimensions;
    let mut vectors = Vec::with_capacity(total_elements);

    match precision {
        PRECISION_F32 => {
            for i in 0..total_elements {
                let offset = i * 4;
                vectors.push(read_f32(&vector_data[offset..offset + 4]));
            }
        }
        PRECISION_F16 => {
            for i in 0..total_elements {
                let offset = i * 2;
                let half_val =
                    f16::from_le_bytes([vector_data[offset], vector_data[offset + 1]]);
                vectors.push(half_val.to_f32());
            }
        }
        _ => unreachable!(),
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

        write_embeddings_bin(&path, &[], 3, PRECISION_F32).unwrap();
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
        write_embeddings_bin(&path, &vectors, 3, PRECISION_F32).unwrap();
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

        write_embeddings_bin(&path, &vectors, dims, PRECISION_F32).unwrap();
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

        write_embeddings_bin(&path, &[], 4, PRECISION_F32).unwrap();
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

    #[test]
    fn f16_round_trip_single_vector() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("f16_single.bin");

        let vectors = vec![vec![1.0f32, 2.0, 3.0]];
        write_embeddings_bin(&path, &vectors, 3, PRECISION_F16).unwrap();
        let result = read_embeddings_bin(&path).unwrap();

        assert_eq!(result.count, 1);
        assert_eq!(result.dimensions, 3);
        assert_eq!(result.precision, PRECISION_F16);
        // f16 can represent small integers exactly
        assert_eq!(result.vectors, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn f16_round_trip_accuracy() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("f16_accuracy.bin");

        let dims = 768;
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..dims)
                    .map(|d| (i * dims + d) as f32 * 0.001)
                    .collect()
            })
            .collect();

        write_embeddings_bin(&path, &vectors, dims, PRECISION_F16).unwrap();
        let result = read_embeddings_bin(&path).unwrap();

        assert_eq!(result.count, 100);
        assert_eq!(result.dimensions, dims);
        assert_eq!(result.precision, PRECISION_F16);

        // Verify accuracy: < 0.1% relative error for non-zero values
        for (i, vec) in vectors.iter().enumerate() {
            for (d, &original) in vec.iter().enumerate() {
                let recovered = result.vectors[i * dims + d];
                if original.abs() < 1e-10 {
                    // For near-zero values, check absolute error
                    assert!(
                        (recovered - original).abs() < 1e-3,
                        "absolute error too large at [{i}][{d}]: original={original}, recovered={recovered}"
                    );
                } else {
                    let relative_error = ((recovered - original) / original).abs();
                    assert!(
                        relative_error < 0.001,
                        "relative error {relative_error:.6} >= 0.1% at [{i}][{d}]: original={original}, recovered={recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn f16_storage_is_half_size() {
        let tmp = TempDir::new().unwrap();
        let path_f32 = tmp.path().join("f32.bin");
        let path_f16 = tmp.path().join("f16.bin");

        let dims = 768;
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..dims).map(|d| (i * dims + d) as f32 * 0.001).collect())
            .collect();

        write_embeddings_bin(&path_f32, &vectors, dims, PRECISION_F32).unwrap();
        write_embeddings_bin(&path_f16, &vectors, dims, PRECISION_F16).unwrap();

        let size_f32 = fs::metadata(&path_f32).unwrap().len();
        let size_f16 = fs::metadata(&path_f16).unwrap().len();

        // f16 data should be about half the f32 data size (both have same 16-byte header)
        let data_f32 = size_f32 - 16;
        let data_f16 = size_f16 - 16;
        assert_eq!(data_f16, data_f32 / 2);
    }

    #[test]
    fn f16_header_has_correct_precision() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("f16_header.bin");

        write_embeddings_bin(&path, &[], 4, PRECISION_F16).unwrap();
        let data = fs::read(&path).unwrap();

        assert_eq!(data.len(), 16);
        assert_eq!(read_u16(&data[12..14]), PRECISION_F16);
    }

    #[test]
    fn unsupported_precision_write_returns_error() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bad_prec.bin");

        let err = write_embeddings_bin(&path, &[], 4, 99).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported embedding precision"),
            "error was: {msg}"
        );
    }

    #[test]
    fn unsupported_precision_read_returns_error() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bad_prec_read.bin");

        // Write a valid header but with unsupported precision
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(MAGIC);
        data[4..6].copy_from_slice(&VERSION.to_le_bytes());
        data[6..8].copy_from_slice(&4u16.to_le_bytes()); // dims
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // count
        data[12..14].copy_from_slice(&99u16.to_le_bytes()); // bad precision
        fs::write(&path, data).unwrap();

        let err = read_embeddings_bin(&path).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported embedding precision"),
            "error was: {msg}"
        );
    }
}
