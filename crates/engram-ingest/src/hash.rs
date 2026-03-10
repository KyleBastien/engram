use sha2::{Digest, Sha256};

/// Returns a hex-encoded SHA-256 hash of the given content.
pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Returns true if the chunk has changed (i.e., the hashes differ).
pub fn has_chunk_changed(old_hash: &str, new_hash: &str) -> bool {
    old_hash != new_hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash_deterministic() {
        let content = "fn main() { println!(\"hello\"); }";
        let hash1 = content_hash(content);
        let hash2 = content_hash(content);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_content_hash_hex_encoded() {
        let hash = content_hash("hello");
        // SHA-256 produces 32 bytes = 64 hex characters
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_content_hash_different_content() {
        let hash1 = content_hash("hello");
        let hash2 = content_hash("world");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_content_hash_empty_string() {
        let hash = content_hash("");
        assert_eq!(hash.len(), 64);
        // Known SHA-256 of empty string
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_content_hash_known_value() {
        // SHA-256 of "hello" is well-known
        let hash = content_hash("hello");
        assert_eq!(
            hash,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_has_chunk_changed_same() {
        let hash = content_hash("some code");
        assert!(!has_chunk_changed(&hash, &hash));
    }

    #[test]
    fn test_has_chunk_changed_different() {
        let hash1 = content_hash("version 1");
        let hash2 = content_hash("version 2");
        assert!(has_chunk_changed(&hash1, &hash2));
    }

    #[test]
    fn test_content_hash_whitespace_sensitive() {
        let hash1 = content_hash("hello world");
        let hash2 = content_hash("hello  world");
        assert_ne!(hash1, hash2);
    }
}
