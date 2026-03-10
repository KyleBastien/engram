use std::path::Path;

use crate::Language;

/// The kind of chunker to use for a given file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkerKind {
    TreeSitter(Language),
    Markdown,
    SlidingWindow,
    Skip,
}

/// Detect the appropriate chunker kind for a file based on its extension.
pub fn detect_language(path: &Path) -> ChunkerKind {
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e.to_lowercase(),
        None => return ChunkerKind::SlidingWindow,
    };

    match ext.as_str() {
        // Tree-sitter supported languages
        "ts" | "tsx" => ChunkerKind::TreeSitter(Language::TypeScript),
        "rs" => ChunkerKind::TreeSitter(Language::Rust),
        "py" => ChunkerKind::TreeSitter(Language::Python),

        // Markdown
        "md" | "mdx" => ChunkerKind::Markdown,

        // Binary extensions — skip
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" | "svg" | "webp" | "tiff" | "tif"
        | "wasm" | "bin" | "exe" | "dll" | "so" | "dylib" | "o" | "a" | "lib" | "obj"
        | "zip" | "tar" | "gz" | "bz2" | "xz" | "7z" | "rar" | "zst" | "pdf" | "doc"
        | "docx" | "xls" | "xlsx" | "ppt" | "pptx" | "mp3" | "mp4" | "avi" | "mov" | "mkv"
        | "wav" | "flac" | "ogg" | "ttf" | "otf" | "woff" | "woff2" | "eot" | "class"
        | "pyc" | "pyo" | "db" | "sqlite" | "sqlite3" => ChunkerKind::Skip,

        // Everything else: sliding window fallback
        _ => ChunkerKind::SlidingWindow,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typescript_ts() {
        assert_eq!(
            detect_language(Path::new("src/index.ts")),
            ChunkerKind::TreeSitter(Language::TypeScript)
        );
    }

    #[test]
    fn typescript_tsx() {
        assert_eq!(
            detect_language(Path::new("component.tsx")),
            ChunkerKind::TreeSitter(Language::TypeScript)
        );
    }

    #[test]
    fn rust_rs() {
        assert_eq!(
            detect_language(Path::new("main.rs")),
            ChunkerKind::TreeSitter(Language::Rust)
        );
    }

    #[test]
    fn python_py() {
        assert_eq!(
            detect_language(Path::new("script.py")),
            ChunkerKind::TreeSitter(Language::Python)
        );
    }

    #[test]
    fn markdown_md() {
        assert_eq!(
            detect_language(Path::new("README.md")),
            ChunkerKind::Markdown
        );
    }

    #[test]
    fn markdown_mdx() {
        assert_eq!(
            detect_language(Path::new("docs/page.mdx")),
            ChunkerKind::Markdown
        );
    }

    #[test]
    fn sliding_window_for_text_files() {
        assert_eq!(
            detect_language(Path::new("config.toml")),
            ChunkerKind::SlidingWindow
        );
        assert_eq!(
            detect_language(Path::new("Makefile.txt")),
            ChunkerKind::SlidingWindow
        );
        assert_eq!(
            detect_language(Path::new("data.json")),
            ChunkerKind::SlidingWindow
        );
        assert_eq!(
            detect_language(Path::new("style.css")),
            ChunkerKind::SlidingWindow
        );
    }

    #[test]
    fn skip_binary_png() {
        assert_eq!(
            detect_language(Path::new("image.png")),
            ChunkerKind::Skip
        );
    }

    #[test]
    fn skip_binary_jpg() {
        assert_eq!(
            detect_language(Path::new("photo.jpg")),
            ChunkerKind::Skip
        );
    }

    #[test]
    fn skip_binary_wasm() {
        assert_eq!(
            detect_language(Path::new("module.wasm")),
            ChunkerKind::Skip
        );
    }

    #[test]
    fn skip_binary_bin() {
        assert_eq!(
            detect_language(Path::new("data.bin")),
            ChunkerKind::Skip
        );
    }

    #[test]
    fn skip_binary_exe() {
        assert_eq!(
            detect_language(Path::new("program.exe")),
            ChunkerKind::Skip
        );
    }

    #[test]
    fn skip_binary_zip() {
        assert_eq!(
            detect_language(Path::new("archive.zip")),
            ChunkerKind::Skip
        );
    }

    #[test]
    fn no_extension_uses_sliding_window() {
        assert_eq!(
            detect_language(Path::new("Makefile")),
            ChunkerKind::SlidingWindow
        );
    }

    #[test]
    fn case_insensitive_extension() {
        assert_eq!(
            detect_language(Path::new("file.RS")),
            ChunkerKind::TreeSitter(Language::Rust)
        );
        assert_eq!(
            detect_language(Path::new("file.Py")),
            ChunkerKind::TreeSitter(Language::Python)
        );
        assert_eq!(
            detect_language(Path::new("file.MD")),
            ChunkerKind::Markdown
        );
        assert_eq!(
            detect_language(Path::new("image.PNG")),
            ChunkerKind::Skip
        );
    }
}
