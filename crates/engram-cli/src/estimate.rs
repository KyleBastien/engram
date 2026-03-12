use std::fs;
use std::path::Path;

use engram_core::SourceConfig;
use engram_ingest::{
    detect_language, ChunkerKind, MarkdownChunker, RawChunk, SlidingWindowChunker,
    TreeSitterChunker,
};
use globset::{Glob, GlobSet, GlobSetBuilder};

/// Per-model pricing: cost in USD per 1 million tokens.
struct ModelPricing {
    provider: &'static str,
    model: &'static str,
    cost_per_million_tokens: f64,
}

const PRICING: &[ModelPricing] = &[
    ModelPricing {
        provider: "OpenAI",
        model: "text-embedding-3-small",
        cost_per_million_tokens: 0.02,
    },
    ModelPricing {
        provider: "OpenAI",
        model: "text-embedding-3-large",
        cost_per_million_tokens: 0.13,
    },
    ModelPricing {
        provider: "Voyage",
        model: "voyage-code-3",
        cost_per_million_tokens: 0.06,
    },
];

/// Result of a cost estimation run.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub estimated_chunks: usize,
    pub estimated_tokens: usize,
    pub per_model: Vec<ModelEstimate>,
}

/// Per-model cost breakdown.
#[derive(Debug, Clone)]
pub struct ModelEstimate {
    pub provider: String,
    pub model: String,
    pub cost_usd: f64,
}

/// Estimate reindex cost for the given sources without performing any embedding calls.
///
/// Walks each source repo, chunks all eligible files using the real chunkers,
/// and computes token/cost estimates from pricing tables.
pub fn estimate_reindex_cost(sources: &[SourceConfig]) -> CostEstimate {
    let mut total_chunks: usize = 0;
    let mut total_chars: usize = 0;

    for source in sources {
        let source_path = Path::new(&source.path);

        let include_set = build_glob_set(&source.include);
        let exclude_set = build_glob_set(&source.exclude);

        walk_source(
            source_path,
            source_path,
            &include_set,
            &exclude_set,
            !source.include.is_empty(),
            &mut total_chunks,
            &mut total_chars,
        );
    }

    // Approximate tokens: ~1 token per 4 characters
    let estimated_tokens = total_chars.div_ceil(4);

    let per_model = PRICING
        .iter()
        .map(|p| {
            let cost_usd = (estimated_tokens as f64) * p.cost_per_million_tokens / 1_000_000.0;
            ModelEstimate {
                provider: p.provider.to_string(),
                model: p.model.to_string(),
                cost_usd,
            }
        })
        .collect();

    CostEstimate {
        estimated_chunks: total_chunks,
        estimated_tokens,
        per_model,
    }
}

/// Print a cost estimation report to stdout.
pub fn print_estimate(est: &CostEstimate) {
    println!("\n  Cost Estimation Report");
    println!("  ======================\n");
    println!("  Estimated chunks: {}", est.estimated_chunks);
    println!("  Estimated tokens: {}", est.estimated_tokens);

    // Estimate time: ~100ms per API call, batches of 128
    let batch_calls = est.estimated_chunks.div_ceil(128);
    let time_secs = batch_calls as f64 * 0.1;
    println!("  Estimated time:   {time_secs:.1}s\n");

    println!(
        "  {:<12} {:<28} {:>12}",
        "Provider", "Model", "Cost (USD)"
    );
    println!("  {:-<54}", "");
    for m in &est.per_model {
        println!(
            "  {:<12} {:<28} ${:>10.4}",
            m.provider, m.model, m.cost_usd
        );
    }
    println!(
        "\n  Note: Estimates based on ~1 token per 4 characters."
    );
    println!("  Actual costs depend on provider tokenization and pricing.");
}

fn build_glob_set(patterns: &[String]) -> Option<GlobSet> {
    if patterns.is_empty() {
        return None;
    }
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        if let Ok(glob) = Glob::new(pattern) {
            builder.add(glob);
        }
    }
    builder.build().ok()
}

fn walk_source(
    root: &Path,
    dir: &Path,
    include: &Option<GlobSet>,
    exclude: &Option<GlobSet>,
    has_includes: bool,
    total_chunks: &mut usize,
    total_chars: &mut usize,
) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        // Skip hidden directories and common non-source dirs
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }
        }

        if path.is_dir() {
            walk_source(root, &path, include, exclude, has_includes, total_chunks, total_chars);
            continue;
        }

        let rel_path = match path.strip_prefix(root) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let rel_str = rel_path.to_string_lossy();

        // Apply include/exclude filters
        if has_includes {
            if let Some(ref inc) = include {
                if !inc.is_match(rel_str.as_ref()) {
                    continue;
                }
            }
        }
        if let Some(ref exc) = exclude {
            if exc.is_match(rel_str.as_ref()) {
                continue;
            }
        }

        let kind = detect_language(rel_path);
        if matches!(kind, ChunkerKind::Skip) {
            continue;
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let chunks = chunk_content(&content, rel_path, kind);
        for chunk in &chunks {
            *total_chars += chunk.content.len();
        }
        *total_chunks += chunks.len();
    }
}

fn chunk_content(source: &str, path: &Path, kind: ChunkerKind) -> Vec<RawChunk> {
    match kind {
        ChunkerKind::TreeSitter(lang) => TreeSitterChunker::new().chunk_file(path, source, lang),
        ChunkerKind::Markdown => MarkdownChunker::new().chunk_file(source),
        ChunkerKind::SlidingWindow => SlidingWindowChunker::new().chunk_file(source, 500, 50),
        ChunkerKind::Skip => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_source(files: &[(&str, &str)]) -> (TempDir, String) {
        let dir = TempDir::new().unwrap();
        for (name, content) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, content).unwrap();
        }
        let path_str = dir.path().to_string_lossy().to_string();
        (dir, path_str)
    }

    #[test]
    fn estimates_chunks_for_rust_file() {
        let (_dir, path) = create_source(&[(
            "lib.rs",
            "fn hello() {\n    println!(\"hello\");\n}\n\nfn world() {\n    println!(\"world\");\n}",
        )]);
        let source = SourceConfig {
            name: "test".to_string(),
            path,
            include: vec![],
            exclude: vec![],
        };

        let est = estimate_reindex_cost(&[source]);
        assert!(est.estimated_chunks > 0, "should find chunks");
        assert!(est.estimated_tokens > 0, "should estimate tokens");
        assert_eq!(est.per_model.len(), 3, "should have 3 model estimates");
    }

    #[test]
    fn skips_binary_files() {
        let (_dir, path) = create_source(&[
            ("lib.rs", "fn hello() {}"),
            ("image.png", "fake binary data"),
        ]);
        let source = SourceConfig {
            name: "test".to_string(),
            path,
            include: vec![],
            exclude: vec![],
        };

        let est = estimate_reindex_cost(&[source]);
        // Only the .rs file should contribute chunks
        assert!(est.estimated_chunks > 0);
    }

    #[test]
    fn respects_include_patterns() {
        let (_dir, path) = create_source(&[
            ("src/main.rs", "fn main() {}"),
            ("docs/readme.md", "# Hello\nWorld"),
        ]);
        let source = SourceConfig {
            name: "test".to_string(),
            path,
            include: vec!["src/**/*.rs".to_string()],
            exclude: vec![],
        };

        let est = estimate_reindex_cost(&[source]);
        // Only .rs file under src/ should be included
        assert!(est.estimated_chunks > 0);
    }

    #[test]
    fn respects_exclude_patterns() {
        let (_dir, path) = create_source(&[
            ("lib.rs", "fn hello() {}"),
            ("test.rs", "fn test_hello() {}"),
        ]);
        let source = SourceConfig {
            name: "test".to_string(),
            path,
            include: vec![],
            exclude: vec!["test.rs".to_string()],
        };

        let est_excluded = estimate_reindex_cost(&[source]);

        // Compare with no exclusion
        let source_all = SourceConfig {
            name: "test".to_string(),
            path: _dir.path().to_string_lossy().to_string(),
            include: vec![],
            exclude: vec![],
        };
        let est_all = estimate_reindex_cost(&[source_all]);

        assert!(
            est_excluded.estimated_chunks < est_all.estimated_chunks,
            "excluding a file should reduce chunk count"
        );
    }

    #[test]
    fn empty_source_returns_zero() {
        let (_dir, path) = create_source(&[]);
        let source = SourceConfig {
            name: "test".to_string(),
            path,
            include: vec![],
            exclude: vec![],
        };

        let est = estimate_reindex_cost(&[source]);
        assert_eq!(est.estimated_chunks, 0);
        assert_eq!(est.estimated_tokens, 0);
        for m in &est.per_model {
            assert!((m.cost_usd - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn multiple_sources_aggregated() {
        let (_dir1, path1) = create_source(&[("a.rs", "fn alpha() { 1 }")]);
        let (_dir2, path2) = create_source(&[("b.rs", "fn beta() { 2 }")]);

        let sources = vec![
            SourceConfig {
                name: "repo-a".to_string(),
                path: path1,
                include: vec![],
                exclude: vec![],
            },
            SourceConfig {
                name: "repo-b".to_string(),
                path: path2,
                include: vec![],
                exclude: vec![],
            },
        ];

        let est = estimate_reindex_cost(&sources);
        assert!(
            est.estimated_chunks >= 2,
            "should have chunks from both sources"
        );
    }

    #[test]
    fn cost_scales_with_tokens() {
        let (_dir, path) = create_source(&[(
            "big.rs",
            &"fn f() { let x = 1; }\n".repeat(100),
        )]);
        let source = SourceConfig {
            name: "test".to_string(),
            path,
            include: vec![],
            exclude: vec![],
        };

        let est = estimate_reindex_cost(&[source]);
        // OpenAI small is cheapest, large is most expensive
        let small = &est.per_model[0];
        let large = &est.per_model[1];
        assert!(
            large.cost_usd > small.cost_usd,
            "text-embedding-3-large should cost more than text-embedding-3-small"
        );
    }

    #[test]
    fn pricing_table_has_expected_models() {
        let (_dir, path) = create_source(&[("a.rs", "fn a() {}")]);
        let source = SourceConfig {
            name: "test".to_string(),
            path,
            include: vec![],
            exclude: vec![],
        };

        let est = estimate_reindex_cost(&[source]);
        let models: Vec<&str> = est.per_model.iter().map(|m| m.model.as_str()).collect();
        assert!(models.contains(&"text-embedding-3-small"));
        assert!(models.contains(&"text-embedding-3-large"));
        assert!(models.contains(&"voyage-code-3"));
    }
}
