use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use engram_core::{EmbedError, EmbeddingProvider};
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

/// Device configuration for ONNX inference.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Device {
    /// CPU inference (default).
    #[default]
    Cpu,
    /// CUDA GPU inference (requires ONNX Runtime with CUDA support).
    Cuda,
}

/// Configuration for the ONNX embedding provider.
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// Model name used for directory naming and display (e.g., "all-MiniLM-L6-v2").
    pub model_name: String,
    /// Root directory for storing downloaded models.
    pub model_dir: PathBuf,
    /// Output embedding dimensions.
    pub dimensions: usize,
    /// Maximum batch size for embed calls.
    pub max_batch_size: usize,
    /// Whether to use INT8 quantized model.
    pub quantized: bool,
    /// Inference device.
    pub device: Device,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        let model_dir = home_dir().join(".engram").join("models");
        Self {
            model_name: "all-MiniLM-L6-v2".to_string(),
            model_dir,
            dimensions: 384,
            max_batch_size: 256,
            quantized: false,
            device: Device::Cpu,
        }
    }
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

impl OnnxConfig {
    /// Returns the directory for this specific model.
    pub fn model_path(&self) -> PathBuf {
        self.model_dir.join(&self.model_name)
    }

    /// Returns the ONNX model filename based on quantization setting.
    pub fn model_filename(&self) -> &str {
        if self.quantized {
            "model_quantized.onnx"
        } else {
            "model.onnx"
        }
    }

    /// Returns the full path to the ONNX model file.
    pub fn model_file_path(&self) -> PathBuf {
        self.model_path().join(self.model_filename())
    }

    /// Returns the full path to the tokenizer file.
    pub fn tokenizer_file_path(&self) -> PathBuf {
        self.model_path().join("tokenizer.json")
    }

    /// Returns the HuggingFace download URL for the model file.
    pub fn model_download_url(&self) -> String {
        format!(
            "https://huggingface.co/sentence-transformers/{}/resolve/main/onnx/{}",
            self.model_name,
            self.model_filename()
        )
    }

    /// Returns the HuggingFace download URL for the tokenizer.
    pub fn tokenizer_download_url(&self) -> String {
        format!(
            "https://huggingface.co/sentence-transformers/{}/resolve/main/tokenizer.json",
            self.model_name
        )
    }
}

/// ONNX-based embedding provider that runs inference in-process.
pub struct OnnxProvider {
    config: OnnxConfig,
    display_name: String,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl OnnxProvider {
    /// Creates a new ONNX provider, downloading the model if not already cached.
    pub async fn new(config: OnnxConfig) -> Result<Self, EmbedError> {
        ensure_model_downloaded(&config).await?;
        Self::from_local(config)
    }

    /// Creates a provider from an already-downloaded model directory.
    pub fn from_local(config: OnnxConfig) -> Result<Self, EmbedError> {
        let model_path = config.model_file_path();
        let tokenizer_path = config.tokenizer_file_path();

        if !model_path.exists() {
            return Err(EmbedError::Unavailable(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(EmbedError::Unavailable(format!(
                "Tokenizer file not found: {}",
                tokenizer_path.display()
            )));
        }

        let session = create_session(&model_path, &config.device)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::Unavailable(format!("Failed to load tokenizer: {e}")))?;

        let display_name = format!("onnx/{}", config.model_name);
        Ok(Self {
            config,
            display_name,
            session: Mutex::new(session),
            tokenizer,
        })
    }
}

fn create_session(model_path: &Path, device: &Device) -> Result<Session, EmbedError> {
    let mut builder = Session::builder()
        .map_err(|e| EmbedError::Unavailable(format!("Failed to create session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| EmbedError::Unavailable(format!("Failed to set optimization level: {e}")))?;

    match device {
        Device::Cpu => builder
            .commit_from_file(model_path)
            .map_err(|e| EmbedError::Unavailable(format!("Failed to load model: {e}"))),
        Device::Cuda => builder
            .with_execution_providers([ort::ep::CUDA::default().build()])
            .map_err(|e| EmbedError::Unavailable(format!("Failed to configure CUDA: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| EmbedError::Unavailable(format!("Failed to load model: {e}"))),
    }
}

async fn ensure_model_downloaded(config: &OnnxConfig) -> Result<(), EmbedError> {
    let model_path = config.model_file_path();
    let tokenizer_path = config.tokenizer_file_path();

    if model_path.exists() && tokenizer_path.exists() {
        return Ok(());
    }

    std::fs::create_dir_all(config.model_path())
        .map_err(|e| EmbedError::Unavailable(format!("Failed to create model directory: {e}")))?;

    let client = reqwest::Client::new();

    if !model_path.exists() {
        download_file(&client, &config.model_download_url(), &model_path).await?;
    }

    if !tokenizer_path.exists() {
        download_file(&client, &config.tokenizer_download_url(), &tokenizer_path).await?;
    }

    Ok(())
}

async fn download_file(
    client: &reqwest::Client,
    url: &str,
    path: &Path,
) -> Result<(), EmbedError> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| EmbedError::Unavailable(format!("Download failed: {e}")))?;

    if !response.status().is_success() {
        return Err(EmbedError::Unavailable(format!(
            "Download failed with status {}: {}",
            response.status(),
            url
        )));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| EmbedError::Unavailable(format!("Failed to read response: {e}")))?;

    std::fs::write(path, &bytes)
        .map_err(|e| EmbedError::Unavailable(format!("Failed to write file: {e}")))?;

    Ok(())
}

/// Builds padded input tensors from tokenizer encodings.
fn build_input_tensors(
    encodings: &[tokenizers::Encoding],
    max_len: usize,
    batch_size: usize,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let mut input_ids = Array2::<i64>::zeros((batch_size, max_len));
    let mut attention_mask = Array2::<i64>::zeros((batch_size, max_len));
    let mut token_type_ids = Array2::<i64>::zeros((batch_size, max_len));

    for (i, encoding) in encodings.iter().enumerate() {
        for (j, &id) in encoding.get_ids().iter().enumerate() {
            input_ids[[i, j]] = id as i64;
        }
        for (j, &mask) in encoding.get_attention_mask().iter().enumerate() {
            attention_mask[[i, j]] = mask as i64;
        }
        for (j, &type_id) in encoding.get_type_ids().iter().enumerate() {
            token_type_ids[[i, j]] = type_id as i64;
        }
    }

    (input_ids, attention_mask, token_type_ids)
}

/// Performs mean pooling over token embeddings using the attention mask, then L2-normalizes.
fn mean_pool_and_normalize(
    output: &ndarray::ArrayViewD<'_, f32>,
    attention_mask: &Array2<i64>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let shape = output.shape();
    if shape.len() < 2 {
        return Err(EmbedError::Api(format!(
            "Unexpected output shape: {shape:?}"
        )));
    }

    // 2D output (batch, hidden) — already pooled by the model
    if shape.len() == 2 {
        let hidden_size = shape[1];
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut embedding = Vec::with_capacity(hidden_size);
            for k in 0..hidden_size {
                embedding.push(output[ndarray::IxDyn(&[i, k])]);
            }
            normalize(&mut embedding);
            results.push(embedding);
        }
        return Ok(results);
    }

    // 3D output (batch, seq_len, hidden) — mean pool over token dimension
    let seq_len = shape[1];
    let hidden_size = shape[2];
    let mut results = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let mut embedding = vec![0.0f32; hidden_size];
        let mut mask_sum = 0.0f32;

        for j in 0..seq_len {
            let mask_val = attention_mask[[i, j]] as f32;
            mask_sum += mask_val;
            for k in 0..hidden_size {
                embedding[k] += output[ndarray::IxDyn(&[i, j, k])] * mask_val;
            }
        }

        if mask_sum > 0.0 {
            for val in &mut embedding {
                *val /= mask_sum;
            }
        }

        normalize(&mut embedding);
        results.push(embedding);
    }

    Ok(results)
}

/// L2-normalizes a vector in-place.
fn normalize(embedding: &mut [f32]) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in embedding.iter_mut() {
            *val /= norm;
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OnnxProvider {
    fn name(&self) -> &str {
        &self.display_name
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn max_batch_size(&self) -> usize {
        self.config.max_batch_size
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Api(format!("Tokenization failed: {e}")))?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let batch_size = texts.len();

        let (input_ids, attention_mask, token_type_ids) =
            build_input_tensors(&encodings, max_len, batch_size);

        // Create tensor references for ONNX Runtime
        let input_ids_tensor = TensorRef::from_array_view(input_ids.view())
            .map_err(|e| EmbedError::Api(format!("Failed to create input tensor: {e}")))?;
        let mask_tensor = TensorRef::from_array_view(attention_mask.view())
            .map_err(|e| EmbedError::Api(format!("Failed to create input tensor: {e}")))?;
        let type_ids_tensor = TensorRef::from_array_view(token_type_ids.view())
            .map_err(|e| EmbedError::Api(format!("Failed to create input tensor: {e}")))?;

        // Run inference (session.run requires &mut self)
        let mut session = self
            .session
            .lock()
            .map_err(|e| EmbedError::Api(format!("Failed to lock session: {e}")))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => type_ids_tensor,
            ])
            .map_err(|e| EmbedError::Api(format!("Inference failed: {e}")))?;

        // Extract output and apply mean pooling + normalization
        let output_array = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| EmbedError::Api(format!("Failed to extract output: {e}")))?;

        mean_pool_and_normalize(&output_array, &attention_mask, batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OnnxConfig::default();
        assert_eq!(config.model_name, "all-MiniLM-L6-v2");
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.max_batch_size, 256);
        assert!(!config.quantized);
        assert_eq!(config.device, Device::Cpu);
    }

    #[test]
    fn test_model_filename() {
        let mut config = OnnxConfig::default();
        assert_eq!(config.model_filename(), "model.onnx");

        config.quantized = true;
        assert_eq!(config.model_filename(), "model_quantized.onnx");
    }

    #[test]
    fn test_model_path_resolution() {
        let config = OnnxConfig {
            model_dir: PathBuf::from("/tmp/models"),
            model_name: "test-model".to_string(),
            ..OnnxConfig::default()
        };
        assert_eq!(config.model_path(), PathBuf::from("/tmp/models/test-model"));
        assert_eq!(
            config.model_file_path(),
            PathBuf::from("/tmp/models/test-model/model.onnx")
        );
        assert_eq!(
            config.tokenizer_file_path(),
            PathBuf::from("/tmp/models/test-model/tokenizer.json")
        );
    }

    #[test]
    fn test_quantized_model_path() {
        let config = OnnxConfig {
            model_dir: PathBuf::from("/tmp/models"),
            model_name: "test-model".to_string(),
            quantized: true,
            ..OnnxConfig::default()
        };
        assert_eq!(
            config.model_file_path(),
            PathBuf::from("/tmp/models/test-model/model_quantized.onnx")
        );
    }

    #[test]
    fn test_download_urls() {
        let config = OnnxConfig::default();
        assert_eq!(
            config.model_download_url(),
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
        );
        assert_eq!(
            config.tokenizer_download_url(),
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
        );
    }

    #[test]
    fn test_quantized_download_url() {
        let config = OnnxConfig {
            quantized: true,
            ..OnnxConfig::default()
        };
        assert!(config
            .model_download_url()
            .contains("model_quantized.onnx"));
    }

    #[test]
    fn test_device_default() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn test_normalize_unit_vector() {
        let mut v = vec![1.0, 0.0, 0.0];
        normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 0.0).abs() < 1e-6);
        assert!((v[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_scales_correctly() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        // norm = 5.0, so [0.6, 0.8]
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize(&mut v);
        // Should remain zero (no division by zero)
        assert!((v[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_single_token() {
        let output = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 1, 3]),
            vec![1.0f32, 0.0, 0.0],
        )
        .unwrap();
        let attention_mask = ndarray::array![[1i64]];
        let result = mean_pool_and_normalize(&output.view(), &attention_mask, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0][0] - 1.0).abs() < 1e-6);
        assert!((result[0][1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_with_mask() {
        // Two tokens, second masked out
        let output = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 2, 2]),
            vec![2.0f32, 0.0, 0.0, 4.0],
        )
        .unwrap();
        let attention_mask = ndarray::array![[1i64, 0]];
        let result = mean_pool_and_normalize(&output.view(), &attention_mask, 1).unwrap();
        assert_eq!(result.len(), 1);
        // Mean pool: [2.0, 0.0] / 1.0 = [2.0, 0.0], normalized: [1.0, 0.0]
        assert!((result[0][0] - 1.0).abs() < 1e-6);
        assert!((result[0][1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_batch() {
        let output = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 1, 2]),
            vec![1.0f32, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let attention_mask = ndarray::array![[1i64], [1i64]];
        let result = mean_pool_and_normalize(&output.view(), &attention_mask, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0][0] - 1.0).abs() < 1e-6);
        assert!((result[0][1] - 0.0).abs() < 1e-6);
        assert!((result[1][0] - 0.0).abs() < 1e-6);
        assert!((result[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_normalization() {
        // Vector [3.0, 4.0] should normalize to [0.6, 0.8]
        let output =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 2]), vec![3.0f32, 4.0])
                .unwrap();
        let attention_mask = ndarray::array![[1i64]];
        let result = mean_pool_and_normalize(&output.view(), &attention_mask, 1).unwrap();
        assert!((result[0][0] - 0.6).abs() < 1e-6);
        assert!((result[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_2d_output() {
        // Some models output already-pooled (batch, hidden) tensors
        let output = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 2]),
            vec![3.0f32, 4.0, 0.0, 5.0],
        )
        .unwrap();
        let attention_mask = ndarray::array![[1i64], [1i64]];
        let result = mean_pool_and_normalize(&output.view(), &attention_mask, 2).unwrap();
        assert_eq!(result.len(), 2);
        // First: [3, 4] normalized = [0.6, 0.8]
        assert!((result[0][0] - 0.6).abs() < 1e-6);
        assert!((result[0][1] - 0.8).abs() < 1e-6);
        // Second: [0, 5] normalized = [0, 1]
        assert!((result[1][0] - 0.0).abs() < 1e-6);
        assert!((result[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_multiple_tokens_averaged() {
        // Two tokens both active: [1, 0] and [0, 1] -> mean = [0.5, 0.5]
        let output = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 2, 2]),
            vec![1.0f32, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let attention_mask = ndarray::array![[1i64, 1]];
        let result = mean_pool_and_normalize(&output.view(), &attention_mask, 1).unwrap();
        // Mean: [0.5, 0.5], norm = sqrt(0.5), normalized = [1/sqrt(2), 1/sqrt(2)]
        let expected = 1.0 / 2.0f32.sqrt();
        assert!((result[0][0] - expected).abs() < 1e-6);
        assert!((result[0][1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_build_input_tensors_empty() {
        let encodings: Vec<tokenizers::Encoding> = vec![];
        let (ids, mask, types) = build_input_tensors(&encodings, 0, 0);
        assert_eq!(ids.shape(), &[0, 0]);
        assert_eq!(mask.shape(), &[0, 0]);
        assert_eq!(types.shape(), &[0, 0]);
    }

    #[test]
    fn test_from_local_missing_model() {
        let config = OnnxConfig {
            model_dir: PathBuf::from("/nonexistent/path"),
            ..OnnxConfig::default()
        };
        let result = OnnxProvider::from_local(config);
        assert!(result.is_err());
        match result {
            Err(EmbedError::Unavailable(msg)) => {
                assert!(msg.contains("not found"), "unexpected message: {msg}");
            }
            Err(other) => panic!("expected Unavailable error, got: {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn test_mean_pool_rejects_1d() {
        let output =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0f32, 2.0, 3.0]).unwrap();
        let attention_mask = ndarray::array![[1i64]];
        let result = mean_pool_and_normalize(&output.view(), &attention_mask, 1);
        assert!(result.is_err());
    }
}
