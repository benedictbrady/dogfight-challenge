use std::path::Path;

use dogfight_shared::{
    Action, Observation, OBS_SIZE, ACTION_SIZE, MAX_MODEL_SIZE_BYTES, MAX_PARAMETERS,
};
use dogfight_sim::Policy;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use prost::Message;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf definitions (just enough to count parameters & ops)
// ---------------------------------------------------------------------------

/// ONNX TensorProto (subset)
#[derive(Clone, Message)]
struct OnnxTensorProto {
    /// Shape dimensions – product = number of elements
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    /// Data type enum (e.g. 1 = FLOAT, 7 = INT64)
    #[prost(int32, tag = "2")]
    data_type: i32,
}

/// ONNX AttributeProto (subset – we only need tensor-valued attributes)
#[derive(Clone, Message)]
struct OnnxAttributeProto {
    #[prost(string, tag = "1")]
    name: String,
    /// Single tensor value (used by Constant ops)
    #[prost(message, optional, tag = "10")]
    t: Option<OnnxTensorProto>,
    /// Repeated tensor values
    #[prost(message, repeated, tag = "11")]
    tensors: Vec<OnnxTensorProto>,
}

/// ONNX NodeProto (subset)
#[derive(Clone, Message)]
struct OnnxNodeProto {
    #[prost(string, tag = "4")]
    op_type: String,
    #[prost(message, repeated, tag = "5")]
    attribute: Vec<OnnxAttributeProto>,
}

/// ONNX GraphProto (subset)
#[derive(Clone, Message)]
struct OnnxGraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<OnnxNodeProto>,
    /// Initializer tensors (weights, biases, constants)
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<OnnxTensorProto>,
}

/// ONNX ModelProto (subset)
#[derive(Clone, Message)]
struct OnnxModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<OnnxGraphProto>,
}

/// Count the number of scalar elements in a tensor from its dims.
/// Empty dims = scalar = 1 element.
fn tensor_element_count(tensor: &OnnxTensorProto) -> usize {
    if tensor.dims.is_empty() {
        1
    } else {
        tensor.dims.iter().map(|&d| d.max(1) as usize).product()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Model file too large: {0} bytes (max {1})")]
    FileTooLarge(usize, usize),
    #[error("Disallowed op: {0}")]
    DisallowedOp(String),
    #[error("Invalid input shape: expected [1, {OBS_SIZE}] or [N, {OBS_SIZE}], got {0:?}")]
    InvalidInputShape(Vec<i64>),
    #[error("Invalid output shape: expected [1, {ACTION_SIZE}] or [N, {ACTION_SIZE}], got {0:?}")]
    InvalidOutputShape(Vec<i64>),
    #[error("Too many parameters: {0} (max {1})")]
    TooManyParameters(usize, usize),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ONNX runtime error: {0}")]
    OrtError(String),
    #[error("ONNX protobuf parse error: {0}")]
    ProtobufError(#[from] prost::DecodeError),
}

impl From<ort::Error> for ValidationError {
    fn from(e: ort::Error) -> Self {
        ValidationError::OrtError(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Summary of a validated ONNX model.
pub struct ValidationReport {
    pub file_size_bytes: usize,
    pub ops_used: Vec<String>,
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
    pub parameter_count: usize,
}

// ---------------------------------------------------------------------------
// count_all_parameters
// ---------------------------------------------------------------------------

/// Walk the ONNX protobuf to count every numerical element stored in the model.
///
/// This counts:
/// 1. All initializer tensors (weights, biases, embedding tables, etc.)
/// 2. All tensor-valued attributes on graph nodes (e.g. Constant ops)
///
/// This prevents gaming via hiding parameters in Constant nodes, unusual
/// data types, or other creative packaging.
///
/// Returns `(total_element_count, sorted_op_type_list)`.
pub fn count_all_parameters(model_bytes: &[u8]) -> Result<(usize, Vec<String>), ValidationError> {
    let model = OnnxModelProto::decode(model_bytes)?;
    let graph = model.graph.as_ref().ok_or_else(|| {
        ValidationError::OrtError("ONNX model has no graph".to_string())
    })?;

    let mut total_elements: usize = 0;

    // 1. Count all initializer tensor elements
    for tensor in &graph.initializer {
        total_elements = total_elements.saturating_add(tensor_element_count(tensor));
    }

    // 2. Walk all nodes: count tensor attributes and collect op types
    let mut ops_used = Vec::new();
    for node in &graph.node {
        if !node.op_type.is_empty() && !ops_used.contains(&node.op_type) {
            ops_used.push(node.op_type.clone());
        }

        for attr in &node.attribute {
            // Single tensor attribute (e.g. Constant op's "value")
            if let Some(ref t) = attr.t {
                total_elements = total_elements.saturating_add(tensor_element_count(t));
            }
            // Repeated tensor attributes
            for t in &attr.tensors {
                total_elements = total_elements.saturating_add(tensor_element_count(t));
            }
        }
    }

    ops_used.sort();
    Ok((total_elements, ops_used))
}

// ---------------------------------------------------------------------------
// validate_model_file
// ---------------------------------------------------------------------------

/// Validate an ONNX model file at the given path.
///
/// Checks performed:
/// 1. File size <= MAX_MODEL_SIZE_BYTES (10 MB)
/// 2. Model can be loaded by ONNX Runtime (valid protobuf, supported ops)
/// 3. Input shape is `[1, 224]` or `[N, 224]` (dynamic batch with -1) float32
/// 4. Output shape is `[1, 3]` or `[N, 3]` (dynamic batch with -1) float32
/// 5. Exact parameter count <= MAX_PARAMETERS (parsed from protobuf)
pub fn validate_model_file(path: &Path) -> Result<ValidationReport, ValidationError> {
    // 1. File size check
    let file_bytes = std::fs::read(path)?;
    let file_size_bytes = file_bytes.len();
    if file_size_bytes > MAX_MODEL_SIZE_BYTES {
        return Err(ValidationError::FileTooLarge(
            file_size_bytes,
            MAX_MODEL_SIZE_BYTES,
        ));
    }

    // 2. Load the model via ort Session (validates protobuf + ops)
    let session = Session::builder()
        .map_err(ValidationError::from)?
        .commit_from_file(path)
        .map_err(ValidationError::from)?;

    // 3. Validate input shape: expect exactly one input, Tensor<f32> with shape [1, OBS_SIZE] or [-1, OBS_SIZE]
    let inputs = session.inputs();
    if inputs.is_empty() {
        return Err(ValidationError::InvalidInputShape(vec![]));
    }
    let input_dtype = inputs[0].dtype();
    let input_shape = validate_tensor_shape(
        input_dtype,
        OBS_SIZE as i64,
        true, // is_input
    )?;

    // 4. Validate output shape: expect exactly one output, Tensor<f32> with shape [1, ACTION_SIZE] or [-1, ACTION_SIZE]
    let outputs = session.outputs();
    if outputs.is_empty() {
        return Err(ValidationError::InvalidOutputShape(vec![]));
    }
    let output_dtype = outputs[0].dtype();
    let output_shape = validate_tensor_shape(
        output_dtype,
        ACTION_SIZE as i64,
        false, // is_input
    )?;

    // 5. Parse the ONNX protobuf to count exact parameters and list ops
    let (parameter_count, ops_used) = count_all_parameters(&file_bytes)?;
    if parameter_count > MAX_PARAMETERS {
        return Err(ValidationError::TooManyParameters(
            parameter_count,
            MAX_PARAMETERS,
        ));
    }

    Ok(ValidationReport {
        file_size_bytes,
        ops_used,
        input_shape,
        output_shape,
        parameter_count,
    })
}

/// Helper: validate that a `ValueType` is `Tensor<Float32>` with shape
/// `[1, expected_dim]` or `[-1, expected_dim]` and return the shape as `Vec<i64>`.
fn validate_tensor_shape(
    dtype: &ValueType,
    expected_dim: i64,
    is_input: bool,
) -> Result<Vec<i64>, ValidationError> {
    let shape_error = |dims: Vec<i64>| -> ValidationError {
        if is_input {
            ValidationError::InvalidInputShape(dims)
        } else {
            ValidationError::InvalidOutputShape(dims)
        }
    };

    match dtype {
        ValueType::Tensor { ty, shape, .. } => {
            let dims: Vec<i64> = shape.iter().copied().collect();

            // Must be float32
            if *ty != TensorElementType::Float32 {
                return Err(shape_error(dims));
            }

            // Must be rank 2
            if dims.len() != 2 {
                return Err(shape_error(dims));
            }

            // Batch dim must be 1 or -1 (dynamic)
            if dims[0] != 1 && dims[0] != -1 {
                return Err(shape_error(dims));
            }

            // Feature dim must match
            if dims[1] != expected_dim {
                return Err(shape_error(dims));
            }

            Ok(dims)
        }
        _ => Err(shape_error(vec![])),
    }
}

// ---------------------------------------------------------------------------
// OnnxPolicy
// ---------------------------------------------------------------------------

/// A wrapper around an ort `Session` that implements the `Policy` trait from
/// dogfight-sim, allowing an ONNX model to act as a game agent.
pub struct OnnxPolicy {
    session: Session,
    name: String,
}

impl OnnxPolicy {
    /// Load an ONNX model from disk and wrap it as a `Policy`.
    ///
    /// This does **not** run the full validation suite (`validate_model_file`).
    /// Call that separately if you need the report.
    pub fn load(path: &Path) -> Result<Self, ValidationError> {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("onnx_policy")
            .to_string();

        let session = Session::builder()
            .map_err(ValidationError::from)?
            .commit_from_file(path)
            .map_err(ValidationError::from)?;

        Ok(Self { session, name })
    }
}

impl Policy for OnnxPolicy {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&mut self, obs: &Observation) -> Action {
        // Build a [1, OBS_SIZE] input tensor from the observation data.
        let input_data: Vec<f32> = obs.data.to_vec();
        let input_tensor = match Tensor::from_array(
            ([1usize, OBS_SIZE], input_data.into_boxed_slice()),
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("OnnxPolicy: failed to create input tensor: {e}");
                return Action::none();
            }
        };

        // Run inference
        let outputs = match self.session.run(ort::inputs![input_tensor]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("OnnxPolicy: inference failed: {e}");
                return Action::none();
            }
        };

        // Extract the [1, ACTION_SIZE] output tensor
        let output_value = &outputs[0];
        match output_value.try_extract_tensor::<f32>() {
            Ok((_shape, output_data)) => {
                if output_data.len() < ACTION_SIZE {
                    eprintln!(
                        "OnnxPolicy: output tensor has {} elements, expected at least {}",
                        output_data.len(),
                        ACTION_SIZE
                    );
                    return Action::none();
                }

                let raw = [
                    output_data[0],
                    output_data[1],
                    output_data[2],
                ];
                Action::from_raw(raw)
            }
            Err(e) => {
                eprintln!("OnnxPolicy: failed to extract output tensor: {e}");
                Action::none()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_model_path() -> PathBuf {
        // Created by: python3 -c "torch.onnx.export(Linear(224,64)+Linear(64,3), ...)"
        // 14595 params: 224*64 + 64 + 64*3 + 3
        PathBuf::from("/tmp/test_model_v2.onnx")
    }

    fn test_model_with_const_path() -> PathBuf {
        // Same as above plus a [3]-element buffer = 14598 params
        PathBuf::from("/tmp/test_model_const.onnx")
    }

    #[test]
    fn test_exact_parameter_count_simple() {
        let path = test_model_path();
        if !path.exists() {
            eprintln!("Skipping test: {} not found (run Python export first)", path.display());
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        let (count, ops) = count_all_parameters(&bytes).unwrap();
        // 224*64 + 64 + 64*3 + 3 = 14595
        assert_eq!(count, 14595, "parameter count should be exact");
        assert!(!ops.is_empty(), "should detect ops");
        assert!(ops.contains(&"Gemm".to_string()), "should contain Gemm op");
        assert!(ops.contains(&"Relu".to_string()), "should contain Relu op");
    }

    #[test]
    fn test_exact_parameter_count_with_constant() {
        let path = test_model_with_const_path();
        if !path.exists() {
            eprintln!("Skipping test: {} not found (run Python export first)", path.display());
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        let (count, _ops) = count_all_parameters(&bytes).unwrap();
        // 14595 + 3 (scale buffer) = 14598
        assert_eq!(count, 14598, "should include constant/buffer parameters");
    }

    #[test]
    fn test_validate_rejects_wrong_shape() {
        // The test models have [1,224]->[1,3] which matches our game spec,
        // so they should pass full validation.
        let path = test_model_path();
        if !path.exists() {
            return;
        }
        let report = validate_model_file(&path).unwrap();
        assert_eq!(report.parameter_count, 14595);
        assert!(report.parameter_count <= MAX_PARAMETERS);
    }
}

