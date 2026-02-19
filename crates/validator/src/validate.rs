use std::path::Path;
use std::time::Instant;

use dogfight_shared::{
    Action, Observation, OBS_SIZE, ACTION_SIZE, MAX_MODEL_SIZE_BYTES, MAX_PARAMETERS,
    CALIBRATION_WARMUP, CALIBRATION_RUNS, TICK_DURATION_US,
};
use dogfight_sim::Policy;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Model file too large: {0} bytes (max {1})")]
    FileTooLarge(usize, usize),
    #[error("Disallowed op: {0}")]
    DisallowedOp(String),
    #[error("Invalid input shape: expected [1, 46] or [N, 46], got {0:?}")]
    InvalidInputShape(Vec<i64>),
    #[error("Invalid output shape: expected [1, 3] or [N, 3], got {0:?}")]
    InvalidOutputShape(Vec<i64>),
    #[error("Too many parameters: {0} (max {1})")]
    TooManyParameters(usize, usize),
    #[error("Inference too slow: {0}ms (max {1}ms)")]
    InferenceTooSlow(u64, u64),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ONNX runtime error: {0}")]
    OrtError(String),
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
// validate_model_file
// ---------------------------------------------------------------------------

/// Validate an ONNX model file at the given path.
///
/// Checks performed:
/// 1. File size <= MAX_MODEL_SIZE_BYTES (10 MB)
/// 2. Model can be loaded by ONNX Runtime (valid protobuf, supported ops)
/// 3. Input shape is `[1, 46]` or `[N, 46]` (dynamic batch with -1) float32
/// 4. Output shape is `[1, 3]` or `[N, 3]` (dynamic batch with -1) float32
/// 5. Estimated parameter count <= MAX_PARAMETERS
pub fn validate_model_file(path: &Path) -> Result<ValidationReport, ValidationError> {
    // 1. File size check
    let metadata = std::fs::metadata(path)?;
    let file_size_bytes = metadata.len() as usize;
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

    // 5. Parameter count estimation based on file size heuristic.
    //    ONNX files store weights as raw float32 (4 bytes each) plus some
    //    protobuf overhead.  We use a conservative estimate: every 4 bytes in
    //    the file could be a parameter.
    // TODO: When ort exposes the model graph proto, walk initializer tensors
    //       for an exact count instead of this heuristic.
    let parameter_count = file_size_bytes / 4;
    if parameter_count > MAX_PARAMETERS {
        return Err(ValidationError::TooManyParameters(
            parameter_count,
            MAX_PARAMETERS,
        ));
    }

    // TODO: Op allowlist checking is deferred because ort v2 does not expose
    //       the ONNX graph proto (node list) directly.  The session creation
    //       above will still reject truly unsupported ops at the runtime level.
    let ops_used = Vec::new();

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
// Calibration
// ---------------------------------------------------------------------------

/// Run a series of dummy inferences to measure the average inference latency,
/// then return a *control period* (number of simulation ticks per decision).
///
/// control_period = ceil(mean_inference_us / TICK_DURATION_US)
///
/// A control period of 1 means the policy can run every tick.
pub fn calibrate_inference(policy: &mut OnnxPolicy) -> u32 {
    let dummy_obs = Observation {
        data: [0.0f32; OBS_SIZE],
    };

    // Warmup
    for _ in 0..CALIBRATION_WARMUP {
        let _ = policy.act(&dummy_obs);
    }

    // Timed runs
    let start = Instant::now();
    for _ in 0..CALIBRATION_RUNS {
        let _ = policy.act(&dummy_obs);
    }
    let elapsed_us = start.elapsed().as_micros() as u64;
    let mean_us = elapsed_us / CALIBRATION_RUNS as u64;

    let period = mean_us.div_ceil(TICK_DURATION_US).max(1);
    period as u32
}
