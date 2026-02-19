// Tick rate
pub const TICK_RATE: u32 = 120;
pub const DT: f32 = 1.0 / TICK_RATE as f32;
pub const TICK_DURATION_US: u64 = 8333;

// Match
pub const MATCH_DURATION_SECS: u32 = 90;
pub const MAX_TICKS: u32 = TICK_RATE * MATCH_DURATION_SECS; // 7200

// Arena (side-view: X=horizontal, Y=altitude)
pub const ARENA_RADIUS: f32 = 500.0;
pub const ARENA_DIAMETER: f32 = ARENA_RADIUS * 2.0;
pub const BOUNDARY_START_HORIZ: f32 = 450.0;
pub const BOUNDARY_FORCE: f32 = 300.0;
pub const GRAVITY: f32 = 80.0;
pub const MAX_ALTITUDE: f32 = 600.0;
pub const ALT_BOUNDARY_LOW: f32 = 50.0;
pub const ALT_BOUNDARY_HIGH: f32 = 550.0;

// Fighter
pub const FIGHTER_RADIUS: f32 = 8.0;
pub const MAX_SPEED: f32 = 250.0;
pub const MIN_SPEED: f32 = 20.0;
pub const MAX_THRUST: f32 = 180.0;
pub const DRAG_COEFF: f32 = 0.9;
pub const MAX_TURN_RATE: f32 = 4.0;
pub const MIN_TURN_RATE: f32 = 0.8;
pub const TURN_BLEED_COEFF: f32 = 0.25;
pub const MAX_HP: u8 = 5;

// Stall mechanic
pub const STALL_SPEED: f32 = 30.0;
pub const STALL_RECOVERY_TICKS: u32 = 36; // ~0.3s at 120Hz
pub const STALL_NOSE_DOWN_RATE: f32 = 2.5; // rad/s

// Damage degradation
pub const DAMAGE_SPEED_PENALTY: f32 = 0.03; // 3% max speed per HP lost
pub const DAMAGE_TURN_PENALTY: f32 = 0.02;  // 2% turn rate per HP lost

// Bullets
pub const BULLET_SPEED: f32 = 400.0;
pub const BULLET_LIFETIME_TICKS: u32 = 60; // 0.5s at 120Hz
pub const BULLET_RADIUS: f32 = 3.0;
pub const GUN_COOLDOWN_TICKS: u32 = 90; // 0.75s at 120Hz

// Rear-aspect armor: bullets from within this cone behind the target do no damage
pub const REAR_ASPECT_CONE: f32 = 0.785; // PI/4 = 45° half-angle
pub const GUN_COOLDOWN_SECS: f32 = GUN_COOLDOWN_TICKS as f32 / TICK_RATE as f32;

// Observation
pub const OBS_SIZE: usize = 46;
pub const ACTION_SIZE: usize = 3;
pub const MAX_BULLET_SLOTS: usize = 8;

// Config observation (opt-in, appended to base obs at PyO3 boundary)
pub const CONFIG_OBS_SIZE: usize = 13;

// Fixed normalization denominators for config observation.
// Each SimConfig parameter is divided by its denominator to produce [0, ~1] values.
pub const CONFIG_NORM_GRAVITY: f32 = 200.0;
pub const CONFIG_NORM_DRAG_COEFF: f32 = 2.0;
pub const CONFIG_NORM_TURN_BLEED_COEFF: f32 = 1.0;
pub const CONFIG_NORM_MAX_SPEED: f32 = 500.0;
pub const CONFIG_NORM_MIN_SPEED: f32 = 100.0;
pub const CONFIG_NORM_MAX_THRUST: f32 = 400.0;
pub const CONFIG_NORM_BULLET_SPEED: f32 = 800.0;
pub const CONFIG_NORM_GUN_COOLDOWN_TICKS: f32 = 240.0;
pub const CONFIG_NORM_BULLET_LIFETIME_TICKS: f32 = 180.0;
pub const CONFIG_NORM_MAX_HP: f32 = 10.0;
pub const CONFIG_NORM_MAX_TURN_RATE: f32 = 8.0;
pub const CONFIG_NORM_MIN_TURN_RATE: f32 = 4.0;
pub const CONFIG_NORM_REAR_ASPECT_CONE: f32 = std::f32::consts::PI;

// Scoring
pub const WIN_ELIMINATION_POINTS: u32 = 3;
pub const WIN_HP_POINTS: u32 = 2;
pub const DRAW_POINTS: u32 = 1;

// Frame streaming
pub const FRAME_INTERVAL: u32 = 4; // stream every 4th tick = 30fps

// ONNX validation
pub const MAX_MODEL_SIZE_BYTES: usize = 50 * 1024 * 1024; // 50 MB
pub const MAX_PARAMETERS: usize = 10_000_000;
pub const MAX_INFERENCE_TIME_MS: u64 = 1000;
pub const CALIBRATION_WARMUP: usize = 10;
pub const CALIBRATION_RUNS: usize = 100;
