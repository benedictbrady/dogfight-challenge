use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query,
    },
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use dogfight_shared::*;
use dogfight_sim::opponents::{AcePolicy, BrawlerPolicy, ChaserPolicy, DogfighterPolicy};
use dogfight_sim::{run_match, DoNothingPolicy, Policy, SimState};
use dogfight_validator::OnnxPolicy;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::LazyLock;
use tower_http::cors::CorsLayer;

/// Discover ONNX model names from baselines/ and models/ directories.
fn discover_onnx_policies() -> Vec<String> {
    let mut names = Vec::new();
    for dir in &["baselines", "models"] {
        let dir_path = Path::new(dir);
        if let Ok(entries) = std::fs::read_dir(dir_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "onnx") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        if !names.contains(&stem.to_string()) {
                            names.push(stem.to_string());
                        }
                    }
                }
            }
        }
    }
    names.sort();
    names
}

/// All available policy names shown in the GUI (ONNX models only).
/// Scripted Rust policies are still available internally via try_resolve_policy().
static ALL_POLICIES: LazyLock<Vec<String>> = LazyLock::new(|| discover_onnx_policies());

// ---------------------------------------------------------------------------
// Serde types for WebSocket messages
// ---------------------------------------------------------------------------

/// Configuration message sent by the client when connecting to /api/match.
#[derive(Debug, Deserialize)]
struct MatchRequest {
    p0: String,
    p1: String,
    seed: Option<u64>,
    randomize_spawns: Option<bool>,
}

/// A single frame streamed to the client.
#[derive(Debug, Serialize)]
struct FrameMessage {
    #[serde(rename = "type")]
    msg_type: &'static str,
    tick: u32,
    fighters: Vec<FighterSnapshot>,
    bullets: Vec<BulletSnapshot>,
}

/// Per-match statistics included in the result message.
#[derive(Debug, Serialize)]
struct StatsPayload {
    p0_hp: u8,
    p1_hp: u8,
    p0_hits: u32,
    p1_hits: u32,
    p0_shots: u32,
    p1_shots: u32,
}

/// Final result message sent after all frames.
#[derive(Debug, Serialize)]
struct ResultMessage {
    #[serde(rename = "type")]
    msg_type: &'static str,
    outcome: MatchOutcome,
    reason: MatchEndReason,
    final_tick: u32,
    stats: StatsPayload,
}

/// Error message sent to the client.
#[derive(Debug, Serialize)]
struct ErrorMessage {
    #[serde(rename = "type")]
    msg_type: &'static str,
    error: String,
}

// ---------------------------------------------------------------------------
// Policy resolution
// ---------------------------------------------------------------------------

/// Resolve a policy by name, returning `None` for unknown names.
///
/// Priority: baselines/ ONNX > models/ ONNX > .onnx path > scripted Rust fallback.
fn try_resolve_policy(name: &str) -> Option<Box<dyn Policy>> {
    // 1. Check baselines/ directory (BC-trained imitation models)
    let baseline_path = Path::new("baselines").join(format!("{name}.onnx"));
    if baseline_path.exists() {
        return load_onnx_policy(&baseline_path);
    }

    // 2. Check models/ directory (RL-trained models)
    let model_path = Path::new("models").join(format!("{name}.onnx"));
    if model_path.exists() {
        return load_onnx_policy(&model_path);
    }

    // 3. Direct .onnx path
    if name.ends_with(".onnx") {
        return load_onnx_policy(Path::new(name));
    }

    // 4. Scripted Rust fallback (used by CLI/tests, not shown in GUI)
    match name {
        "do_nothing" => Some(Box::new(DoNothingPolicy)),
        "chaser" => Some(Box::new(ChaserPolicy::new())),
        "dogfighter" => Some(Box::new(DogfighterPolicy::new())),
        "ace" => Some(Box::new(AcePolicy::new())),
        "brawler" => Some(Box::new(BrawlerPolicy::new())),
        _ => None,
    }
}

/// All policies shown in the GUI are ONNX — they all run at 12Hz.
fn is_onnx_policy(name: &str) -> bool {
    if name.ends_with(".onnx") {
        return true;
    }
    Path::new("baselines").join(format!("{name}.onnx")).exists()
        || Path::new("models").join(format!("{name}.onnx")).exists()
}

fn load_onnx_policy(path: &Path) -> Option<Box<dyn Policy>> {
    match OnnxPolicy::load(path) {
        Ok(p) => {
            println!("Loaded ONNX policy from {}", path.display());
            Some(Box::new(p))
        }
        Err(e) => {
            eprintln!("Failed to load ONNX policy from {}: {e}", path.display());
            None
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP / WebSocket handlers
// ---------------------------------------------------------------------------

/// GET /api/policies -- returns all available policy names (built-in + neural models).
async fn get_policies() -> Json<Vec<String>> {
    Json(ALL_POLICIES.clone())
}

/// Query params for GET /api/spawn.
#[derive(Debug, Deserialize)]
struct SpawnQuery {
    seed: Option<u64>,
}

/// GET /api/spawn -- returns starting positions for a given seed.
async fn get_spawn(Query(q): Query<SpawnQuery>) -> Json<serde_json::Value> {
    let seed = q.seed.unwrap_or(0);
    let state = SimState::new_with_seed(seed, true);
    let f = &state.fighters;
    Json(serde_json::json!({
        "fighters": [
            { "x": f[0].position.x, "y": f[0].position.y, "yaw": f[0].yaw, "speed": f[0].speed, "hp": f[0].hp, "alive": f[0].alive },
            { "x": f[1].position.x, "y": f[1].position.y, "yaw": f[1].yaw, "speed": f[1].speed, "hp": f[1].hp, "alive": f[1].alive },
        ]
    }))
}

/// GET /api/match -- WebSocket upgrade endpoint.
async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

/// Handle an individual WebSocket connection.
async fn handle_socket(mut socket: WebSocket) {
    // 1. Wait for the client's config message.
    let config_msg = match socket.recv().await {
        Some(Ok(Message::Text(text))) => text,
        Some(Ok(Message::Close(_))) | None => return,
        Some(Ok(_)) => {
            let _ = send_error(&mut socket, "expected a JSON text message").await;
            return;
        }
        Some(Err(_)) => return,
    };

    let req: MatchRequest = match serde_json::from_str(&config_msg) {
        Ok(r) => r,
        Err(e) => {
            let _ = send_error(&mut socket, &format!("invalid config JSON: {e}")).await;
            return;
        }
    };

    // 2. Validate policy names before resolving (to send error over WS).
    fn is_valid_policy(name: &str) -> bool {
        ALL_POLICIES.iter().any(|p| p == name) || name.ends_with(".onnx")
    }

    for (label, name) in [("p0", &req.p0), ("p1", &req.p1)] {
        if !is_valid_policy(name) {
            let _ = send_error(
                &mut socket,
                &format!("unknown policy for {label}: {name}"),
            )
            .await;
            return;
        }
    }

    // 3. Build match config and run the match on a blocking thread so that
    //    `Box<dyn Policy>` (which is not `Send`) never lives across an await.
    let seed = req.seed.unwrap_or(0);
    let randomize_spawns = req.randomize_spawns.unwrap_or(false);
    let p0_name = req.p0;
    let p1_name = req.p1;

    let replay = tokio::task::spawn_blocking(move || {
        let mut p0 = try_resolve_policy(&p0_name).expect("policy already validated");
        let mut p1 = try_resolve_policy(&p1_name).expect("policy already validated");

        // ONNX models run at 12Hz (CONTROL_PERIOD ticks per decision)
        let p0_period = if is_onnx_policy(&p0_name) { CONTROL_PERIOD } else { 1 };
        let p1_period = if is_onnx_policy(&p1_name) { CONTROL_PERIOD } else { 1 };

        let match_config = MatchConfig {
            seed,
            p0_name,
            p1_name,
            p0_control_period: p0_period,
            p1_control_period: p1_period,
            randomize_spawns,
            ..Default::default()
        };
        run_match(&match_config, p0.as_mut(), p1.as_mut())
    })
    .await
    .expect("match task panicked");

    // 4. Stream each frame.
    for frame in &replay.frames {
        let msg = FrameMessage {
            msg_type: "frame",
            tick: frame.tick,
            fighters: frame.fighters.to_vec(),
            bullets: frame.bullets.clone(),
        };
        let json = match serde_json::to_string(&msg) {
            Ok(j) => j,
            Err(_) => continue,
        };
        if socket.send(Message::Text(json.into())).await.is_err() {
            return; // client disconnected
        }
    }

    // 5. Send the result message.
    let result_msg = ResultMessage {
        msg_type: "result",
        outcome: replay.result.outcome,
        reason: replay.result.reason,
        final_tick: replay.result.final_tick,
        stats: StatsPayload {
            p0_hp: replay.result.stats.p0_hp,
            p1_hp: replay.result.stats.p1_hp,
            p0_hits: replay.result.stats.p0_hits,
            p1_hits: replay.result.stats.p1_hits,
            p0_shots: replay.result.stats.p0_shots,
            p1_shots: replay.result.stats.p1_shots,
        },
    };

    let json = match serde_json::to_string(&result_msg) {
        Ok(j) => j,
        Err(_) => return,
    };
    let _ = socket.send(Message::Text(json.into())).await;
}

/// Send a JSON error message over the WebSocket and close.
async fn send_error(
    socket: &mut WebSocket,
    error: &str,
) -> Result<(), axum::Error> {
    let msg = ErrorMessage {
        msg_type: "error",
        error: error.to_string(),
    };
    let json = serde_json::to_string(&msg).unwrap_or_default();
    socket.send(Message::Text(json.into())).await
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

/// Build the axum `Router`.
pub fn app() -> Router {
    Router::new()
        .route("/api/policies", get(get_policies))
        .route("/api/spawn", get(get_spawn))
        .route("/api/match", get(ws_handler))
        .layer(CorsLayer::permissive())
}

/// Start the server on the given port.
pub async fn run_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = app();
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    println!("dogfight server listening on port {port}");
    axum::serve(listener, app).await?;
    Ok(())
}
