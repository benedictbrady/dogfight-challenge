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
use std::time::Duration;
use tower_http::cors::CorsLayer;

/// Available built-in policy names.
const MANUAL_POLICY: &str = "manual";
const AVAILABLE_POLICIES: &[&str] = &[
    "dogfighter",
    "chaser",
    "ace",
    "brawler",
    "do_nothing",
    "neural",
    MANUAL_POLICY,
];

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

/// Manual control input message sent by the client after match start.
#[derive(Debug, Deserialize)]
struct InputMessage {
    #[serde(rename = "type")]
    msg_type: String,
    player: usize,
    action: InputActionPayload,
}

/// Action payload for manual control.
#[derive(Debug, Deserialize)]
struct InputActionPayload {
    yaw_input: f32,
    throttle: f32,
    shoot: bool,
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

/// Resolve a policy by name. Returns a boxed `Policy` trait object.
///
/// Supports built-in names and ONNX model paths:
/// - `"neural"` loads `policy.onnx` from the current directory
/// - Any name ending in `.onnx` is loaded as a file path
pub fn resolve_policy(name: &str) -> Box<dyn Policy> {
    try_resolve_policy(name).unwrap_or_else(|| panic!("unknown policy: {name}"))
}

/// Try to resolve a policy by name, returning `None` for unknown names.
fn try_resolve_policy(name: &str) -> Option<Box<dyn Policy>> {
    match name {
        "do_nothing" => Some(Box::new(DoNothingPolicy)),
        "chaser" => Some(Box::new(ChaserPolicy::new())),
        "dogfighter" => Some(Box::new(DogfighterPolicy::new())),
        "ace" => Some(Box::new(AcePolicy::new())),
        "brawler" => Some(Box::new(BrawlerPolicy::new())),
        "neural" => load_onnx_policy(Path::new("policy.onnx")),
        path if path.ends_with(".onnx") => load_onnx_policy(Path::new(path)),
        _ => None,
    }
}

fn is_onnx_policy(name: &str) -> bool {
    name == "neural" || name.ends_with(".onnx")
}

fn is_manual_policy(name: &str) -> bool {
    name == MANUAL_POLICY
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

/// GET /api/policies -- returns available built-in policy names.
async fn get_policies() -> Json<Vec<&'static str>> {
    Json(AVAILABLE_POLICIES.to_vec())
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
        AVAILABLE_POLICIES.contains(&name) || name.ends_with(".onnx")
    }

    if !is_valid_policy(&req.p0) {
        let _ = send_error(&mut socket, &format!("unknown policy for p0: {}", req.p0)).await;
        return;
    }

    if !is_valid_policy(&req.p1) {
        let _ = send_error(&mut socket, &format!("unknown policy for p1: {}", req.p1)).await;
        return;
    }

    if is_manual_policy(&req.p0) || is_manual_policy(&req.p1) {
        if let Err(e) = run_realtime_match(&mut socket, req).await {
            eprintln!("realtime match error: {e}");
        }
        return;
    }

    // 3. Build match config and run the match on a blocking thread so that
    //    `Box<dyn Policy>` (which is not `Send`) never lives across an await.
    let p0_name = req.p0.clone();
    let p1_name = req.p1.clone();
    let seed = req.seed.unwrap_or(0);
    let randomize_spawns = req.randomize_spawns.unwrap_or(false);

    let replay = tokio::task::spawn_blocking(move || {
        let mut p0 = try_resolve_policy(&p0_name).unwrap();
        let mut p1 = try_resolve_policy(&p1_name).unwrap();

        // ONNX models were trained with action_repeat=10, so set control_period=10
        let p0_period = if is_onnx_policy(&p0_name) { 10 } else { 1 };
        let p1_period = if is_onnx_policy(&p1_name) { 10 } else { 1 };

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
        if send_frame(&mut socket, frame).await.is_err() {
            return; // client disconnected
        }
    }

    // 5. Send the result message.
    let _ = send_result(&mut socket, &replay.result).await;
}

/// Run a realtime match that accepts manual input for either player.
async fn run_realtime_match(socket: &mut WebSocket, req: MatchRequest) -> Result<(), axum::Error> {
    let seed = req.seed.unwrap_or(0);
    let randomize_spawns = req.randomize_spawns.unwrap_or(false);

    let manual_p0 = is_manual_policy(&req.p0);
    let manual_p1 = is_manual_policy(&req.p1);

    let mut p0_policy = if manual_p0 {
        None
    } else {
        Some(resolve_policy(&req.p0))
    };
    let mut p1_policy = if manual_p1 {
        None
    } else {
        Some(resolve_policy(&req.p1))
    };

    // ONNX models were trained with action_repeat=10, so set control_period=10.
    let p0_period = if !manual_p0 && is_onnx_policy(&req.p0) {
        10
    } else {
        1
    };
    let p1_period = if !manual_p1 && is_onnx_policy(&req.p1) {
        10
    } else {
        1
    };

    let mut action0 = if manual_p0 {
        Action {
            yaw_input: 0.0,
            throttle: 0.85,
            shoot: false,
        }
    } else {
        Action::none()
    };
    let mut action1 = if manual_p1 {
        Action {
            yaw_input: 0.0,
            throttle: 0.85,
            shoot: false,
        }
    } else {
        Action::none()
    };

    let mut state = SimState::new_with_seed(seed, randomize_spawns);
    send_frame(socket, &state.snapshot()).await?;

    let mut ticker = tokio::time::interval(Duration::from_micros(TICK_DURATION_US));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                if state.is_terminal() {
                    break;
                }

                if !manual_p0 && state.tick % p0_period == 0 {
                    if let Some(policy) = p0_policy.as_mut() {
                        action0 = policy.act(&state.observe(0));
                    }
                }
                if !manual_p1 && state.tick % p1_period == 0 {
                    if let Some(policy) = p1_policy.as_mut() {
                        action1 = policy.act(&state.observe(1));
                    }
                }

                state.step(&[action0, action1]);

                if state.tick % FRAME_INTERVAL == 0 || state.is_terminal() {
                    send_frame(socket, &state.snapshot()).await?;
                }

                if state.is_terminal() {
                    break;
                }
            }
            inbound = socket.recv() => {
                match inbound {
                    Some(Ok(Message::Text(text))) => {
                        apply_manual_input(&text, manual_p0, manual_p1, &mut action0, &mut action1);
                    }
                    Some(Ok(Message::Close(_))) | None => return Ok(()),
                    Some(Ok(_)) => {}
                    Some(Err(e)) => return Err(e),
                }
            }
        }
    }

    let (outcome, reason) = state.outcome();
    let result = MatchResult {
        outcome,
        reason,
        final_tick: state.tick,
        stats: state.stats,
    };

    send_result(socket, &result).await
}

fn apply_manual_input(
    text: &str,
    manual_p0: bool,
    manual_p1: bool,
    action0: &mut Action,
    action1: &mut Action,
) {
    let msg = match serde_json::from_str::<InputMessage>(text) {
        Ok(m) => m,
        Err(_) => return,
    };
    if msg.msg_type != "input" {
        return;
    }

    let action = Action {
        yaw_input: msg.action.yaw_input.clamp(-1.0, 1.0),
        throttle: msg.action.throttle.clamp(0.0, 1.0),
        shoot: msg.action.shoot,
    };

    match msg.player {
        0 if manual_p0 => *action0 = action,
        1 if manual_p1 => *action1 = action,
        _ => {}
    }
}

async fn send_frame(socket: &mut WebSocket, frame: &ReplayFrame) -> Result<(), axum::Error> {
    let msg = FrameMessage {
        msg_type: "frame",
        tick: frame.tick,
        fighters: frame.fighters.to_vec(),
        bullets: frame.bullets.clone(),
    };
    let json = match serde_json::to_string(&msg) {
        Ok(j) => j,
        Err(_) => return Ok(()),
    };
    socket.send(Message::Text(json.into())).await
}

async fn send_result(socket: &mut WebSocket, result: &MatchResult) -> Result<(), axum::Error> {
    let result_msg = ResultMessage {
        msg_type: "result",
        outcome: result.outcome,
        reason: result.reason,
        final_tick: result.final_tick,
        stats: StatsPayload {
            p0_hp: result.stats.p0_hp,
            p1_hp: result.stats.p1_hp,
            p0_hits: result.stats.p0_hits,
            p1_hits: result.stats.p1_hits,
            p0_shots: result.stats.p0_shots,
            p1_shots: result.stats.p1_shots,
        },
    };

    let json = match serde_json::to_string(&result_msg) {
        Ok(j) => j,
        Err(_) => return Ok(()),
    };
    socket.send(Message::Text(json.into())).await
}

/// Send a JSON error message over the WebSocket and close.
async fn send_error(socket: &mut WebSocket, error: &str) -> Result<(), axum::Error> {
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
