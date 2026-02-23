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
use dogfight_sim::{run_match, Policy, SimState};
use dogfight_validator::OnnxPolicy;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::LazyLock;
use tower_http::cors::CorsLayer;

/// Discover ONNX model names from a single directory.
fn discover_onnx_in(dir: &str) -> Vec<String> {
    let mut names = Vec::new();
    let dir_path = Path::new(dir);
    if let Ok(entries) = std::fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "onnx") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
    }
    names.sort();
    names
}

/// Scripted opponent names always available in the GUI.
const SCRIPTED_OPPONENTS: &[&str] = &["chaser", "dogfighter", "ace", "brawler"];

/// Structured policy lists for the GUI.
#[derive(Debug, Clone, Serialize)]
struct PolicyLists {
    user_models: Vec<String>,
    opponents: Vec<String>,
}

static POLICY_LISTS: LazyLock<PolicyLists> = LazyLock::new(|| {
    let user_models = discover_onnx_in("models");

    let mut opponents: Vec<String> = SCRIPTED_OPPONENTS.iter().map(|s| s.to_string()).collect();
    opponents.sort();

    PolicyLists {
        user_models,
        opponents,
    }
});

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
/// Priority: scripted Rust opponents > models/ ONNX > .onnx path.
fn try_resolve_policy(name: &str) -> Option<Box<dyn Policy>> {
    // 1. Scripted Rust opponents
    match name {
        "chaser" => return Some(Box::new(ChaserPolicy::new())),
        "dogfighter" => return Some(Box::new(DogfighterPolicy::new())),
        "ace" => return Some(Box::new(AcePolicy::new())),
        "brawler" => return Some(Box::new(BrawlerPolicy::new())),
        _ => {}
    }

    // 2. Check models/ directory (user-trained ONNX models)
    let model_path = Path::new("models").join(format!("{name}.onnx"));
    if model_path.exists() {
        return load_onnx_policy(&model_path);
    }

    // 3. Direct .onnx path
    if name.ends_with(".onnx") {
        return load_onnx_policy(Path::new(name));
    }

    None
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

/// GET /api/policies -- returns structured { user_models, opponents }.
async fn get_policies() -> Json<PolicyLists> {
    Json(POLICY_LISTS.clone())
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
        let lists = &*POLICY_LISTS;
        lists.user_models.iter().any(|p| p == name)
            || lists.opponents.iter().any(|p| p == name)
            || name.ends_with(".onnx")
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

        let match_config = MatchConfig {
            seed,
            p0_name,
            p1_name,
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
