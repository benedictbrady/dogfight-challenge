use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use dogfight_shared::*;
use dogfight_sim::opponents::{ChaserPolicy, DogfighterPolicy};
use dogfight_sim::{run_match, DoNothingPolicy, Policy};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

/// Available built-in policy names.
const AVAILABLE_POLICIES: &[&str] = &["dogfighter", "chaser", "do_nothing"];

// ---------------------------------------------------------------------------
// Serde types for WebSocket messages
// ---------------------------------------------------------------------------

/// Configuration message sent by the client when connecting to /api/match.
#[derive(Debug, Deserialize)]
struct MatchRequest {
    p0: String,
    p1: String,
    seed: Option<u64>,
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
/// Supported names: `"do_nothing"`, `"chaser"`, `"dogfighter"`.
pub fn resolve_policy(name: &str) -> Box<dyn Policy> {
    match name {
        "do_nothing" => Box::new(DoNothingPolicy),
        "chaser" => Box::new(ChaserPolicy),
        "dogfighter" => Box::new(DogfighterPolicy::new()),
        other => panic!("unknown policy: {other}"),
    }
}

/// Try to resolve a policy by name, returning `None` for unknown names.
fn try_resolve_policy(name: &str) -> Option<Box<dyn Policy>> {
    match name {
        "do_nothing" => Some(Box::new(DoNothingPolicy)),
        "chaser" => Some(Box::new(ChaserPolicy)),
        "dogfighter" => Some(Box::new(DogfighterPolicy::new())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// HTTP / WebSocket handlers
// ---------------------------------------------------------------------------

/// GET /api/policies -- returns available built-in policy names.
async fn get_policies() -> Json<Vec<&'static str>> {
    Json(AVAILABLE_POLICIES.to_vec())
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
    if !AVAILABLE_POLICIES.contains(&req.p0.as_str()) {
        let _ = send_error(
            &mut socket,
            &format!("unknown policy for p0: {}", req.p0),
        )
        .await;
        return;
    }

    if !AVAILABLE_POLICIES.contains(&req.p1.as_str()) {
        let _ = send_error(
            &mut socket,
            &format!("unknown policy for p1: {}", req.p1),
        )
        .await;
        return;
    }

    // 3. Build match config and run the match on a blocking thread so that
    //    `Box<dyn Policy>` (which is not `Send`) never lives across an await.
    let p0_name = req.p0.clone();
    let p1_name = req.p1.clone();
    let seed = req.seed.unwrap_or(0);

    let replay = tokio::task::spawn_blocking(move || {
        let mut p0 = try_resolve_policy(&p0_name).unwrap();
        let mut p1 = try_resolve_policy(&p1_name).unwrap();
        let match_config = MatchConfig {
            seed,
            p0_name,
            p1_name,
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
