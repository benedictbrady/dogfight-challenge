# Dogfight Challenge

A 2D side-view dogfight simulator with AI opponents. Pit scripted fighters against each other — or bring your own neural network policy.

## What is this?

A Rust-powered physics simulator where AI-controlled biplanes battle in a 2D arena. The platform includes five built-in opponents of increasing difficulty, a real-time visualization frontend, and an ONNX model interface so you can train and deploy your own fighter policy.

## Quick Start

```bash
# Build
make build

# Run a match in the terminal
make run P0=brawler P1=ace

# Or launch the GUI (two terminals)
make serve    # Terminal 1: starts backend on :3001
make viz      # Terminal 2: starts frontend on :3000
```

Open [http://localhost:3000](http://localhost:3000) to watch matches in the browser.

## Game Rules

| Rule | Detail |
|------|--------|
| **Arena** | 1000m wide (X: -500 to 500), 600m tall (Y: 0 to 600) |
| **Physics** | 120Hz tick rate. Gravity affects speed — climbing costs speed, diving gains it |
| **Turn rate** | Speed-dependent: 4.0 rad/s at 40 m/s, 0.8 rad/s at 250 m/s |
| **HP** | 5 hit points per fighter. 0 HP = eliminated |
| **Bullets** | 400 m/s, ~200m range, 0.75s cooldown between shots |
| **Rear-aspect armor** | Bullets from within 45° directly behind a target glance off (no damage) |
| **Match length** | 90 seconds max. Win by elimination or HP advantage at timeout |

## Built-in Opponents

From weakest to strongest:

| Policy | Style |
|--------|-------|
| `do_nothing` | No inputs. Falls from the sky. |
| `dogfighter` | Adaptive mode-switching: attack, defend, energy management, disengage |
| `chaser` | Pressure pursuit with yo-yo maneuvers and bullet evasion |
| `ace` | Defensive energy fighter with perpendicular break turns, prefers high altitude |
| `brawler` | Close-range turn fighter, baits overshoots, excels with rear-aspect armor |

## Neural Policy

The repo includes a baseline neural network (`policy.onnx`) that can beat all built-in opponents. Select "neural" in the GUI dropdown or run:

```bash
make run P0=neural P1=brawler
```

### Bring Your Own Model

Your ONNX model must satisfy:

| Constraint | Value |
|------------|-------|
| Input shape | `[1, 46]` float32 (observation vector) |
| Output shape | `[1, 3]` float32 (action vector) |
| Max file size | 10 MB |
| Max parameters | 2,000,000 |

Validate your model:
```bash
make validate MODEL=your_model.onnx
```

Run it directly:
```bash
make run P0=your_model.onnx P1=brawler
```

## Observation Space (46 floats)

| Index | Field | Normalization |
|-------|-------|---------------|
| 0 | Speed | / MAX_SPEED |
| 1-2 | cos(yaw), sin(yaw) | [-1, 1] |
| 3 | HP | / MAX_HP |
| 4 | Gun cooldown | / COOLDOWN_TICKS |
| 5 | Altitude | / MAX_ALTITUDE |
| 6-7 | Opponent relative X, Y | / ARENA_DIAMETER |
| 8 | Opponent speed | / MAX_SPEED |
| 9-10 | cos(opp_yaw), sin(opp_yaw) | [-1, 1] |
| 11 | Opponent HP | / MAX_HP |
| 12 | Distance to opponent | / ARENA_DIAMETER |
| 13-44 | 8 bullet slots × 4 (rel_x, rel_y, is_friendly, angle) | Normalized |
| 45 | Ticks remaining | / MAX_TICKS |

## Action Space (3 floats)

| Index | Field | Range |
|-------|-------|-------|
| 0 | Yaw input | [-1, 1] (left/right turn) |
| 1 | Throttle | [0, 1] |
| 2 | Shoot | > 0 fires |

## Architecture

```
crates/
  shared/      Types, constants, observation layout
  sim/         Physics engine, Policy trait, built-in opponents
  validator/   ONNX model validation (ort v2.0.0-rc.11)
  server/      Axum WebSocket server
  cli/         CLI: run, serve, tournament, validate, analyze

viz/           Next.js + React Three Fiber frontend
```

The sim crate is the core — it runs the physics loop and defines the `Policy` trait that all fighters implement. The server crate wraps it in a WebSocket API, and the viz frontend connects to stream match frames in real time.

## CLI Commands

```bash
# Single match
make run P0=chaser P1=ace SEED=42

# WebSocket server for the GUI
make serve PORT=3001

# Round-robin tournament
make tournament POLICIES=chaser,dogfighter,ace,brawler,do_nothing ROUNDS=10

# Validate an ONNX model
make validate MODEL=policy.onnx

# Analyze battle dynamics (circling, speed variance, engagement metrics)
make analyze ANALYZE_POLICIES=chaser,dogfighter,ace,brawler ANALYZE_SEEDS=5
```

## Tech Stack

- **Rust** — Physics simulation, WebSocket server, CLI
- **Axum** — Async HTTP/WebSocket framework
- **ort** — ONNX Runtime bindings for neural policy inference
- **Next.js** — Frontend framework
- **React Three Fiber** — 3D/2D rendering via Three.js
- **glam** — Vector math
