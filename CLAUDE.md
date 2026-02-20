# Dogfight Challenge

2D side-view dogfight platform. Rust backend (physics + WebSocket server) + Next.js frontend (Three.js visualization).

## Quick Start

```bash
# Build everything
make build

# Run a match (CLI)
make run P0=chaser P1=dogfighter SEED=42

# Interactive viz (two terminals)
make serve          # Terminal 1: backend on :3001
make viz            # Terminal 2: frontend on :3000

# Run tests
make test

# Tournament
make tournament POLICIES=chaser,dogfighter,do_nothing ROUNDS=5
```

## Workspace Structure

```
crates/
  shared/     # Types, constants, observation layout
  sim/        # Physics engine, Policy trait, built-in opponents
  validator/  # ONNX model validation (ort v2.0.0-rc.11)
  server/     # Axum WebSocket server
  cli/        # CLI entry point (run, serve, tournament, validate)
viz/          # Next.js + React Three Fiber visualization
```

## Game Rules

- **Arena**: X: [-500, 500] horizontal, Y: [0, 600] altitude
- **Physics**: 120Hz tick rate. Gravity affects speed via `speed += (-GRAVITY * yaw.sin()) * DT` — climbing costs speed, diving gains it
- **Turn rate**: Speed-dependent. Slow (40 m/s) = 4.0 rad/s, Fast (250 m/s) = 0.8 rad/s
- **HP**: 5 hit points per fighter. 0 HP = eliminated
- **Rear-aspect armor**: Bullets from within 45° behind target glance off (no damage). Forces crossing/beam attacks.
- **Bullets**: 400 m/s, 0.5s lifetime (~200m range), 0.75s cooldown between shots
- **Match**: 90 seconds max (10800 ticks). Win by elimination or HP advantage at timeout
- **Initial positions**: P0 at (-200, 300), P1 at (200, 300), both at MIN_SPEED

## Key Types

**Action** — `[yaw_input, throttle, shoot]` (3 floats)
- yaw_input: [-1, 1], throttle: [0, 1], shoot: > 0 fires

**Observation** — 46 floats fed to policies:
- [0:6] Self state (speed, cos/sin yaw, hp, cooldown, altitude)
- [6:13] Opponent relative state (rel_pos, speed, heading, hp, distance)
- [13:45] Bullet slots (8 slots x 4: rel_x, rel_y, is_friendly, angle)
- [45] Ticks remaining normalized

**Policy trait** — `Send`. Use `spawn_blocking` in async contexts for `Box<dyn Policy>`.

## Built-in Opponents

`do_nothing` < `dogfighter` < `chaser` < `ace` < `brawler` (verified by tests + tournament)

- **DoNothing**: No inputs. Falls from sky.
- **Chaser**: Pressure pursuit with yo-yo maneuvers, bullet evasion, crossing aim when behind
- **Dogfighter**: Adaptive mode-switching (attack/defend/energy/disengage), crossing aim when behind
- **Ace**: Defensive energy fighter with perpendicular break turns, high-altitude preference
- **Brawler**: Close-range turn fighter, overshoot baiting, naturally excels with rear-aspect armor

## WebSocket Protocol

1. Client opens `ws://localhost:3001/api/match`
2. Client sends: `{ "p0": "chaser", "p1": "dogfighter", "seed": 42 }`
3. Server streams: `{ "type": "frame", "tick": N, "fighters": [...], "bullets": [...] }`
4. Server sends: `{ "type": "result", "outcome": "Player0Win", "reason": "Elimination", ... }`

HTTP: `GET /api/policies` returns available policy names.

## Viz Components

- `Scene.tsx` — Canvas, camera (orthographic, centered at arena midpoint), OrbitControls
- `Arena.tsx` — Sky gradient, clouds, mountains, ground layers, arena boundaries
- `Fighter.tsx` — Biplane silhouette, trail lines, labels
- `Controls.tsx` — Play/pause, frame scrubber, speed selector
- `MatchSetup.tsx` — Policy dropdowns, seed, Scramble button, server status
- `useMatch.ts` — WebSocket hook, frame accumulation, playback timer

## Gotchas

- `ort` crate must be pinned to `2.0.0-rc.11` (pre-release, needs explicit version)
- `glam` needs `serde` feature enabled for Vec2 serialization
- serde can't derive for `[f32; 46]` (arrays > 32) — uses custom serialize/deserialize
- Frame streaming is 30fps (every 4th physics tick)
- CORS is permissive on the server for local dev
