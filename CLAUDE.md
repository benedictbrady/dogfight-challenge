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

**Observation** — 224 floats (4 × 56-float frames stacked):
- Single frame (56 floats):
  - [0:8] Self state (speed, cos/sin yaw, hp, cooldown, altitude, x_pos, energy)
  - [8:19] Opponent state (rel_pos, speed, heading, hp, distance, closure_rate, ang_vel, energy, angle_off_tail)
  - [19:51] Bullet slots (8 × 4: rel_x, rel_y, is_friendly, angle)
  - [51:55] Relative geometry (angle_off_nose, opp_angle_off_nose, rel_vel_x, rel_vel_y)
  - [55] Ticks remaining normalized
- Stacked: `[current(56), prev_1(56), prev_2(56), prev_3(56)]`
- See `RULES.md` for full layout documentation

**Competition ONNX spec**: `[1, 224] → [1, 3]`, stateless, 12Hz decisions. See `RULES.md`.

**Policy trait** — NOT `Send`. Use `spawn_blocking` in async contexts.

## Built-in Opponents

`do_nothing` < `chaser` < `ace` < `dogfighter` < `brawler` (verified by tests + tournament)

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

## Modal Training

See `.claude/projects/.../memory/modal-guide.md` for full details. Key rules:
- **Always use `modal run --detach`** for real training runs — without it, closing terminal kills the container
- **Modal CLI is slow** (~2 min per invocation due to image rebuild) — never run interactively or in parallel
- Check Slack or Modal dashboard for run status, not CLI commands
- Training run history tracked in `.claude/projects/.../memory/modal-training-runs.md`

## Gotchas

- `ort` crate must be pinned to `2.0.0-rc.11` (pre-release, needs explicit version)
- `glam` needs `serde` feature enabled for Vec2 serialization
- serde can't derive for `[f32; 224]` (arrays > 32) — uses custom serialize/deserialize
- Frame streaming is 30fps (every 4th physics tick)
- CORS is permissive on the server for local dev
