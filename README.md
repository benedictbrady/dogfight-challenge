# Dogfight Challenge

A 2D side-view dogfight arena where AI pilots battle for air superiority. Build a stateless ONNX model, drop it in, and see how it fares against four scripted opponents — or against other players' models in a round-robin tournament.

**Rust physics engine (120Hz) | Next.js + Three.js visualization | ONNX model interface**

---

## How It Works

```
                     ┌─────────────┐
                     │  Your Model  │
                     │  (ONNX file) │
                     └──────┬──────┘
                            │
              224 floats in  │  3 floats out
            (4 stacked obs)  │  (yaw, throttle, shoot)
                            ▼
┌──────────────────────────────────────────────┐
│              Rust Physics Engine              │
│                                              │
│  120Hz tick rate · gravity · rear-aspect armor│
│  bullet physics · stall mechanics · damage    │
│                                              │
│  Scripted opponents: dogfighter, chaser,      │
│  ace, brawler                                 │
└──────────────────┬───────────────────────────┘
                   │
                   │ WebSocket (30fps frames)
                   ▼
┌──────────────────────────────────────────────┐
│          Next.js + Three.js Frontend          │
│                                              │
│  Real-time match visualization, playback      │
│  controls, policy selection, seed picker      │
└──────────────────────────────────────────────┘
```

The sim runs the full match server-side, streams frames over WebSocket, and the browser renders them. Your model makes decisions at 12Hz (every 10 physics ticks) — the sim handles the rest.

---

## Quick Start

```bash
# Build
make build

# Run a match in the terminal
make run P0=brawler P1=ace

# Watch in the browser (two terminals)
make serve    # Terminal 1 — backend on :3001
make viz      # Terminal 2 — frontend on :3000
```

Open [http://localhost:3000](http://localhost:3000) to watch matches live.

---

## The Challenge

Submit a **stateless ONNX model** that controls a fighter. The sim provides all context via the observation vector — no hidden state needed.

### Model Contract

| | |
|---|---|
| **Input** | `float32[1, 224]` — 4 stacked observation frames |
| **Output** | `float32[1, 3]` — `[yaw_input, throttle, shoot]` |
| **Decision rate** | 12 Hz (every 10 physics ticks) |
| **Max file size** | 10 MB |
| **Max parameters** | 2,000,000 |

Output clamping: yaw `[-1, 1]`, throttle `[0, 1]`, shoot fires if `> 0`.

### Run Your Model

```bash
# Validate shape and size constraints
make validate MODEL=your_model.onnx

# Battle against the strongest built-in opponent
make run P0=your_model.onnx P1=brawler

# Or place it in baselines/ and use by name
cp your_model.onnx baselines/
make run P0=your_model P1=ace
```

Full rules and observation layout in [`RULES.md`](RULES.md).

---

## Game Rules

Two fighters spawn in a 2D arena and have 90 seconds to fight.

| | |
|---|---|
| **Arena** | 1000m wide, 600m tall |
| **Boundaries** | Horizontal edges wrap around. Ground (y &le; 5m) is fatal. Ceiling zone (above 550m) drains speed; hard cap at 600m |
| **Physics** | 120Hz. Gravity-based energy model — climbing costs speed, diving gains it |
| **HP** | 5 per fighter. 0 = eliminated |
| **Bullets** | 400 m/s, ~200m range, 0.75s cooldown |
| **Rear-aspect armor** | Bullets from within 45° behind a target glance off — no damage |
| **Stall** | Below 30 m/s, the fighter stalls: nose drops, no control or shooting until speed recovers |
| **Damage** | Each HP lost reduces max speed by 3% and turn rate by 2% |
| **Win** | Eliminate opponent, or have more HP when time runs out |

### Scoring (Tournament)

| Outcome | Points |
|---------|--------|
| Win by elimination | 3 |
| Win by HP at timeout | 2 |
| Draw | 1 |
| Loss | 0 |

---

## Observation Space (224 floats)

4 stacked frames of 56 floats each: `[current, prev_1, prev_2, prev_3]`

Each 56-float frame contains:

```
Self state [0..8)
  speed, cos(yaw), sin(yaw), hp, gun_cooldown, altitude, x_position, energy

Opponent state [8..19)
  rel_x, rel_y, speed, cos(yaw), sin(yaw), hp, distance,
  closure_rate, angular_velocity, energy, angle_off_tail

Bullets [19..51) — 8 nearest bullets × 4 floats
  rel_x, rel_y, is_friendly, angle

Relative geometry [51..55)
  angle_off_nose, opp_angle_off_nose, rel_vel_x, rel_vel_y

Meta [55]
  ticks_remaining
```

All values normalized to roughly [-1, 1]. See [`RULES.md`](RULES.md) for exact normalization constants.

---

## Built-in Opponents

From weakest to strongest:

| Policy | Style | Key trait |
|--------|-------|-----------|
| `dogfighter` | Adaptive | Mode-switches between attack, defend, energy management |
| `chaser` | Aggressive | Relentless pursuit with yo-yo maneuvers |
| `ace` | Defensive | Energy fighting, perpendicular breaks, altitude advantage |
| `brawler` | Close-range | Turn fighting, overshoot baiting, exploits rear-aspect armor |

### Baseline ONNX Models

Pre-trained baselines (behavioral cloning from scripted policies) are in `baselines/`. These are approximate ONNX copies of the scripted opponents, useful as reference models and for GUI playback:

```bash
make run P0=chaser P1=brawler           # scripted vs scripted
make run P0=baselines/ace.onnx P1=ace   # baseline ONNX vs scripted
```

---

## CLI Commands

```bash
make build                                    # Build everything
make test                                     # Run all tests
make run P0=chaser P1=ace SEED=42             # Single match
make serve                                    # WebSocket server (:3001)
make viz                                      # Frontend dev server (:3000)
make tournament POLICIES=chaser,ace,brawler ROUNDS=10
make validate MODEL=your_model.onnx
make analyze                                  # Battle dynamics analysis
```

---

## Architecture

```
crates/
  shared/      Constants, types, observation layout
  sim/         Physics engine, Policy trait, 4 scripted opponents
  validator/   ONNX model validation (shape, size, speed)
  server/      Axum WebSocket server — streams match frames
  cli/         CLI: run, serve, tournament, validate, analyze

baselines/     Pre-trained ONNX baseline models
viz/           Next.js + React Three Fiber frontend
```

**Tech:** Rust, Axum, ort (ONNX Runtime), Next.js, React Three Fiber, glam
