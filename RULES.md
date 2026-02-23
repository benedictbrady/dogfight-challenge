# Competition Rules

## Model Specification

Participants submit a **stateless ONNX model** that controls a fighter in a 2D side-view dogfight.

| Property | Value |
|----------|-------|
| Input | `float32[1, 224]` — stacked observation |
| Output | `float32[1, 3]` — `[yaw_input, throttle, shoot]` |
| Max file size | 10 MB |
| Max parameters | 250,000 |
| Decision rate | 12 Hz (every 10 physics ticks at 120 Hz) |

### Output Clamping

The sim clamps model outputs before applying them:
- `yaw_input`: clamped to `[-1, 1]` — turn rate scaling
- `throttle`: clamped to `[0, 1]` — engine power
- `shoot`: fires if `> 0` (binary threshold)

### Stateless Requirement

Models must be **stateless** — all temporal context is provided by the sim via frame stacking. No RNN/LSTM hidden states are permitted.

## Observation Layout (224 floats)

The observation consists of 4 stacked frames of 56 floats each:

```
[current_frame(56), prev_frame_1(56), prev_frame_2(56), prev_frame_3(56)]
```

### Single Frame Layout (56 floats)

```
Self state [0..8):
  [0]  speed / MAX_SPEED                     (250.0)
  [1]  cos(yaw)
  [2]  sin(yaw)
  [3]  hp / MAX_HP                            (5)
  [4]  gun_cooldown / GUN_COOLDOWN_TICKS      (90)
  [5]  altitude / MAX_ALTITUDE                (600.0)
  [6]  x_position / ARENA_RADIUS              (500.0)
  [7]  energy (v² + 2gh) / MAX_ENERGY

Opponent state [8..19):
  [8]  rel_x / ARENA_DIAMETER                 (1000.0)
  [9]  rel_y / ARENA_DIAMETER
  [10] speed / MAX_SPEED
  [11] cos(yaw)
  [12] sin(yaw)
  [13] hp / MAX_HP
  [14] distance / ARENA_DIAMETER
  [15] closure_rate / MAX_SPEED
  [16] angular_velocity / MAX_TURN_RATE       (4.0 rad/s)
  [17] energy (v² + 2gh) / MAX_ENERGY
  [18] angle_off_tail / PI

Bullets [19..51) — 8 slots × 4 floats:
  [19+s*4]   rel_x / ARENA_DIAMETER
  [19+s*4+1] rel_y / ARENA_DIAMETER
  [19+s*4+2] is_friendly (0 = enemy, 1 = own)
  [19+s*4+3] angle / PI

Relative geometry [51..55):
  [51] angle_off_nose / PI
  [52] opp_angle_off_nose / PI
  [53] rel_vel_x / MAX_SPEED
  [54] rel_vel_y / MAX_SPEED

Meta [55]:
  [55] ticks_remaining / MAX_TICKS
```

## Game Rules

- **Arena**: 1000m wide (X: -500 to 500), 600m tall (Y: 0 to 600)
- **Horizontal boundaries**: wrap around (flying off the right edge puts you on the left)
- **Ground**: hitting the ground (Y &le; 5m) is fatal — instant elimination
- **Ceiling**: speed drains progressively above 550m altitude; hard cap at 600m
- **Physics**: 120 Hz, gravity = 130 m/s². Climbing costs speed, diving gains it
- **HP**: 5 per fighter. 0 = eliminated
- **Rear-aspect armor**: bullets within 45° behind glance off — no damage
- **Bullets**: 400 m/s, 0.5s lifetime (~200m range), 0.75s cooldown
- **Stall**: below 30 m/s, nose drops toward ground, no control or shooting until speed exceeds 40 m/s
- **Damage degradation**: each HP lost reduces max speed by 3% and turn rate by 2%
- **Match**: 90 seconds (10800 ticks)
- **Win**: elimination or HP advantage at timeout

## Evaluation

Round-robin tournament: Win by elimination = 3 pts, Win by HP = 2 pts, Draw = 1 pt, Loss = 0.

## Validation

```bash
make validate MODEL=your_model.onnx
```

## Built-in Opponents

Four scripted Rust opponents with increasing difficulty: chaser, dogfighter, ace, brawler.
