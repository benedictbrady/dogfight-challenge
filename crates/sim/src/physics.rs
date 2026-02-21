use dogfight_shared::*;
use glam::Vec2;
use rand::Rng;
use rand_pcg::Pcg64;
use rand::SeedableRng;

/// Full simulation state for a 1v1 match.
#[derive(Debug, Clone)]
pub struct SimState {
    pub fighters: [FighterState; 2],
    pub bullets: Vec<Bullet>,
    pub tick: u32,
    pub stats: MatchStats,
    pub config: SimConfig,
    /// Previous tick's fighter states (for computing derivatives like closure rate).
    pub prev_fighters: [FighterState; 2],
    /// Per-player observation frame history (last 3 frames, newest at index 0).
    pub obs_history: [[SingleFrameObs; 3]; 2],
    /// Per-player count of frames observed so far (for zero-padding logic).
    pub obs_history_count: [u32; 2],
}

impl SimState {
    /// Default initial speed: above stall threshold for safe spawn.
    pub const SPAWN_SPEED: f32 = 50.0;

    pub fn new() -> Self {
        Self::new_with_config(SimConfig::default())
    }

    pub fn new_with_config(config: SimConfig) -> Self {
        let hp = config.max_hp;
        let fighters = [
            FighterState {
                position: Vec2::new(-200.0, 300.0),
                yaw: 0.0,
                speed: Self::SPAWN_SPEED,
                hp,
                gun_cooldown_ticks: 0,
                alive: true,
                stall_ticks: 0,
            },
            FighterState {
                position: Vec2::new(200.0, 300.0),
                yaw: std::f32::consts::PI,
                speed: Self::SPAWN_SPEED,
                hp,
                gun_cooldown_ticks: 0,
                alive: true,
                stall_ticks: 0,
            },
        ];
        Self {
            prev_fighters: fighters.clone(),
            fighters,
            bullets: Vec::new(),
            tick: 0,
            stats: MatchStats {
                p0_hp: hp,
                p1_hp: hp,
                p0_hits: 0,
                p1_hits: 0,
                p0_shots: 0,
                p1_shots: 0,
            },
            config,
            obs_history: Default::default(),
            obs_history_count: [0; 2],
        }
    }

    pub fn new_with_seed(seed: u64, randomize: bool) -> Self {
        Self::new_with_seed_and_config(seed, randomize, SimConfig::default())
    }

    pub fn new_with_seed_and_config(seed: u64, randomize: bool, config: SimConfig) -> Self {
        if !randomize {
            return Self::new_with_config(config);
        }

        let hp = config.max_hp;
        let mut rng = Pcg64::seed_from_u64(seed);
        let x_offset = rng.gen_range(100.0..300.0f32);
        let alt0 = rng.gen_range(150.0..450.0f32);
        let alt1 = rng.gen_range(150.0..450.0f32);
        let yaw0 = rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
        let yaw1 = rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
        let speed0 = rng.gen_range((config.min_speed + 15.0)..80.0f32);
        let speed1 = rng.gen_range((config.min_speed + 15.0)..80.0f32);

        let fighters = [
            FighterState {
                position: Vec2::new(-x_offset, alt0),
                yaw: yaw0,
                speed: speed0,
                hp,
                gun_cooldown_ticks: 0,
                alive: true,
                stall_ticks: 0,
            },
            FighterState {
                position: Vec2::new(x_offset, alt1),
                yaw: yaw1,
                speed: speed1,
                hp,
                gun_cooldown_ticks: 0,
                alive: true,
                stall_ticks: 0,
            },
        ];
        Self {
            prev_fighters: fighters.clone(),
            fighters,
            bullets: Vec::new(),
            tick: 0,
            stats: MatchStats {
                p0_hp: hp,
                p1_hp: hp,
                p0_hits: 0,
                p1_hits: 0,
                p0_shots: 0,
                p1_shots: 0,
            },
            config,
            obs_history: Default::default(),
            obs_history_count: [0; 2],
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.tick >= MAX_TICKS || !self.fighters[0].alive || !self.fighters[1].alive
    }

    pub fn outcome(&self) -> (MatchOutcome, MatchEndReason) {
        let p0_alive = self.fighters[0].alive;
        let p1_alive = self.fighters[1].alive;

        if !p0_alive && !p1_alive {
            (MatchOutcome::Draw, MatchEndReason::Elimination)
        } else if !p1_alive {
            (MatchOutcome::Player0Win, MatchEndReason::Elimination)
        } else if !p0_alive {
            (MatchOutcome::Player1Win, MatchEndReason::Elimination)
        } else if self.tick >= MAX_TICKS {
            let p0_hp = self.fighters[0].hp;
            let p1_hp = self.fighters[1].hp;
            if p0_hp > p1_hp {
                (MatchOutcome::Player0Win, MatchEndReason::Timeout)
            } else if p1_hp > p0_hp {
                (MatchOutcome::Player1Win, MatchEndReason::Timeout)
            } else {
                (MatchOutcome::Draw, MatchEndReason::Timeout)
            }
        } else {
            (MatchOutcome::Draw, MatchEndReason::Timeout)
        }
    }

    pub fn snapshot(&self) -> ReplayFrame {
        ReplayFrame {
            tick: self.tick,
            fighters: [
                FighterSnapshot::from(&self.fighters[0]),
                FighterSnapshot::from(&self.fighters[1]),
            ],
            bullets: self.bullets.iter().map(BulletSnapshot::from).collect(),
        }
    }

    /// Advance one physics tick.
    pub fn step(&mut self, actions: &[Action; 2]) {
        // Save previous state for derivative computations (closure rate, angular velocity)
        self.prev_fighters = self.fighters.clone();

        for (i, action) in actions.iter().enumerate() {
            if !self.fighters[i].alive {
                continue;
            }
            self.step_fighter(i, action);
        }

        self.step_bullets();
        self.check_collisions();
        self.bullets.retain(|b| b.ticks_remaining > 0);

        self.stats.p0_hp = self.fighters[0].hp;
        self.stats.p1_hp = self.fighters[1].hp;

        self.tick += 1;
    }

    fn step_fighter(&mut self, idx: usize, action: &Action) {
        let cfg = self.config;
        let f = &mut self.fighters[idx];

        if f.stall_ticks > 0 {
            f.stall_ticks -= 1;

            // Rotate nose toward straight down (-PI/2)
            let target = -std::f32::consts::FRAC_PI_2;
            let diff = {
                let mut d = target - f.yaw;
                while d > std::f32::consts::PI { d -= 2.0 * std::f32::consts::PI; }
                while d < -std::f32::consts::PI { d += 2.0 * std::f32::consts::PI; }
                d
            };
            let max_rot = STALL_NOSE_DOWN_RATE * DT;
            f.yaw += diff.clamp(-max_rot, max_rot);
            f.yaw = normalize_angle(f.yaw);

            // Only gravity and drag during stall (no thrust, no yaw input)
            f.speed += (-cfg.gravity * f.yaw.sin()) * DT;
            f.speed -= cfg.drag_coeff * f.speed * DT;
            f.speed = f.speed.clamp(cfg.min_speed, cfg_effective_max_speed(&cfg, f.hp));

            // Early recovery: if speed recovers above STALL_SPEED + 10, clear stall
            if f.speed > STALL_SPEED + 10.0 {
                f.stall_ticks = 0;
            }

            // Integrate position
            let forward = f.forward();
            f.position += forward * f.speed * DT;
            apply_boundaries(f);

            // Cooldown still ticks during stall
            if f.gun_cooldown_ticks > 0 {
                f.gun_cooldown_ticks -= 1;
            }

            // No shooting during stall
            return;
        }

        if f.speed < STALL_SPEED {
            f.stall_ticks = STALL_RECOVERY_TICKS;
            return;
        }

        let turn_rate = cfg_effective_turn_rate(&cfg, f.speed, f.hp);

        // Apply yaw
        let yaw_delta = action.yaw_input.clamp(-1.0, 1.0) * turn_rate * DT;
        f.yaw += yaw_delta;
        f.yaw = normalize_angle(f.yaw);

        // Energy bleed from turning
        f.speed -= cfg.turn_bleed_coeff * yaw_delta.abs() * f.speed;

        // Thrust and drag
        let throttle = action.throttle.clamp(0.0, 1.0);
        f.speed += (throttle * cfg.max_thrust - cfg.drag_coeff * f.speed) * DT;

        // Gravity: climbing (sin>0) costs speed, diving (sin<0) gains speed
        f.speed += (-cfg.gravity * f.yaw.sin()) * DT;

        // Clamp speed (with damage penalty on max)
        f.speed = f.speed.clamp(cfg.min_speed, cfg_effective_max_speed(&cfg, f.hp));

        // Compute forward vector and integrate position
        let forward = f.forward();
        f.position += forward * f.speed * DT;

        // Apply arena boundaries
        apply_boundaries(f);

        // Tick gun cooldown
        if f.gun_cooldown_ticks > 0 {
            f.gun_cooldown_ticks -= 1;
        }

        // Spawn bullet if shooting
        if action.shoot && f.gun_cooldown_ticks == 0 {
            let velocity = forward * cfg.bullet_speed;
            let spawn_pos = f.position + forward * (FIGHTER_RADIUS + BULLET_RADIUS + 1.0);
            self.bullets.push(Bullet {
                position: spawn_pos,
                velocity,
                owner: idx,
                ticks_remaining: cfg.bullet_lifetime_ticks,
            });
            f.gun_cooldown_ticks = cfg.gun_cooldown_ticks;

            match idx {
                0 => self.stats.p0_shots += 1,
                1 => self.stats.p1_shots += 1,
                _ => unreachable!(),
            }
        }
    }

    fn step_bullets(&mut self) {
        for bullet in &mut self.bullets {
            bullet.position += bullet.velocity * DT;
            if bullet.ticks_remaining > 0 {
                bullet.ticks_remaining -= 1;
            }
        }
    }

    fn check_collisions(&mut self) {
        let collision_dist = FIGHTER_RADIUS + BULLET_RADIUS;
        let collision_dist_sq = collision_dist * collision_dist;
        let rear_aspect_cos = self.config.rear_aspect_cone.cos();

        for bullet in &mut self.bullets {
            if bullet.ticks_remaining == 0 {
                continue;
            }

            for (i, fighter) in self.fighters.iter_mut().enumerate() {
                if bullet.owner == i || !fighter.alive {
                    continue;
                }

                let dist_sq = (fighter.position - bullet.position).length_squared();
                if dist_sq <= collision_dist_sq {
                    // Rear-aspect armor check: bullets from behind glance off
                    let bullet_dir = bullet.velocity.normalize();
                    let fighter_fwd = fighter.forward();
                    let dot = bullet_dir.dot(fighter_fwd);

                    // Consume bullet regardless
                    bullet.ticks_remaining = 0;

                    // If bullet is from behind (dot > cos(rear_aspect_cone)),
                    // it glances off — no damage, no hit stat
                    if dot > rear_aspect_cos {
                        break;
                    }

                    fighter.hp = fighter.hp.saturating_sub(1);
                    if fighter.hp == 0 {
                        fighter.alive = false;
                    }

                    match bullet.owner {
                        0 => self.stats.p0_hits += 1,
                        1 => self.stats.p1_hits += 1,
                        _ => unreachable!(),
                    }
                    break;
                }
            }
        }
    }
}

/// Config-aware turn rate at speed.
fn cfg_turn_rate_at_speed(cfg: &SimConfig, speed: f32) -> f32 {
    let t = ((speed - cfg.min_speed) / (cfg.max_speed - cfg.min_speed)).clamp(0.0, 1.0);
    cfg.max_turn_rate + t * (cfg.min_turn_rate - cfg.max_turn_rate)
}

/// Config-aware effective max speed accounting for damage.
fn cfg_effective_max_speed(cfg: &SimConfig, hp: u8) -> f32 {
    cfg.max_speed * (1.0 - DAMAGE_SPEED_PENALTY * (cfg.max_hp - hp) as f32)
}

/// Config-aware effective turn rate accounting for speed and damage.
fn cfg_effective_turn_rate(cfg: &SimConfig, speed: f32, hp: u8) -> f32 {
    cfg_turn_rate_at_speed(cfg, speed) * (1.0 - DAMAGE_TURN_PENALTY * (cfg.max_hp - hp) as f32)
}

/// Compute turn rate based on speed (higher speed = lower turn rate).
/// Uses default config values — kept for external callers and tests.
pub fn turn_rate_at_speed(speed: f32) -> f32 {
    let t = ((speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)).clamp(0.0, 1.0);
    MAX_TURN_RATE + t * (MIN_TURN_RATE - MAX_TURN_RATE)
}

/// Effective max speed accounting for damage.
/// Uses default config values — kept for external callers and tests.
pub fn effective_max_speed(hp: u8) -> f32 {
    MAX_SPEED * (1.0 - DAMAGE_SPEED_PENALTY * (MAX_HP - hp) as f32)
}

/// Effective turn rate accounting for speed and damage.
/// Uses default config values — kept for external callers and tests.
pub fn effective_turn_rate(speed: f32, hp: u8) -> f32 {
    turn_rate_at_speed(speed) * (1.0 - DAMAGE_TURN_PENALTY * (MAX_HP - hp) as f32)
}

/// Normalize angle to [-PI, PI].
fn normalize_angle(mut a: f32) -> f32 {
    use std::f32::consts::PI;
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Apply arena boundary forces and hard clamps (rectangular: horizontal + ground/ceiling).
fn apply_boundaries(f: &mut FighterState) {
    // Horizontal boundaries (x axis)
    let abs_x = f.position.x.abs();
    if abs_x > BOUNDARY_START_HORIZ {
        let penetration =
            ((abs_x - BOUNDARY_START_HORIZ) / (ARENA_RADIUS - BOUNDARY_START_HORIZ)).clamp(0.0, 1.0);
        let force = penetration * penetration * BOUNDARY_FORCE;
        f.position.x -= f.position.x.signum() * force * DT;
    }
    f.position.x = f.position.x.clamp(-ARENA_RADIUS, ARENA_RADIUS);

    // Ground boundary (y=0)
    if f.position.y < ALT_BOUNDARY_LOW {
        let penetration =
            ((ALT_BOUNDARY_LOW - f.position.y) / ALT_BOUNDARY_LOW).clamp(0.0, 1.0);
        let force = penetration * penetration * BOUNDARY_FORCE;
        f.position.y += force * DT;
    }
    if f.position.y < 0.0 {
        f.position.y = 0.0;
    }

    // Ceiling boundary (y=MAX_ALTITUDE)
    if f.position.y > ALT_BOUNDARY_HIGH {
        let penetration =
            ((f.position.y - ALT_BOUNDARY_HIGH) / (MAX_ALTITUDE - ALT_BOUNDARY_HIGH)).clamp(0.0, 1.0);
        let force = penetration * penetration * BOUNDARY_FORCE;
        f.position.y -= force * DT;
    }
    if f.position.y > MAX_ALTITUDE {
        f.position.y = MAX_ALTITUDE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turn_rate_at_speeds() {
        let rate_min = turn_rate_at_speed(MIN_SPEED);
        let rate_max = turn_rate_at_speed(MAX_SPEED);
        assert!((rate_min - MAX_TURN_RATE).abs() < 0.001);
        assert!((rate_max - MIN_TURN_RATE).abs() < 0.001);

        let rate_mid = turn_rate_at_speed(150.0);
        assert!(rate_mid > MIN_TURN_RATE);
        assert!(rate_mid < MAX_TURN_RATE);
    }

    #[test]
    fn test_initial_state() {
        let state = SimState::new();
        assert!(state.fighters[0].alive);
        assert!(state.fighters[1].alive);
        assert_eq!(state.fighters[0].hp, MAX_HP);
        assert_eq!(state.fighters[1].hp, MAX_HP);
        assert_eq!(state.tick, 0);
        assert!(state.bullets.is_empty());
    }

    #[test]
    fn test_do_nothing_movement() {
        let mut state = SimState::new();
        let actions = [Action::none(), Action::none()];

        for _ in 0..120 {
            state.step(&actions);
        }

        assert!(state.fighters[0].alive);
        assert!(state.fighters[1].alive);
        assert_eq!(state.tick, 120);
    }

    #[test]
    fn test_turning_bleeds_speed() {
        let mut state = SimState::new();
        state.fighters[0].speed = 150.0;
        let initial_speed = state.fighters[0].speed;

        let actions = [
            Action {
                yaw_input: 1.0,
                throttle: 0.0,
                shoot: false,
            },
            Action::none(),
        ];

        for _ in 0..60 {
            state.step(&actions);
        }

        // Turning with no throttle should bleed speed
        assert!(state.fighters[0].speed < initial_speed);
    }

    #[test]
    fn test_bullet_spawn_and_cooldown() {
        let mut state = SimState::new();
        let shoot_action = Action {
            yaw_input: 0.0,
            throttle: 1.0,
            shoot: true,
        };

        state.step(&[shoot_action, Action::none()]);
        assert_eq!(state.bullets.len(), 1);
        assert_eq!(state.fighters[0].gun_cooldown_ticks, GUN_COOLDOWN_TICKS);

        state.step(&[shoot_action, Action::none()]);
        assert_eq!(state.bullets.len(), 1);
    }

    #[test]
    fn test_bullet_collision() {
        let mut state = SimState::new();
        state.fighters[0].position = Vec2::new(0.0, 300.0);
        state.fighters[0].yaw = 0.0;
        state.fighters[1].position = Vec2::new(50.0, 300.0);
        state.fighters[1].yaw = std::f32::consts::PI;

        let shoot_action = Action {
            yaw_input: 0.0,
            throttle: 0.0,
            shoot: true,
        };

        state.step(&[shoot_action, shoot_action]);

        let no_action = [Action::none(), Action::none()];
        for _ in 0..30 {
            state.step(&no_action);
            if state.fighters[0].hp < MAX_HP || state.fighters[1].hp < MAX_HP {
                break;
            }
        }

        let total_hp = state.fighters[0].hp as u32 + state.fighters[1].hp as u32;
        assert!(total_hp < (MAX_HP as u32) * 2);
    }

    #[test]
    fn test_speed_clamping() {
        let mut state = SimState::new();
        state.fighters[0].speed = MAX_SPEED + 100.0;

        state.step(&[Action::none(), Action::none()]);
        assert!(state.fighters[0].speed <= MAX_SPEED);

        // Speed below stall triggers stall, but speed is still clamped >= MIN_SPEED
        state.fighters[0].speed = STALL_SPEED + 1.0; // above stall, won't trigger stall entry
        state.step(&[Action::none(), Action::none()]);
        assert!(state.fighters[0].speed >= MIN_SPEED);
    }

    #[test]
    fn test_arena_boundary_horizontal() {
        let mut state = SimState::new();
        state.fighters[0].position = Vec2::new(600.0, 300.0);

        state.step(&[Action::none(), Action::none()]);

        assert!(state.fighters[0].position.x <= ARENA_RADIUS + 1.0);
    }

    #[test]
    fn test_arena_boundary_ground() {
        let mut state = SimState::new();
        state.fighters[0].position = Vec2::new(0.0, -50.0);

        state.step(&[Action::none(), Action::none()]);

        assert!(state.fighters[0].position.y >= 0.0);
    }

    #[test]
    fn test_arena_boundary_ceiling() {
        let mut state = SimState::new();
        state.fighters[0].position = Vec2::new(0.0, 700.0);

        state.step(&[Action::none(), Action::none()]);

        assert!(state.fighters[0].position.y <= MAX_ALTITUDE + 1.0);
    }

    #[test]
    fn test_gravity_diving_gains_speed() {
        let mut state = SimState::new();
        // Point downward (negative sin = diving)
        state.fighters[0].yaw = -std::f32::consts::FRAC_PI_2; // pointing straight down
        state.fighters[0].speed = 60.0; // low speed so gravity exceeds drag
        state.fighters[0].position = Vec2::new(0.0, 400.0); // high altitude so no ground push

        let actions = [
            Action { yaw_input: 0.0, throttle: 0.0, shoot: false },
            Action::none(),
        ];

        state.step(&actions);

        // Diving should gain speed from gravity (overcoming drag at low speed)
        assert!(state.fighters[0].speed > 60.0,
            "Diving should gain speed, got {}", state.fighters[0].speed);
    }

    #[test]
    fn test_initial_positions_match_expected() {
        let state = SimState::new();
        let snap = state.snapshot();
        // P0 starts at (-200, 300)
        assert!(
            (snap.fighters[0].x - (-200.0)).abs() < 0.001,
            "P0 x should be -200, got {}",
            snap.fighters[0].x
        );
        assert!(
            (snap.fighters[0].y - 300.0).abs() < 0.001,
            "P0 y should be 300, got {}",
            snap.fighters[0].y
        );
        // P1 starts at (200, 300)
        assert!(
            (snap.fighters[1].x - 200.0).abs() < 0.001,
            "P1 x should be 200, got {}",
            snap.fighters[1].x
        );
        assert!(
            (snap.fighters[1].y - 300.0).abs() < 0.001,
            "P1 y should be 300, got {}",
            snap.fighters[1].y
        );
    }

    #[test]
    fn test_frame_position_continuity() {
        // Verify no teleportation: consecutive frames should never have
        // position jumps larger than physics allows.
        // Max possible displacement per frame interval:
        //   MAX_SPEED * FRAME_INTERVAL * DT = 250 * 4 * (1/120) ≈ 8.33
        // Use generous threshold to account for boundary forces.
        let max_delta = MAX_SPEED * FRAME_INTERVAL as f32 * DT * 1.5;

        let mut state = SimState::new();
        let actions = [Action::none(), Action::none()];
        let mut prev_snap = state.snapshot();

        // Run 600 ticks (5 seconds) and check every frame
        for _ in 0..600 {
            state.step(&actions);
            if state.tick % FRAME_INTERVAL == 0 {
                let snap = state.snapshot();
                for p in 0..2 {
                    let dx = snap.fighters[p].x - prev_snap.fighters[p].x;
                    let dy = snap.fighters[p].y - prev_snap.fighters[p].y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    assert!(
                        dist <= max_delta,
                        "P{} teleported {:.1} units at tick {} (max {:.1}): ({:.1},{:.1}) -> ({:.1},{:.1})",
                        p,
                        dist,
                        state.tick,
                        max_delta,
                        prev_snap.fighters[p].x,
                        prev_snap.fighters[p].y,
                        snap.fighters[p].x,
                        snap.fighters[p].y,
                    );
                }
                prev_snap = snap;
            }
        }
    }

    #[test]
    fn test_frame_continuity_with_chaser() {
        // Same continuity test but with active policies (Chaser)
        // to catch teleportation during aggressive maneuvering.
        use crate::opponents::ChaserPolicy;

        let config = MatchConfig {
            max_ticks: 600,
            ..Default::default()
        };
        let mut p0 = ChaserPolicy::new();
        let mut p1 = ChaserPolicy::new();

        let replay = crate::run_match(&config, &mut p0, &mut p1);

        let max_delta = MAX_SPEED * FRAME_INTERVAL as f32 * DT * 1.5;

        for i in 1..replay.frames.len() {
            let prev = &replay.frames[i - 1];
            let curr = &replay.frames[i];
            for p in 0..2 {
                let dx = curr.fighters[p].x - prev.fighters[p].x;
                let dy = curr.fighters[p].y - prev.fighters[p].y;
                let dist = (dx * dx + dy * dy).sqrt();
                assert!(
                    dist <= max_delta,
                    "P{} teleported {:.1} units between frames {} and {} (ticks {}->{}): ({:.1},{:.1}) -> ({:.1},{:.1})",
                    p,
                    dist,
                    i - 1,
                    i,
                    prev.tick,
                    curr.tick,
                    prev.fighters[p].x,
                    prev.fighters[p].y,
                    curr.fighters[p].x,
                    curr.fighters[p].y,
                );
            }
        }
    }

    #[test]
    fn test_gravity_climbing_costs_speed() {
        let mut state = SimState::new();
        // Point upward (positive sin = climbing)
        state.fighters[0].yaw = std::f32::consts::FRAC_PI_2; // pointing straight up
        state.fighters[0].speed = 150.0;
        state.fighters[0].position = Vec2::new(0.0, 300.0);

        let actions = [
            Action { yaw_input: 0.0, throttle: 0.0, shoot: false },
            Action::none(),
        ];

        state.step(&actions);

        // Climbing should cost speed due to gravity
        assert!(state.fighters[0].speed < 150.0,
            "Climbing should cost speed, got {}", state.fighters[0].speed);
    }

    #[test]
    fn test_stall_entry_at_low_speed() {
        let mut state = SimState::new();
        // Set speed just below stall threshold
        state.fighters[0].speed = STALL_SPEED - 1.0;
        state.fighters[0].position = Vec2::new(0.0, 300.0);

        state.step(&[Action::none(), Action::none()]);

        assert!(
            state.fighters[0].stall_ticks > 0,
            "Fighter below stall speed should enter stall, stall_ticks={}",
            state.fighters[0].stall_ticks
        );
    }

    #[test]
    fn test_stall_ignores_yaw_input() {
        let mut state = SimState::new();
        state.fighters[0].speed = STALL_SPEED - 1.0;
        state.fighters[0].yaw = 0.0; // facing right
        state.fighters[0].position = Vec2::new(0.0, 300.0);

        // Trigger stall
        state.step(&[Action::none(), Action::none()]);
        assert!(state.fighters[0].stall_ticks > 0);

        // Now try to yaw hard left — should be ignored, nose should drift downward
        let hard_left = Action { yaw_input: 1.0, throttle: 1.0, shoot: false };
        for _ in 0..10 {
            if state.fighters[0].stall_ticks == 0 { break; }
            state.step(&[hard_left, Action::none()]);
        }

        // Yaw should have moved toward -PI/2 (down), not toward +PI/2 (up)
        assert!(
            state.fighters[0].yaw.sin() < 0.1,
            "During stall, nose should rotate downward, yaw={:.2} sin={:.2}",
            state.fighters[0].yaw,
            state.fighters[0].yaw.sin()
        );
    }

    #[test]
    fn test_stall_recovery() {
        let mut state = SimState::new();
        // Start in stall: low speed, pointing slightly down (will gain speed from gravity)
        state.fighters[0].speed = STALL_SPEED - 5.0;
        state.fighters[0].yaw = -0.5; // slightly downward
        state.fighters[0].position = Vec2::new(0.0, 400.0);

        // Trigger stall entry
        state.step(&[Action::none(), Action::none()]);
        assert!(state.fighters[0].stall_ticks > 0, "Should enter stall");

        // Run through stall recovery (gravity while nose-down should recover speed)
        for _ in 0..120 {
            state.step(&[Action { yaw_input: 0.0, throttle: 1.0, shoot: false }, Action::none()]);
        }

        // Should have exited stall by now
        assert_eq!(
            state.fighters[0].stall_ticks, 0,
            "Fighter should recover from stall after enough ticks"
        );
    }

    #[test]
    fn test_no_shooting_during_stall() {
        let mut state = SimState::new();
        state.fighters[0].speed = STALL_SPEED - 1.0;
        state.fighters[0].position = Vec2::new(0.0, 300.0);

        // Enter stall
        state.step(&[Action::none(), Action::none()]);
        assert!(state.fighters[0].stall_ticks > 0);

        // Try to shoot during stall
        let shoot = Action { yaw_input: 0.0, throttle: 1.0, shoot: true };
        let bullets_before = state.bullets.len();
        state.step(&[shoot, Action::none()]);

        assert_eq!(
            state.bullets.len(),
            bullets_before,
            "Should not be able to shoot during stall"
        );
    }

    #[test]
    fn test_damage_reduces_max_speed() {
        // Full HP: max speed = 250
        assert!((effective_max_speed(MAX_HP) - MAX_SPEED).abs() < 0.01);

        // 1 HP lost: max speed = 250 * (1 - 0.03) = 242.5
        let one_damage = effective_max_speed(MAX_HP - 1);
        assert!(
            one_damage < MAX_SPEED,
            "Damaged fighter should have lower max speed: {}",
            one_damage
        );
        assert!(
            (one_damage - MAX_SPEED * (1.0 - DAMAGE_SPEED_PENALTY)).abs() < 0.01,
            "One damage should reduce max speed by DAMAGE_SPEED_PENALTY: {}",
            one_damage
        );

        // 2 HP lost: even lower
        let two_damage = effective_max_speed(MAX_HP - 2);
        assert!(two_damage < one_damage);
    }

    #[test]
    fn test_damage_reduces_turn_rate() {
        let speed = 100.0;
        let full_hp_rate = effective_turn_rate(speed, MAX_HP);
        let damaged_rate = effective_turn_rate(speed, MAX_HP - 1);
        let badly_damaged_rate = effective_turn_rate(speed, MAX_HP - 2);

        assert!(
            damaged_rate < full_hp_rate,
            "Damaged fighter should turn slower: {} vs {}",
            damaged_rate,
            full_hp_rate
        );
        assert!(
            badly_damaged_rate < damaged_rate,
            "More damage = slower turns: {} vs {}",
            badly_damaged_rate,
            damaged_rate
        );
    }

    #[test]
    fn test_rear_aspect_bullet_no_damage() {
        // P0 behind P1, both facing right. P0's bullet hits P1 from behind.
        let mut state = SimState::new();
        state.fighters[0].position = Vec2::new(0.0, 300.0);
        state.fighters[0].yaw = 0.0; // facing right
        state.fighters[1].position = Vec2::new(50.0, 300.0);
        state.fighters[1].yaw = 0.0; // also facing right (P0 is behind P1)

        let shoot = Action { yaw_input: 0.0, throttle: 0.0, shoot: true };
        state.step(&[shoot, Action::none()]);

        // Bullet should be spawned
        assert_eq!(state.bullets.len(), 1);

        // Run until bullet reaches P1
        let no_action = [Action::none(), Action::none()];
        for _ in 0..30 {
            state.step(&no_action);
        }

        // Rear-aspect: P1 should take NO damage
        assert_eq!(
            state.fighters[1].hp, MAX_HP,
            "Rear-aspect bullet should do no damage. HP={}",
            state.fighters[1].hp
        );
    }

    #[test]
    fn test_side_aspect_bullet_full_damage() {
        // P0 shoots P1 from perpendicular (side aspect)
        let mut state = SimState::new();
        state.fighters[0].position = Vec2::new(0.0, 300.0);
        state.fighters[0].yaw = 0.0; // facing right
        state.fighters[1].position = Vec2::new(50.0, 300.0);
        state.fighters[1].yaw = std::f32::consts::FRAC_PI_2; // facing up (perpendicular)

        let shoot = Action { yaw_input: 0.0, throttle: 0.0, shoot: true };
        state.step(&[shoot, Action::none()]);

        let no_action = [Action::none(), Action::none()];
        for _ in 0..30 {
            state.step(&no_action);
        }

        // Side aspect: P1 should take full damage
        assert!(
            state.fighters[1].hp < MAX_HP,
            "Side-aspect bullet should deal damage. HP={}",
            state.fighters[1].hp
        );
    }

    #[test]
    fn test_head_on_bullet_full_damage() {
        // Head-on: P0 facing right, P1 facing left (toward each other)
        let mut state = SimState::new();
        state.fighters[0].position = Vec2::new(0.0, 300.0);
        state.fighters[0].yaw = 0.0; // facing right
        state.fighters[1].position = Vec2::new(50.0, 300.0);
        state.fighters[1].yaw = std::f32::consts::PI; // facing left

        let shoot = Action { yaw_input: 0.0, throttle: 0.0, shoot: true };
        state.step(&[shoot, Action::none()]);

        let no_action = [Action::none(), Action::none()];
        for _ in 0..30 {
            state.step(&no_action);
        }

        // Head-on: P1 should take full damage
        assert!(
            state.fighters[1].hp < MAX_HP,
            "Head-on bullet should deal damage. HP={}",
            state.fighters[1].hp
        );
    }

    #[test]
    fn test_randomized_spawns() {
        let state1 = SimState::new_with_seed(42, true);
        let state2 = SimState::new_with_seed(42, true);
        let state_default = SimState::new();

        // Same seed should produce same state
        assert!(
            (state1.fighters[0].position.x - state2.fighters[0].position.x).abs() < 0.001,
            "Same seed should produce same positions"
        );

        // Randomized should differ from default
        assert!(
            (state1.fighters[0].position.y - state_default.fighters[0].position.y).abs() > 0.1
                || (state1.fighters[0].yaw - state_default.fighters[0].yaw).abs() > 0.1,
            "Randomized spawn should differ from default"
        );
    }

    #[test]
    fn test_randomized_spawns_false_returns_default() {
        let state = SimState::new_with_seed(42, false);
        let default = SimState::new();

        assert!(
            (state.fighters[0].position.x - default.fighters[0].position.x).abs() < 0.001,
            "randomize=false should return default positions"
        );
    }
}
