use dogfight_shared::*;
use glam::Vec2;

/// Full simulation state for a 1v1 match.
#[derive(Debug, Clone)]
pub struct SimState {
    pub fighters: [FighterState; 2],
    pub bullets: Vec<Bullet>,
    pub tick: u32,
    pub stats: MatchStats,
}

impl SimState {
    pub fn new() -> Self {
        Self {
            fighters: [
                FighterState {
                    position: Vec2::new(-200.0, 300.0),
                    yaw: 0.0, // facing right
                    speed: MIN_SPEED,
                    hp: MAX_HP,
                    gun_cooldown_ticks: 0,
                    alive: true,
                },
                FighterState {
                    position: Vec2::new(200.0, 300.0),
                    yaw: std::f32::consts::PI, // facing left
                    speed: MIN_SPEED,
                    hp: MAX_HP,
                    gun_cooldown_ticks: 0,
                    alive: true,
                },
            ],
            bullets: Vec::new(),
            tick: 0,
            stats: MatchStats {
                p0_hp: MAX_HP,
                p1_hp: MAX_HP,
                p0_hits: 0,
                p1_hits: 0,
                p0_shots: 0,
                p1_shots: 0,
            },
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
        let f = &mut self.fighters[idx];

        // Compute turn rate at current speed
        let turn_rate = turn_rate_at_speed(f.speed);

        // Apply yaw
        let yaw_delta = action.yaw_input.clamp(-1.0, 1.0) * turn_rate * DT;
        f.yaw += yaw_delta;
        f.yaw = normalize_angle(f.yaw);

        // Energy bleed from turning
        f.speed -= TURN_BLEED_COEFF * yaw_delta.abs() * f.speed;

        // Thrust and drag
        let throttle = action.throttle.clamp(0.0, 1.0);
        f.speed += (throttle * MAX_THRUST - DRAG_COEFF * f.speed) * DT;

        // Gravity: climbing (sin>0) costs speed, diving (sin<0) gains speed
        f.speed += (-GRAVITY * f.yaw.sin()) * DT;

        // Clamp speed
        f.speed = f.speed.clamp(MIN_SPEED, MAX_SPEED);

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
            let velocity = forward * BULLET_SPEED;
            let spawn_pos = f.position + forward * (FIGHTER_RADIUS + BULLET_RADIUS + 1.0);
            self.bullets.push(Bullet {
                position: spawn_pos,
                velocity,
                owner: idx,
                ticks_remaining: BULLET_LIFETIME_TICKS,
            });
            f.gun_cooldown_ticks = GUN_COOLDOWN_TICKS;

            match idx {
                0 => self.stats.p0_shots += 1,
                1 => self.stats.p1_shots += 1,
                _ => {}
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
                    fighter.hp = fighter.hp.saturating_sub(1);
                    if fighter.hp == 0 {
                        fighter.alive = false;
                    }
                    bullet.ticks_remaining = 0;

                    match bullet.owner {
                        0 => self.stats.p0_hits += 1,
                        1 => self.stats.p1_hits += 1,
                        _ => {}
                    }
                    break;
                }
            }
        }
    }
}

/// Compute turn rate based on speed (higher speed = lower turn rate).
pub fn turn_rate_at_speed(speed: f32) -> f32 {
    let t = ((speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)).clamp(0.0, 1.0);
    MAX_TURN_RATE + t * (MIN_TURN_RATE - MAX_TURN_RATE)
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

        state.fighters[0].speed = 0.0;
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
        use crate::policy::Policy;

        let config = MatchConfig {
            max_ticks: 600,
            ..Default::default()
        };
        let mut p0 = ChaserPolicy;
        let mut p1 = ChaserPolicy;

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
}
