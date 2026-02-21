use dogfight_shared::*;
use std::f32::consts::PI;

use crate::physics::SimState;

/// Normalize angle to [-PI, PI].
fn normalize_angle(mut a: f32) -> f32 {
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

impl SimState {
    /// Compute a single enriched observation frame (56 floats) for the given player.
    pub fn observe_single_frame(&self, player: usize) -> SingleFrameObs {
        let mut data = [0.0f32; SINGLE_FRAME_OBS_SIZE];
        let opp = 1 - player;
        let me = &self.fighters[player];
        let them = &self.fighters[opp];
        let prev_me = &self.prev_fighters[player];
        let prev_them = &self.prev_fighters[opp];

        // SELF STATE (8 floats) [0..8)
        data[0] = me.speed / self.config.max_speed;
        data[1] = me.yaw.cos();
        data[2] = me.yaw.sin();
        data[3] = me.hp as f32 / self.config.max_hp as f32;
        data[4] = me.gun_cooldown_ticks as f32 / self.config.gun_cooldown_ticks as f32;
        data[5] = me.position.y / MAX_ALTITUDE;
        data[6] = me.position.x / ARENA_RADIUS;
        let my_energy = me.speed * me.speed + 2.0 * self.config.gravity * me.position.y;
        data[7] = my_energy / MAX_ENERGY;

        // OPPONENT STATE (11 floats) [8..19)
        let rel = them.position - me.position;
        let distance = rel.length();
        data[8] = rel.x / ARENA_DIAMETER;
        data[9] = rel.y / ARENA_DIAMETER;
        data[10] = them.speed / self.config.max_speed;
        data[11] = them.yaw.cos();
        data[12] = them.yaw.sin();
        data[13] = them.hp as f32 / self.config.max_hp as f32;
        data[14] = distance / ARENA_DIAMETER;

        let prev_rel = prev_them.position - prev_me.position;
        let prev_distance = prev_rel.length();
        let closure_rate = if self.tick > 0 {
            (prev_distance - distance) / DT
        } else {
            0.0
        };
        data[15] = (closure_rate / self.config.max_speed).clamp(-1.0, 1.0);

        let angular_velocity = if self.tick > 0 {
            normalize_angle(them.yaw - prev_them.yaw) / DT
        } else {
            0.0
        };
        data[16] = (angular_velocity / MAX_TURN_RATE).clamp(-1.0, 1.0);

        let opp_energy = them.speed * them.speed + 2.0 * self.config.gravity * them.position.y;
        data[17] = opp_energy / MAX_ENERGY;

        let angle_opp_to_me = f32::atan2(-rel.y, -rel.x);
        let angle_off_tail = normalize_angle(angle_opp_to_me - them.yaw);
        data[18] = angle_off_tail / PI;

        // BULLETS (8 slots x 4 floats = 32 floats) [19..51)
        let mut bullet_entries: Vec<(f32, &Bullet)> = self
            .bullets
            .iter()
            .map(|b| {
                let dist = (b.position - me.position).length();
                (dist, b)
            })
            .collect();
        bullet_entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (slot, (_dist, bullet)) in bullet_entries.iter().take(MAX_BULLET_SLOTS).enumerate() {
            let base = 19 + slot * 4;
            let brel = bullet.position - me.position;
            data[base] = brel.x / ARENA_DIAMETER;
            data[base + 1] = brel.y / ARENA_DIAMETER;
            data[base + 2] = if bullet.owner == player { 1.0 } else { 0.0 };
            let angle = brel.y.atan2(brel.x);
            data[base + 3] = angle / PI;
        }

        // RELATIVE GEOMETRY (4 floats) [51..55)
        let angle_to_opp = f32::atan2(rel.y, rel.x);
        let angle_off_nose = normalize_angle(angle_to_opp - me.yaw);
        data[51] = angle_off_nose / PI;

        let opp_angle_to_me_dir = f32::atan2(-rel.y, -rel.x);
        let opp_angle_off_nose = normalize_angle(opp_angle_to_me_dir - them.yaw);
        data[52] = opp_angle_off_nose / PI;

        let my_vel_x = me.speed * me.yaw.cos();
        let my_vel_y = me.speed * me.yaw.sin();
        let opp_vel_x = them.speed * them.yaw.cos();
        let opp_vel_y = them.speed * them.yaw.sin();
        data[53] = (my_vel_x - opp_vel_x) / self.config.max_speed;
        data[54] = (my_vel_y - opp_vel_y) / self.config.max_speed;

        // META (1 float) [55]
        let ticks_remaining = MAX_TICKS.saturating_sub(self.tick);
        data[55] = ticks_remaining as f32 / MAX_TICKS as f32;

        SingleFrameObs { data }
    }

    /// Build the full stacked observation vector for the given player.
    pub fn observe(&mut self, player: usize) -> Observation {
        let current = self.observe_single_frame(player);

        let mut data = [0.0f32; OBS_SIZE];
        data[..SINGLE_FRAME_OBS_SIZE].copy_from_slice(&current.data);

        let count = self.obs_history_count[player] as usize;
        for i in 0..3 {
            let dest_start = (i + 1) * SINGLE_FRAME_OBS_SIZE;
            if i < count {
                data[dest_start..dest_start + SINGLE_FRAME_OBS_SIZE]
                    .copy_from_slice(&self.obs_history[player][i].data);
            }
        }

        self.obs_history[player][2] = self.obs_history[player][1];
        self.obs_history[player][1] = self.obs_history[player][0];
        self.obs_history[player][0] = current;
        if self.obs_history_count[player] < 3 {
            self.obs_history_count[player] += 1;
        }

        Observation { data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_size() {
        let mut state = SimState::new();
        let obs = state.observe(0);
        assert_eq!(obs.data.len(), OBS_SIZE);
    }

    #[test]
    fn test_single_frame_size() {
        let state = SimState::new();
        let frame = state.observe_single_frame(0);
        assert_eq!(frame.data.len(), SINGLE_FRAME_OBS_SIZE);
    }

    #[test]
    fn test_observation_symmetry() {
        let mut state = SimState::new();
        let obs0 = state.observe(0);
        let obs1 = state.observe(1);
        assert!((obs0.data[0] - obs1.data[0]).abs() < 0.001);
        assert!((obs0.data[3] - obs1.data[3]).abs() < 0.001);
        assert!((obs0.data[8] + obs1.data[8]).abs() < 0.001);
    }

    #[test]
    fn test_observation_initial_values() {
        let mut state = SimState::new();
        let obs = state.observe(0);
        let expected_speed = SimState::SPAWN_SPEED / MAX_SPEED;
        assert!((obs.data[0] - expected_speed).abs() < 0.001);
        assert!((obs.data[3] - 1.0).abs() < 0.001);
        assert!((obs.data[4] - 0.0).abs() < 0.001);
        assert!((obs.data[55] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_stacking_zero_padding() {
        let mut state = SimState::new();
        let obs = state.observe(0);
        for i in SINGLE_FRAME_OBS_SIZE..OBS_SIZE {
            assert!(obs.data[i].abs() < 0.001, "index {} should be zero, got {}", i, obs.data[i]);
        }
    }

    #[test]
    fn test_frame_stacking_accumulation() {
        let mut state = SimState::new();
        let obs1 = state.observe(0);
        let frame1_speed = obs1.data[0];
        state.step(&[Action::none(), Action::none()]);
        let obs2 = state.observe(0);
        assert!((obs2.data[SINGLE_FRAME_OBS_SIZE] - frame1_speed).abs() < 0.01);
        let frame3_start = 3 * SINGLE_FRAME_OBS_SIZE;
        assert!(obs2.data[frame3_start].abs() < 0.001);
    }

    #[test]
    fn test_derived_features_tick_zero() {
        let state = SimState::new();
        let frame = state.observe_single_frame(0);
        assert!((frame.data[15]).abs() < 0.001);
        assert!((frame.data[16]).abs() < 0.001);
        assert!(frame.data[7] > 0.0);
        assert!(frame.data[17] > 0.0);
    }

    #[test]
    fn test_history_cleared_on_reset() {
        let mut state = SimState::new();
        state.observe(0);
        state.step(&[Action::none(), Action::none()]);
        state.observe(0);
        let mut state2 = SimState::new();
        let obs = state2.observe(0);
        for i in SINGLE_FRAME_OBS_SIZE..OBS_SIZE {
            assert!(obs.data[i].abs() < 0.001, "index {} should be zero, got {}", i, obs.data[i]);
        }
    }
}
