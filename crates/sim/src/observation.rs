use dogfight_shared::*;

use crate::physics::SimState;

impl SimState {
    /// Build the observation vector for the given player.
    ///
    /// Layout (46 floats):
    ///   Self state:     [0..6)   6 floats
    ///   Opponent state: [6..13)  7 floats
    ///   Bullets:        [13..45) 32 floats (8 slots × 4)
    ///   Meta:           [45]     1 float
    pub fn observe(&self, player: usize) -> Observation {
        let mut data = [0.0f32; OBS_SIZE];
        let opp = 1 - player;
        let me = &self.fighters[player];
        let them = &self.fighters[opp];

        // SELF STATE (6 floats)
        data[0] = me.speed / MAX_SPEED;
        data[1] = me.yaw.cos();
        data[2] = me.yaw.sin();
        data[3] = me.hp as f32 / MAX_HP as f32;
        data[4] = me.gun_cooldown_ticks as f32 / GUN_COOLDOWN_TICKS as f32;
        data[5] = me.position.y / MAX_ALTITUDE;

        // OPPONENT STATE (7 floats)
        let rel = them.position - me.position;
        data[6] = rel.x / ARENA_DIAMETER;
        data[7] = rel.y / ARENA_DIAMETER;
        data[8] = them.speed / MAX_SPEED;
        data[9] = them.yaw.cos();
        data[10] = them.yaw.sin();
        data[11] = them.hp as f32 / MAX_HP as f32;
        data[12] = rel.length() / ARENA_DIAMETER;

        // BULLETS (8 slots × 4 floats = 32 floats, sorted by distance to self)
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
            let base = 13 + slot * 4;
            let brel = bullet.position - me.position;
            data[base] = brel.x / ARENA_DIAMETER;
            data[base + 1] = brel.y / ARENA_DIAMETER;
            data[base + 2] = if bullet.owner == player { 1.0 } else { 0.0 };
            // Angle to bullet (normalized by PI)
            let angle = brel.y.atan2(brel.x);
            data[base + 3] = angle / std::f32::consts::PI;
        }

        // META (1 float)
        let ticks_remaining = MAX_TICKS.saturating_sub(self.tick);
        data[45] = ticks_remaining as f32 / MAX_TICKS as f32;

        Observation { data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_size() {
        let state = SimState::new();
        let obs = state.observe(0);
        assert_eq!(obs.data.len(), OBS_SIZE);
    }

    #[test]
    fn test_observation_symmetry() {
        let state = SimState::new();
        let obs0 = state.observe(0);
        let obs1 = state.observe(1);

        // Speed should be identical
        assert!((obs0.data[0] - obs1.data[0]).abs() < 0.001);
        // HP should be identical
        assert!((obs0.data[3] - obs1.data[3]).abs() < 0.001);
        // Relative positions should be opposite
        assert!((obs0.data[6] + obs1.data[6]).abs() < 0.001);
    }

    #[test]
    fn test_observation_initial_values() {
        let state = SimState::new();
        let obs = state.observe(0);

        // Speed should be MIN_SPEED / MAX_SPEED
        assert!((obs.data[0] - MIN_SPEED / MAX_SPEED).abs() < 0.001);
        // HP should be 1.0 (full)
        assert!((obs.data[3] - 1.0).abs() < 0.001);
        // Gun cooldown should be 0 (ready)
        assert!((obs.data[4] - 0.0).abs() < 0.001);
        // Ticks remaining should be 1.0
        assert!((obs.data[45] - 1.0).abs() < 0.001);
    }
}
