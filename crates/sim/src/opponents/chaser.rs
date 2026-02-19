use dogfight_shared::*;
use crate::policy::Policy;
use super::tactics::*;

/// Upgraded pressure fighter: relentless pursuit with yo-yo maneuvers and bullet evasion.
/// Beats Brawler (constant pressure prevents slow turn-fights), loses to Ace (can't catch altitude).
pub struct ChaserPolicy {
    config: SimConfig,
    evade_timer: u32,
    evade_dir: f32,
    yo_yo_timer: u32,
    /// 1.0 = high yo-yo (pull up), -1.0 = low yo-yo (push down)
    yo_yo_phase: f32,
}

impl ChaserPolicy {
    pub fn new() -> Self {
        Self::with_config(SimConfig::default())
    }

    pub fn with_config(config: SimConfig) -> Self {
        Self {
            config,
            evade_timer: 0,
            evade_dir: 1.0,
            yo_yo_timer: 0,
            yo_yo_phase: 0.0,
        }
    }
}

impl Default for ChaserPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Policy for ChaserPolicy {
    fn name(&self) -> &str {
        "chaser"
    }

    fn act(&mut self, obs: &Observation) -> Action {
        let ts = extract_tactical_state_with_config(obs, &self.config);

        // Emergency altitude override
        if let Some(yaw_input) = altitude_safety(ts.altitude, ts.my_yaw) {
            return Action {
                yaw_input,
                throttle: 1.0,
                shoot: false,
            };
        }

        // Tick timers
        if self.evade_timer > 0 {
            self.evade_timer -= 1;
        }
        if self.yo_yo_timer > 0 {
            self.yo_yo_timer -= 1;
        }

        // Bullet evasion: short 15-tick hard turn when enemy bullet within 80m
        if ts.nearest_enemy_bullet_dist < 80.0 && self.evade_timer == 0 {
            self.evade_timer = 15;
            self.evade_dir = -self.evade_dir;
        }

        if self.evade_timer > 0 {
            return Action {
                yaw_input: self.evade_dir,
                throttle: 0.8,
                shoot: can_shoot(&ts, 0.3, 320.0),
            };
        }

        // Yo-yo maneuvers to manage energy during pursuit
        if self.yo_yo_timer == 0 {
            // High yo-yo: when closing too fast at close range — pull up to bleed speed
            // and drop behind the opponent
            if ts.distance < 120.0 && ts.closing_rate > 50.0 && ts.altitude < 500.0 {
                self.yo_yo_timer = 40;
                self.yo_yo_phase = 1.0; // pull up
            }
            // Low yo-yo: when separating and distant — dive to gain speed for catch-up
            else if ts.distance > 300.0 && ts.closing_rate < -20.0 && ts.altitude > 150.0 {
                self.yo_yo_timer = 30;
                self.yo_yo_phase = -1.0; // push down
            }
        }

        let desired_yaw = smart_aim(&ts, &self.config, 1.0);
        let mut yaw_diff = angle_diff(desired_yaw, ts.my_yaw);

        // Apply yo-yo bias
        if self.yo_yo_timer > 0 {
            yaw_diff += self.yo_yo_phase * 0.4;
        }

        // Altitude management: nudge toward mid-altitude band
        if ts.altitude < 120.0 && ts.my_yaw.sin() < 0.1 {
            yaw_diff += 0.25;
        } else if ts.altitude > 520.0 && ts.my_yaw.sin() > -0.1 {
            yaw_diff -= 0.25;
        }

        let yaw_input = (yaw_diff * 2.5).clamp(-1.0, 1.0);

        // Throttle management: reduce during hard turns at high speed
        let throttle: f32 = if ts.my_speed > 120.0 && yaw_input.abs() > 0.7 {
            0.7
        } else {
            1.0
        };

        let shoot = can_shoot(&ts, 0.22, 320.0);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);
        let throttle = throttle.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::SimState;

    #[test]
    fn test_chaser_produces_actions() {
        let state = SimState::new();
        let obs = state.observe(0);
        let mut chaser = ChaserPolicy::new();
        let action = chaser.act(&obs);

        // Player faces east (yaw=0), opponent is east -> yaw_diff ~ 0
        assert!(action.yaw_input.abs() < 0.5);
        assert!(action.throttle > 0.0);
    }
}
