use dogfight_shared::*;
use crate::policy::Policy;
use super::tactics::*;

#[derive(Debug, Clone, Copy, PartialEq)]
enum BrawlerPhase {
    /// Close distance to get into knife-fight range
    Close,
    /// In close range: slow down, maximize turn rate, snap-shoot
    Brawl,
    /// Opponent behind and closing: throttle zero, wait for overshoot
    OvershootBait,
    /// Opponent overshot: punish with wide-angle shots
    OvershootPunish,
    /// Low energy retreat: climb to recover altitude/speed
    Retreat,
}

/// Close-range turn fighter. Deliberately slows down for maximum turn rate,
/// forces overshoots, then punishes. Beats Ace (jinks disrupt dive windows),
/// loses to Chaser (constant pressure prevents settling into turn fight).
pub struct BrawlerPolicy {
    config: SimConfig,
    phase: BrawlerPhase,
    phase_timer: u32,
    jink_timer: u32,
    jink_dir: f32,
    overshoot_timer: u32,
}

impl BrawlerPolicy {
    pub fn new() -> Self {
        Self::with_config(SimConfig::default())
    }

    pub fn with_config(config: SimConfig) -> Self {
        Self {
            config,
            phase: BrawlerPhase::Close,
            phase_timer: 0,
            jink_timer: 0,
            jink_dir: 1.0,
            overshoot_timer: 0,
        }
    }
}

impl Default for BrawlerPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Policy for BrawlerPolicy {
    fn name(&self) -> &str {
        "brawler"
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
        if self.phase_timer > 0 {
            self.phase_timer -= 1;
        }
        if self.jink_timer > 0 {
            self.jink_timer -= 1;
        }
        if self.overshoot_timer > 0 {
            self.overshoot_timer -= 1;
        }

        // Short defensive jinks when enemy bullets are close
        if ts.nearest_enemy_bullet_dist < 100.0 && self.jink_timer == 0 {
            self.jink_timer = 8; // brief jink — don't disrupt turn fight
            self.jink_dir = -self.jink_dir;
        }

        // Phase transitions
        self.update_phase(&ts);

        // Execute current phase
        match self.phase {
            BrawlerPhase::Close => self.act_close(&ts),
            BrawlerPhase::Brawl => self.act_brawl(&ts),
            BrawlerPhase::OvershootBait => self.act_overshoot_bait(&ts),
            BrawlerPhase::OvershootPunish => self.act_overshoot_punish(&ts),
            BrawlerPhase::Retreat => self.act_retreat(&ts),
        }
    }
}

impl BrawlerPolicy {
    fn update_phase(&mut self, ts: &TacticalState) {
        // Don't switch too fast (hysteresis)
        if self.phase_timer > 0 {
            return;
        }

        let new_phase = match self.phase {
            BrawlerPhase::Close => {
                if ts.distance < 200.0 {
                    Some(BrawlerPhase::Brawl)
                } else {
                    None
                }
            }
            BrawlerPhase::Brawl => {
                if ts.distance > 300.0 {
                    Some(BrawlerPhase::Close)
                } else if ts.opponent_behind_me && ts.distance < 180.0 && ts.closing_rate > 30.0 {
                    Some(BrawlerPhase::OvershootBait)
                } else if ts.altitude < 80.0 && ts.my_speed < 60.0 {
                    Some(BrawlerPhase::Retreat)
                } else {
                    None
                }
            }
            BrawlerPhase::OvershootBait => {
                if ts.am_behind_opponent || (ts.angle_off_nose.abs() < 1.0 && ts.distance < 200.0) {
                    Some(BrawlerPhase::OvershootPunish)
                } else if ts.distance > 250.0 {
                    Some(BrawlerPhase::Close)
                } else if !ts.opponent_behind_me {
                    Some(BrawlerPhase::Brawl)
                } else {
                    None
                }
            }
            BrawlerPhase::OvershootPunish => {
                if self.overshoot_timer == 0 {
                    Some(BrawlerPhase::Brawl)
                } else {
                    None
                }
            }
            BrawlerPhase::Retreat => {
                if ts.altitude > 180.0 && ts.my_speed > 80.0 {
                    Some(BrawlerPhase::Close)
                } else {
                    None
                }
            }
        };

        if let Some(phase) = new_phase {
            self.phase = phase;
            self.phase_timer = 20; // min ticks before next transition
            if phase == BrawlerPhase::OvershootPunish {
                self.overshoot_timer = 60; // punish window
            }
        }
    }

    fn act_close(&self, ts: &TacticalState) -> Action {
        let desired_yaw = smart_aim(ts, &self.config, 1.0);
        let mut yaw_input = yaw_toward(desired_yaw, ts.my_yaw, 3.0);

        if ts.altitude < 120.0 && ts.my_yaw.sin() < 0.0 {
            yaw_input += 0.3;
            yaw_input = yaw_input.clamp(-1.0, 1.0);
        }

        let shoot = can_shoot(ts, 0.25, 300.0);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);
        let throttle = 1.0f32.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot,
        }
    }

    fn act_brawl(&self, ts: &TacticalState) -> Action {
        let desired_yaw = smart_aim(ts, &self.config, 0.7);
        let mut yaw_input = yaw_toward(desired_yaw, ts.my_yaw, 4.0);

        if self.jink_timer > 0 {
            yaw_input = self.jink_dir;
        }

        let throttle: f32 = if ts.my_speed > 120.0 {
            0.0
        } else if ts.my_speed < 70.0 {
            0.6
        } else {
            0.2
        };

        let shoot = can_shoot(ts, 0.30, 250.0);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);
        let throttle = throttle.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot,
        }
    }

    fn act_overshoot_bait(&self, ts: &TacticalState) -> Action {
        let perp_yaw = ts.angle_to_opp + std::f32::consts::FRAC_PI_2 * self.jink_dir;
        let yaw_input = yaw_toward(perp_yaw, ts.my_yaw, 2.0);

        let shoot = can_shoot(ts, 0.35, 200.0);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);

        Action {
            yaw_input,
            throttle: min_throttle,
            shoot,
        }
    }

    fn act_overshoot_punish(&self, ts: &TacticalState) -> Action {
        let desired_yaw = smart_aim(ts, &self.config, 0.8);
        let yaw_input = yaw_toward(desired_yaw, ts.my_yaw, 4.0);

        let shoot = can_shoot(ts, 0.35, 300.0);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);
        let throttle = 0.5f32.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot,
        }
    }

    fn act_retreat(&self, ts: &TacticalState) -> Action {
        let climb_yaw = if ts.my_yaw.cos() > 0.0 { 0.5 } else { std::f32::consts::PI - 0.5 };
        let yaw_input = yaw_toward(climb_yaw, ts.my_yaw, 2.0);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);
        let throttle = 1.0f32.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::SimState;

    #[test]
    fn test_brawler_produces_actions() {
        let state = SimState::new();
        let obs = state.observe(0);
        let mut brawler = BrawlerPolicy::new();
        let action = brawler.act(&obs);

        assert!(action.yaw_input >= -1.0 && action.yaw_input <= 1.0);
        assert!(action.throttle >= 0.0 && action.throttle <= 1.0);
    }

    #[test]
    fn test_brawler_starts_in_close_phase() {
        let brawler = BrawlerPolicy::new();
        assert_eq!(brawler.phase, BrawlerPhase::Close);
    }
}
