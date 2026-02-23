use dogfight_shared::*;
use crate::policy::Policy;
use super::tactics::*;
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
enum DogfighterMode {
    /// Lead pursuit, throttle management, shoot when aimed
    Attack,
    /// Break turn perpendicular, tight turns to shake pursuer
    Defend,
    /// Climb to build altitude/speed when at energy disadvantage
    Energy,
    /// Turn away, full throttle, break off bad engagement
    Disengage,
}

/// Adaptive mode-switching fighter. Reads the situation and picks the right
/// response. Jack-of-all-trades — ~50% vs each specialist.
pub struct DogfighterPolicy {
    config: SimConfig,
    mode: DogfighterMode,
    mode_timer: u32,
    attack_patience: u32,
    last_distance: f32,
    evade_timer: u32,
    evade_dir: f32,
}

impl DogfighterPolicy {
    pub fn new() -> Self {
        Self::with_config(SimConfig::default())
    }

    pub fn with_config(config: SimConfig) -> Self {
        Self {
            config,
            mode: DogfighterMode::Attack,
            mode_timer: 0,
            attack_patience: 0,
            last_distance: 400.0,
            evade_timer: 0,
            evade_dir: 1.0,
        }
    }
}

impl Default for DogfighterPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Policy for DogfighterPolicy {
    fn name(&self) -> &str {
        "dogfighter"
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
        if self.mode_timer > 0 {
            self.mode_timer -= 1;
        }
        if self.evade_timer > 0 {
            self.evade_timer -= 1;
        }

        // Bullet evasion (active in all modes)
        if ts.nearest_enemy_bullet_dist < 90.0 && self.evade_timer == 0 {
            self.evade_timer = 18;
            self.evade_dir = -self.evade_dir;
        }

        if self.evade_timer > 0 {
            return self.act_evade(&ts);
        }

        // Mode switching
        self.update_mode(&ts);

        // Track attack patience
        if self.mode == DogfighterMode::Attack {
            self.attack_patience += 1;
        } else {
            self.attack_patience = 0;
        }

        self.last_distance = ts.distance;

        match self.mode {
            DogfighterMode::Attack => self.act_attack(&ts),
            DogfighterMode::Defend => self.act_defend(&ts),
            DogfighterMode::Energy => self.act_energy(&ts),
            DogfighterMode::Disengage => self.act_disengage(&ts),
        }
    }
}

impl DogfighterPolicy {
    fn update_mode(&mut self, ts: &TacticalState) {
        if self.mode_timer > 0 {
            return;
        }

        let new_mode = match self.mode {
            DogfighterMode::Attack => {
                if ts.opponent_behind_me && ts.distance < 200.0 {
                    Some(DogfighterMode::Defend)
                } else if ts.energy_advantage < 0.5 && ts.altitude > 100.0 {
                    // Only disengage for energy when at serious disadvantage
                    Some(DogfighterMode::Energy)
                } else {
                    None
                }
            }
            DogfighterMode::Defend => {
                if !ts.opponent_behind_me || ts.distance > 300.0 {
                    Some(DogfighterMode::Attack)
                } else if ts.energy_advantage < 0.4 {
                    Some(DogfighterMode::Energy)
                } else {
                    None
                }
            }
            DogfighterMode::Energy => {
                // Return to attack quickly — don't lap around passively
                if ts.energy_advantage > 0.7 || ts.distance < 200.0 {
                    Some(DogfighterMode::Attack)
                } else if ts.opponent_behind_me && ts.distance < 150.0 {
                    Some(DogfighterMode::Defend)
                } else {
                    None
                }
            }
            // Disengage removed — dogfighter should always be fighting
            DogfighterMode::Disengage => {
                Some(DogfighterMode::Attack)
            }
        };

        if let Some(mode) = new_mode {
            self.mode = mode;
            // Hysteresis: different cooldowns per mode
            self.mode_timer = match mode {
                DogfighterMode::Attack => 30,
                DogfighterMode::Defend => 45,
                DogfighterMode::Energy => 30,
                DogfighterMode::Disengage => 0, // immediately transition out
            };
        }
    }

    fn act_evade(&self, ts: &TacticalState) -> Action {
        let shoot = can_shoot(ts, 0.3, 350.0);

        let evading_downward = self.evade_dir * ts.my_yaw.cos() < 0.0;
        let evade_yaw = if ts.altitude < 80.0 && evading_downward && ts.my_yaw.sin() < 0.1 {
            if ts.my_yaw.cos() > 0.0 { 1.0 } else { -1.0 }
        } else {
            self.evade_dir
        };

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, evade_yaw);
        let throttle = 0.8f32.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot,
        }
    }

    fn act_attack(&self, ts: &TacticalState) -> Action {
        let desired_yaw = smart_aim(ts, &self.config, 1.0);
        let mut yaw_diff = angle_diff(desired_yaw, ts.my_yaw);

        yaw_diff += altitude_bias(ts.altitude, ts.my_yaw, ts.my_speed);

        let yaw_input = (yaw_diff * 3.0).clamp(-1.0, 1.0);

        let throttle: f32 = if ts.my_speed < 80.0 {
            1.0
        } else if yaw_input.abs() > 0.7 {
            0.5
        } else if ts.distance > 250.0 {
            1.0
        } else {
            0.7
        };

        let shoot = can_shoot(ts, 0.22, 350.0);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);
        let throttle = throttle.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot,
        }
    }

    fn act_defend(&self, ts: &TacticalState) -> Action {
        let break_dir = if ts.angle_off_nose > 0.0 { 1.0 } else { -1.0 };
        let perp_yaw = ts.angle_to_opp + std::f32::consts::FRAC_PI_2 * break_dir;
        let yaw_input = yaw_toward(perp_yaw, ts.my_yaw, 3.5);

        let throttle: f32 = if ts.my_speed > 100.0 { 0.3 } else { 0.6 };

        let shoot = can_shoot(ts, 0.3, 250.0);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);
        let throttle = throttle.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot,
        }
    }

    fn act_energy(&self, ts: &TacticalState) -> Action {
        let desired_yaw = if ts.altitude < 400.0 {
            let away_x = -ts.rel_x;
            let climb_angle = f32::atan2(0.5, away_x.signum());
            if away_x > 0.0 { climb_angle } else { PI - climb_angle }
        } else {
            if ts.my_yaw.cos() > 0.0 { 0.0 } else { PI }
        };

        let yaw_input = yaw_toward(desired_yaw, ts.my_yaw, 2.0);

        let shoot = can_shoot(ts, 0.25, 300.0);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);
        let throttle = 1.0f32.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot,
        }
    }

    fn act_disengage(&self, ts: &TacticalState) -> Action {
        let away_yaw = f32::atan2(-ts.rel_y, -ts.rel_x);
        let yaw_input = yaw_toward(away_yaw, ts.my_yaw, 2.5);

        let (yaw_input, min_throttle) = stall_avoidance(ts.my_speed, yaw_input);
        let throttle = 1.0f32.max(min_throttle);

        Action {
            yaw_input,
            throttle,
            shoot: false,
        }
    }
}

/// Altitude bias nudge for attack mode (kept from original dogfighter).
fn altitude_bias(altitude: f32, yaw: f32, speed: f32) -> f32 {
    let mut bias = 0.0;

    if altitude > ALT_BOUNDARY_HIGH {
        let urgency = ((altitude - ALT_BOUNDARY_HIGH) / 50.0).min(1.0);
        if yaw.sin() > 0.0 {
            bias -= urgency * 0.3;
        }
    } else if altitude > 450.0 && yaw.sin() > 0.3 {
        bias -= 0.1;
    }

    if altitude < 130.0 {
        let urgency = ((130.0 - altitude) / 60.0).min(1.0);
        if yaw.sin() < 0.1 {
            bias += urgency * 0.4;
        }
    } else if altitude < 180.0 && yaw.sin() < -0.2 {
        bias += 0.15;
    }

    if speed < 100.0 && altitude > 250.0 && yaw.sin() > -0.2 {
        bias -= 0.08;
    }

    bias
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::SimState;

    #[test]
    fn test_dogfighter_produces_actions() {
        let mut state = SimState::new();
        let obs = state.observe(0);
        let mut df = DogfighterPolicy::new();
        let action = df.act(&obs);

        assert!(action.yaw_input >= -1.0 && action.yaw_input <= 1.0);
        assert!(action.throttle >= 0.0 && action.throttle <= 1.0);
    }
}
