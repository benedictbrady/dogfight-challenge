use dogfight_shared::*;
use crate::policy::Policy;
use super::tactics::*;

/// Defensive energy fighter. Lead pursuit with strong evasion and altitude
/// management. When under pressure, uses perpendicular break turns (like
/// a defensive dogfighter) while continuing to fire — never runs away.
///
/// Beats Chaser: stronger evasion + altitude advantage. Chaser's pursuit
/// forces it to climb after ace, bleeding speed. Ace's defensive breaks
/// create angular offset that disrupts chaser's lead pursuit.
///
/// Loses to Brawler: brawler's slow speed + close range means evasion
/// has less effect (short bullet travel time). Brawler doesn't trigger
/// ace's defensive breaks as aggressively.
pub struct AcePolicy {
    evade_timer: u32,
    evade_dir: f32,
}

impl AcePolicy {
    pub fn new() -> Self {
        Self {
            evade_timer: 0,
            evade_dir: 1.0,
        }
    }
}

impl Default for AcePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Policy for AcePolicy {
    fn name(&self) -> &str {
        "ace"
    }

    fn act(&mut self, obs: &Observation) -> Action {
        let ts = extract_tactical_state(obs);

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

        // Strong bullet evasion: 18-tick dodge triggered at 130m
        if ts.nearest_enemy_bullet_dist < 130.0 && self.evade_timer == 0 {
            self.evade_timer = 18;
            self.evade_dir = -self.evade_dir;
        }

        // Evasion mode: hard turn with opportunistic shots
        if self.evade_timer > 0 {
            return self.act_evade(&ts);
        }

        // Defensive mode: when opponent is behind at close range
        if ts.opponent_behind_me && ts.distance < 250.0 {
            return self.act_defend(&ts);
        }

        // Standard pursuit mode
        self.act_pursue(&ts)
    }
}

impl AcePolicy {
    /// Evasion: hard dodge, altitude-safe, keep shooting
    fn act_evade(&self, ts: &TacticalState) -> Action {
        let mut evade_yaw = self.evade_dir;

        // Don't evade into ground
        if ts.altitude < 90.0 && evade_yaw < 0.0 && ts.my_yaw.sin() < 0.0 {
            evade_yaw = 1.0;
        }
        // Don't evade into ceiling
        if ts.altitude > 540.0 && evade_yaw > 0.0 && ts.my_yaw.sin() > 0.0 {
            evade_yaw = -1.0;
        }

        let can_shoot = ts.angle_off_nose.abs() < 0.30
            && ts.distance < 380.0
            && ts.gun_cooldown < 0.01;

        Action {
            yaw_input: evade_yaw,
            throttle: 0.7,
            shoot: can_shoot,
        }
    }

    /// Defensive break: turn perpendicular to opponent (stay in fight, don't run)
    fn act_defend(&self, ts: &TacticalState) -> Action {
        // Break perpendicular to opponent's bearing
        let break_dir = if ts.angle_off_nose > 0.0 { 1.0 } else { -1.0 };
        let perp_yaw = ts.angle_to_opp + std::f32::consts::FRAC_PI_2 * break_dir;
        // Slight upward bias during defense to maintain altitude
        let target_yaw = perp_yaw + 0.15;
        let yaw_input = yaw_toward(target_yaw, ts.my_yaw, 3.5);

        // Slow down for tighter defensive turns
        let throttle = if ts.my_speed > 100.0 { 0.3 } else { 0.5 };

        // Wide shoot angle during defense — take any shot opportunity
        let can_shoot = ts.angle_off_nose.abs() < 0.35
            && ts.distance < 350.0
            && ts.gun_cooldown < 0.01;

        Action {
            yaw_input,
            throttle,
            shoot: can_shoot,
        }
    }

    /// Standard pursuit: lead aim with altitude bias
    fn act_pursue(&self, ts: &TacticalState) -> Action {
        let desired_yaw = lead_aim(
            ts.rel_x, ts.rel_y, ts.opp_speed, ts.opp_yaw, ts.distance, 1.0,
        );
        let mut yaw_diff = angle_diff(desired_yaw, ts.my_yaw);

        // Altitude management: prefer 320-460m band (higher than other policies)
        if ts.altitude < 250.0 {
            let urgency = ((250.0 - ts.altitude) / 100.0).min(1.0);
            if ts.my_yaw.sin() < 0.2 {
                yaw_diff += urgency * 0.35;
            }
        } else if ts.altitude > 480.0 {
            let urgency = ((ts.altitude - 480.0) / 80.0).min(1.0);
            if ts.my_yaw.sin() > -0.2 {
                yaw_diff -= urgency * 0.3;
            }
        }

        // When slow, bias toward diving for speed (above 300m)
        if ts.my_speed < 90.0 && ts.altitude > 300.0 && ts.my_yaw.sin() > -0.1 {
            yaw_diff -= 0.08;
        }

        let yaw_input = (yaw_diff * 3.0).clamp(-1.0, 1.0);

        // Throttle management
        let throttle = if ts.my_speed < 80.0 {
            1.0
        } else if yaw_input.abs() > 0.7 {
            0.5
        } else if ts.distance > 250.0 {
            1.0
        } else {
            0.7
        };

        // Tight shooting — fewer but more accurate shots
        let shoot = ts.angle_off_nose.abs() < 0.20
            && ts.distance < 380.0
            && ts.gun_cooldown < 0.01;

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
    fn test_ace_produces_actions() {
        let state = SimState::new();
        let obs = state.observe(0);
        let mut ace = AcePolicy::new();
        let action = ace.act(&obs);

        assert!(action.yaw_input >= -1.0 && action.yaw_input <= 1.0);
        assert!(action.throttle >= 0.0 && action.throttle <= 1.0);
    }
}
