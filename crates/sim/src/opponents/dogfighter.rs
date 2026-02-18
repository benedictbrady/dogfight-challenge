use dogfight_shared::*;
use crate::policy::Policy;
use std::f32::consts::PI;

/// Medium opponent: lead pursuit with evasion, altitude management, and throttle control.
pub struct DogfighterPolicy {
    evade_timer: u32,
    evade_dir: f32,
}

impl DogfighterPolicy {
    pub fn new() -> Self {
        Self {
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
        let d = &obs.data;

        // Extract self state
        let my_yaw = f32::atan2(d[2], d[1]);
        let my_speed = d[0] * MAX_SPEED;
        let gun_cooldown = d[4];
        let altitude = d[5] * MAX_ALTITUDE;

        // Opponent relative position
        let rel_x = d[6] * ARENA_DIAMETER;
        let rel_y = d[7] * ARENA_DIAMETER;
        let opp_speed = d[8] * MAX_SPEED;
        let opp_yaw = f32::atan2(d[10], d[9]);
        let distance = d[12] * ARENA_DIAMETER;

        // Check for nearby enemy bullets (evasion)
        let bullet_threat = detect_bullet_threat(d);

        // Tick evade timer
        if self.evade_timer > 0 {
            self.evade_timer -= 1;
        }

        if bullet_threat && self.evade_timer == 0 {
            self.evade_timer = 20;
            self.evade_dir = -self.evade_dir;
        }

        // Evasion mode — can still fire opportunistic shots
        if self.evade_timer > 0 {
            let desired_yaw = f32::atan2(rel_y, rel_x);
            let yaw_diff = angle_diff(desired_yaw, my_yaw);
            let can_shoot = yaw_diff.abs() < 0.3 && distance < 350.0 && gun_cooldown < 0.01;

            // Even while evading, respect altitude limits
            let evade_yaw = if altitude < 80.0 && self.evade_dir < 0.0 && my_yaw.sin() < 0.0 {
                // Don't evade downward near ground
                1.0
            } else {
                self.evade_dir
            };

            return Action {
                yaw_input: evade_yaw,
                throttle: 1.0,
                shoot: can_shoot,
            };
        }

        // Emergency ground avoidance
        let heading_down = my_yaw.sin() < -0.5;
        if altitude < 60.0 && heading_down {
            let pull_up_yaw = if my_yaw.cos() > 0.0 { 1.0 } else { -1.0 };
            return Action {
                yaw_input: pull_up_yaw,
                throttle: 1.0,
                shoot: false,
            };
        }

        // Lead pursuit: predict where opponent will be
        let opp_forward_x = opp_yaw.cos();
        let opp_forward_y = opp_yaw.sin();
        let time_to_target = distance / BULLET_SPEED;
        let lead_x = rel_x + opp_forward_x * opp_speed * time_to_target;
        let lead_y = rel_y + opp_forward_y * opp_speed * time_to_target;

        // Desired angle to lead position
        let desired_yaw = f32::atan2(lead_y, lead_x);
        let mut yaw_diff = angle_diff(desired_yaw, my_yaw);

        // Altitude management: keep planes fighting in mid-altitude band
        yaw_diff += altitude_bias(altitude, my_yaw, my_speed);

        // Yaw input with gain
        let yaw_input = (yaw_diff * 3.0).clamp(-1.0, 1.0);

        // Throttle management: reduce speed before hard turns for tighter turning
        let turn_magnitude = yaw_input.abs();
        let throttle = if my_speed < 80.0 {
            1.0
        } else if turn_magnitude > 0.7 {
            0.6
        } else if distance > 250.0 {
            1.0
        } else {
            0.8
        };

        // Shoot when well-aimed and in range
        let well_aimed = yaw_diff.abs() < 0.25;
        let in_range = distance < 350.0;
        let shoot = well_aimed && in_range && gun_cooldown < 0.01;

        Action {
            yaw_input,
            throttle,
            shoot,
        }
    }
}

/// Altitude bias: yaw nudge to keep planes fighting at mid-altitude.
/// More aggressive than chaser — this is a key advantage.
fn altitude_bias(altitude: f32, yaw: f32, speed: f32) -> f32 {
    let mut bias = 0.0;

    // Strong push away from ceiling
    if altitude > ALT_BOUNDARY_HIGH {
        let urgency = ((altitude - ALT_BOUNDARY_HIGH) / 50.0).min(1.0);
        if yaw.sin() > 0.0 {
            bias -= urgency * 0.3;
        }
    } else if altitude > 450.0 {
        // Gentle nudge away from ceiling zone
        if yaw.sin() > 0.3 {
            bias -= 0.1;
        }
    }

    // Strong push away from ground
    if altitude < 100.0 {
        let urgency = ((100.0 - altitude) / 60.0).min(1.0);
        if yaw.sin() < 0.1 {
            bias += urgency * 0.4;
        }
    } else if altitude < 180.0 {
        // Gentle nudge upward in low zone
        if yaw.sin() < -0.2 {
            bias += 0.15;
        }
    }

    // Dive when slow to gain speed via gravity (only if safely above ground)
    if speed < 100.0 && altitude > 250.0 {
        if yaw.sin() > -0.2 {
            bias -= 0.08;
        }
    }

    bias
}

/// Check if any nearby enemy bullet is heading toward us.
fn detect_bullet_threat(obs: &[f32; OBS_SIZE]) -> bool {
    for slot in 0..MAX_BULLET_SLOTS {
        let base = 13 + slot * 4;
        let is_friendly = obs[base + 2];

        // Only care about enemy bullets
        if is_friendly > 0.5 {
            continue;
        }

        // Check if bullet exists (non-zero position)
        let bx = obs[base];
        let by = obs[base + 1];
        if bx == 0.0 && by == 0.0 {
            continue;
        }

        // Check distance (already relative, normalized by ARENA_DIAMETER)
        let dist = (bx * bx + by * by).sqrt() * ARENA_DIAMETER;
        if dist > 100.0 {
            continue;
        }

        return true;
    }
    false
}

fn angle_diff(target: f32, current: f32) -> f32 {
    let mut diff = target - current;
    while diff > PI {
        diff -= 2.0 * PI;
    }
    while diff < -PI {
        diff += 2.0 * PI;
    }
    diff
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::SimState;

    #[test]
    fn test_dogfighter_produces_actions() {
        let state = SimState::new();
        let obs = state.observe(0);
        let mut df = DogfighterPolicy::new();
        let action = df.act(&obs);

        assert!(action.yaw_input >= -1.0 && action.yaw_input <= 1.0);
        assert!(action.throttle >= 0.0 && action.throttle <= 1.0);
    }
}
