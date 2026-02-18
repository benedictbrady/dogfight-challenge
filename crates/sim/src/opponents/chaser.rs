use dogfight_shared::*;
use crate::policy::Policy;
use std::f32::consts::PI;

/// Easy opponent: partial lead pursuit with altitude management and range-limited shooting.
/// Still no evasion, no throttle management, no energy management.
pub struct ChaserPolicy;

impl Policy for ChaserPolicy {
    fn name(&self) -> &str {
        "chaser"
    }

    fn act(&mut self, obs: &Observation) -> Action {
        let d = &obs.data;

        // Extract self state
        let my_yaw = f32::atan2(d[2], d[1]);
        let altitude = d[5] * MAX_ALTITUDE;

        // Extract opponent relative position (un-normalize)
        let rel_x = d[6] * ARENA_DIAMETER;
        let rel_y = d[7] * ARENA_DIAMETER;
        let opp_speed = d[8] * MAX_SPEED;
        let opp_yaw = f32::atan2(d[10], d[9]);
        let distance = d[12] * ARENA_DIAMETER;

        // Emergency ground avoidance: pull up hard when low and diving
        let heading_down = my_yaw.sin() < -0.3;
        if altitude < 80.0 && heading_down {
            let pull_up_yaw = if my_yaw.cos() > 0.0 { 1.0 } else { -1.0 };
            return Action {
                yaw_input: pull_up_yaw,
                throttle: 1.0,
                shoot: false,
            };
        }

        // Partial lead prediction (30%)
        let time_to_target = distance / BULLET_SPEED;
        let opp_fwd_x = opp_yaw.cos();
        let opp_fwd_y = opp_yaw.sin();
        let lead_x = rel_x + opp_fwd_x * opp_speed * time_to_target * 0.3;
        let lead_y = rel_y + opp_fwd_y * opp_speed * time_to_target * 0.3;

        // Compute desired yaw to predicted position
        let desired_yaw = f32::atan2(lead_y, lead_x);
        let mut yaw_diff = angle_diff(desired_yaw, my_yaw);

        // Basic altitude management: bias toward mid-altitude
        // Nudge upward when below 150m, nudge downward when above 500m
        if altitude < 150.0 && my_yaw.sin() < 0.1 {
            yaw_diff += 0.2;
        } else if altitude > 500.0 && my_yaw.sin() > -0.1 {
            yaw_diff -= 0.2;
        }

        // Convert to control input [-1, 1]
        let yaw_input = (yaw_diff * 2.0).clamp(-1.0, 1.0);

        // Range-limited shooting: only shoot within 350m and roughly aimed
        let shoot = yaw_diff.abs() < 0.26 && distance < 350.0;
        let gun_ready = d[4] < 0.01;

        Action {
            yaw_input,
            throttle: 1.0,
            shoot: shoot && gun_ready,
        }
    }
}

/// Compute shortest angular difference.
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
    fn test_chaser_produces_actions() {
        let state = SimState::new();
        let obs = state.observe(0);
        let mut chaser = ChaserPolicy;
        let action = chaser.act(&obs);

        // Player faces east (yaw=0), opponent is east → yaw_diff ≈ 0
        assert!(action.yaw_input.abs() < 0.5);
        assert_eq!(action.throttle, 1.0);
    }
}
