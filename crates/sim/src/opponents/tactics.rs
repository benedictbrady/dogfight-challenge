use dogfight_shared::*;
use std::f32::consts::PI;

/// Tactical state derived from the raw 46-float observation.
/// Provides high-level situational awareness for all policies.
pub struct TacticalState {
    // Self state
    pub my_speed: f32,
    pub my_yaw: f32,
    pub my_hp: f32,
    pub gun_cooldown: f32,
    pub altitude: f32,

    // Opponent state
    pub rel_x: f32,
    pub rel_y: f32,
    pub opp_speed: f32,
    pub opp_yaw: f32,
    pub opp_hp: f32,
    pub distance: f32,

    // Derived angles
    /// Absolute angle from me to opponent
    pub angle_to_opp: f32,
    /// How far off my nose the opponent is (signed)
    pub angle_off_nose: f32,
    /// How far off opponent's nose I am (signed, from their perspective)
    pub opp_angle_to_me: f32,

    // Situation
    /// Rate at which distance is decreasing (positive = closing)
    pub closing_rate: f32,
    /// My energy proxy: v^2 + 2*g*h
    pub my_energy: f32,
    /// Opponent energy proxy
    pub opp_energy: f32,
    /// my_energy / opp_energy ratio (>1 = I have advantage)
    pub energy_advantage: f32,
    /// My altitude minus opponent altitude
    pub altitude_advantage: f32,
    /// True if I'm roughly behind the opponent (they can't easily aim at me)
    pub am_behind_opponent: bool,
    /// True if opponent is roughly behind me (they can aim at me)
    pub opponent_behind_me: bool,

    // Bullet threats
    pub nearest_enemy_bullet_dist: f32,
    pub nearest_enemy_bullet_angle: f32,
    pub enemy_bullet_threat_count: u32,

    // Time
    pub ticks_remaining_frac: f32,
}

/// Extract tactical state from raw observation.
pub fn extract_tactical_state(obs: &Observation) -> TacticalState {
    let d = &obs.data;

    // Self state
    let my_speed = d[0] * MAX_SPEED;
    let my_yaw = f32::atan2(d[2], d[1]);
    let my_hp = d[3];
    let gun_cooldown = d[4];
    let altitude = d[5] * MAX_ALTITUDE;

    // Opponent state
    let rel_x = d[6] * ARENA_DIAMETER;
    let rel_y = d[7] * ARENA_DIAMETER;
    let opp_speed = d[8] * MAX_SPEED;
    let opp_yaw = f32::atan2(d[10], d[9]);
    let opp_hp = d[11];
    let distance = d[12] * ARENA_DIAMETER;

    // Derived angles
    let angle_to_opp = f32::atan2(rel_y, rel_x);
    let angle_off_nose = angle_diff(angle_to_opp, my_yaw);
    // Angle from opponent to me (opposite direction)
    let angle_opp_to_me = f32::atan2(-rel_y, -rel_x);
    let opp_angle_to_me = angle_diff(angle_opp_to_me, opp_yaw);

    // Closing rate: projection of relative velocity onto the line between us
    let my_vx = my_speed * my_yaw.cos();
    let my_vy = my_speed * my_yaw.sin();
    let opp_vx = opp_speed * opp_yaw.cos();
    let opp_vy = opp_speed * opp_yaw.sin();
    let closing_rate = if distance > 1.0 {
        (rel_x * (my_vx - opp_vx) + rel_y * (my_vy - opp_vy)) / distance
    } else {
        0.0
    };

    // Energy: v^2 + 2*g*h (simplified energy proxy)
    let my_energy = my_speed * my_speed + 2.0 * GRAVITY * altitude;
    let opp_altitude = altitude + rel_y;
    let opp_energy = opp_speed * opp_speed + 2.0 * GRAVITY * opp_altitude;
    let energy_advantage = if opp_energy > 1.0 {
        my_energy / opp_energy
    } else {
        2.0
    };
    let altitude_advantage = altitude - opp_altitude;

    // Behind checks: I'm behind opponent if they'd need to turn > 120 deg to face me
    let am_behind_opponent = opp_angle_to_me.abs() > 2.0;
    // Opponent is behind me if I'd need to turn > 120 deg to face them
    let opponent_behind_me = angle_off_nose.abs() > 2.0;

    // Bullet threats
    let (nearest_enemy_bullet_dist, nearest_enemy_bullet_angle, enemy_bullet_threat_count) =
        compute_bullet_threats(d);

    let ticks_remaining_frac = d[45];

    TacticalState {
        my_speed,
        my_yaw,
        my_hp,
        gun_cooldown,
        altitude,
        rel_x,
        rel_y,
        opp_speed,
        opp_yaw,
        opp_hp,
        distance,
        angle_to_opp,
        angle_off_nose,
        opp_angle_to_me,
        closing_rate,
        my_energy,
        opp_energy,
        energy_advantage,
        altitude_advantage,
        am_behind_opponent,
        opponent_behind_me,
        nearest_enemy_bullet_dist,
        nearest_enemy_bullet_angle,
        enemy_bullet_threat_count,
        ticks_remaining_frac,
    }
}

/// Compute lead aim: predict where opponent will be and return desired yaw.
pub fn lead_aim(
    rel_x: f32,
    rel_y: f32,
    opp_speed: f32,
    opp_yaw: f32,
    distance: f32,
    lead_factor: f32,
) -> f32 {
    let time_to_target = distance / BULLET_SPEED;
    let opp_fwd_x = opp_yaw.cos();
    let opp_fwd_y = opp_yaw.sin();
    let lead_x = rel_x + opp_fwd_x * opp_speed * time_to_target * lead_factor;
    let lead_y = rel_y + opp_fwd_y * opp_speed * time_to_target * lead_factor;
    f32::atan2(lead_y, lead_x)
}

/// Shortest angular difference (signed), result in [-PI, PI].
pub fn angle_diff(target: f32, current: f32) -> f32 {
    let mut diff = target - current;
    while diff > PI {
        diff -= 2.0 * PI;
    }
    while diff < -PI {
        diff += 2.0 * PI;
    }
    diff
}

/// Compute yaw_input to steer toward a desired yaw, with configurable gain.
/// Returns clamped [-1, 1].
pub fn yaw_toward(desired: f32, current: f32, gain: f32) -> f32 {
    let diff = angle_diff(desired, current);
    (diff * gain).clamp(-1.0, 1.0)
}

/// Emergency altitude safety override. Returns Some(yaw_input) if an emergency
/// pull-up or push-down is needed, None otherwise.
pub fn altitude_safety(altitude: f32, yaw: f32) -> Option<f32> {
    // Emergency ground avoidance
    if altitude < 70.0 && yaw.sin() < -0.3 {
        let pull_up = if yaw.cos() > 0.0 { 1.0 } else { -1.0 };
        return Some(pull_up);
    }
    // Emergency ceiling avoidance
    if altitude > 570.0 && yaw.sin() > 0.3 {
        let push_down = if yaw.cos() > 0.0 { -1.0 } else { 1.0 };
        return Some(push_down);
    }
    None
}

/// Compute bullet threat info from observation.
/// Returns (nearest_enemy_dist, nearest_enemy_angle, threat_count_within_150m).
pub fn compute_bullet_threats(obs: &[f32; OBS_SIZE]) -> (f32, f32, u32) {
    let mut nearest_dist = f32::MAX;
    let mut nearest_angle = 0.0f32;
    let mut threat_count = 0u32;

    for slot in 0..MAX_BULLET_SLOTS {
        let base = 13 + slot * 4;
        let is_friendly = obs[base + 2];

        // Only care about enemy bullets
        if is_friendly > 0.5 {
            continue;
        }

        let bx = obs[base];
        let by = obs[base + 1];
        if bx == 0.0 && by == 0.0 {
            continue;
        }

        let dist = (bx * bx + by * by).sqrt() * ARENA_DIAMETER;
        let angle = obs[base + 3] * PI;

        if dist < 150.0 {
            threat_count += 1;
        }

        if dist < nearest_dist {
            nearest_dist = dist;
            nearest_angle = angle;
        }
    }

    (nearest_dist, nearest_angle, threat_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_diff() {
        assert!((angle_diff(0.0, 0.0)).abs() < 0.001);
        assert!((angle_diff(PI, 0.0) - PI).abs() < 0.001);
        assert!((angle_diff(0.0, PI) - (-PI)).abs() < 0.001);
        // Wrapping: 3.0 - (-3.0) should give ~-0.28 not ~6.0
        let diff = angle_diff(3.0, -3.0);
        assert!(diff.abs() < PI + 0.001);
    }

    #[test]
    fn test_yaw_toward() {
        let y = yaw_toward(0.5, 0.0, 2.0);
        assert!(y > 0.0 && y <= 1.0);

        let y = yaw_toward(-0.5, 0.0, 2.0);
        assert!(y < 0.0 && y >= -1.0);
    }

    #[test]
    fn test_altitude_safety_ground() {
        // Low altitude, diving — should get pull-up
        let result = altitude_safety(50.0, -1.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_altitude_safety_safe() {
        // Mid altitude, level — no override
        let result = altitude_safety(300.0, 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_tactical_state() {
        use crate::physics::SimState;
        let state = SimState::new();
        let obs = state.observe(0);
        let ts = extract_tactical_state(&obs);

        // P0 at (-200,300), P1 at (200,300) — distance should be ~400
        assert!((ts.distance - 400.0).abs() < 10.0);
        // Altitude should be ~300
        assert!((ts.altitude - 300.0).abs() < 10.0);
        // Speed should be MIN_SPEED
        assert!((ts.my_speed - MIN_SPEED).abs() < 1.0);
    }
}
