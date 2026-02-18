use dogfight_shared::*;

/// Aggregate metrics quantifying battle quality and dynamism.
#[derive(Debug, Clone)]
pub struct BattleMetrics {
    /// Fraction of time both fighters are in sustained same-direction turning.
    pub circling_index: f32,
    /// Standard deviation of fighter speeds across the match.
    pub speed_variance: f32,
    /// Max altitude minus min altitude used by fighters.
    pub altitude_range: f32,
    /// Number of distance threshold crossings (close <-> far).
    pub engagement_count: u32,
    /// Hits / shots fired.
    pub hit_rate: f32,
    /// Seconds until first hit, or None if no hits.
    pub time_to_first_hit: Option<f32>,
    /// Mean speed when fighters are within 200m.
    pub avg_combat_speed: f32,
    /// 1.0 if match ended by elimination, 0.0 if timeout.
    pub elimination_rate: f32,
    /// Weighted composite score 0-100.
    pub dynamism_score: f32,
}

/// Analyze a replay and compute battle metrics.
pub fn analyze(replay: &Replay) -> BattleMetrics {
    let frames = &replay.frames;
    if frames.is_empty() {
        return BattleMetrics {
            circling_index: 0.0,
            speed_variance: 0.0,
            altitude_range: 0.0,
            engagement_count: 0,
            hit_rate: 0.0,
            time_to_first_hit: None,
            avg_combat_speed: 0.0,
            elimination_rate: 0.0,
            dynamism_score: 0.0,
        };
    }

    // --- Circling detection ---
    // Sliding window: check if both fighters are turning same direction over 30 frames.
    let window = 30usize;
    let mut circling_frames = 0u32;
    let mut total_windowed = 0u32;

    if frames.len() > window {
        for i in window..frames.len() {
            // Compute yaw delta for each fighter over the window
            let yaw_delta = |p: usize| -> f32 {
                let mut total = 0.0f32;
                for j in (i - window + 1)..=i {
                    let prev_yaw = frames[j - 1].fighters[p].yaw;
                    let curr_yaw = frames[j].fighters[p].yaw;
                    let mut d = curr_yaw - prev_yaw;
                    while d > std::f32::consts::PI { d -= 2.0 * std::f32::consts::PI; }
                    while d < -std::f32::consts::PI { d += 2.0 * std::f32::consts::PI; }
                    total += d;
                }
                total
            };

            let d0 = yaw_delta(0);
            let d1 = yaw_delta(1);

            total_windowed += 1;
            // Both turning same direction with significant magnitude
            if d0.signum() == d1.signum() && d0.abs() > 0.5 && d1.abs() > 0.5 {
                circling_frames += 1;
            }
        }
    }

    let circling_index = if total_windowed > 0 {
        circling_frames as f32 / total_windowed as f32
    } else {
        0.0
    };

    // --- Speed statistics ---
    let mut speed_sum = 0.0f32;
    let mut speed_sq_sum = 0.0f32;
    let mut speed_count = 0u32;
    let mut min_alt = f32::MAX;
    let mut max_alt = f32::MIN;

    for frame in frames {
        for f in &frame.fighters {
            if f.alive {
                speed_sum += f.speed;
                speed_sq_sum += f.speed * f.speed;
                speed_count += 1;

                if f.y < min_alt { min_alt = f.y; }
                if f.y > max_alt { max_alt = f.y; }
            }
        }
    }

    let speed_mean = if speed_count > 0 { speed_sum / speed_count as f32 } else { 0.0 };
    let speed_variance = if speed_count > 1 {
        (speed_sq_sum / speed_count as f32 - speed_mean * speed_mean).max(0.0).sqrt()
    } else {
        0.0
    };
    let altitude_range = if min_alt < f32::MAX { max_alt - min_alt } else { 0.0 };

    // --- Engagement counting ---
    let engage_threshold = 200.0f32;
    let mut engagement_count = 0u32;
    let mut was_close = false;

    for frame in frames {
        let dx = frame.fighters[0].x - frame.fighters[1].x;
        let dy = frame.fighters[0].y - frame.fighters[1].y;
        let dist = (dx * dx + dy * dy).sqrt();
        let is_close = dist < engage_threshold;

        if is_close != was_close {
            engagement_count += 1;
        }
        was_close = is_close;
    }

    // --- Hit rate ---
    let stats = &replay.result.stats;
    let total_shots = stats.p0_shots + stats.p1_shots;
    let total_hits = stats.p0_hits + stats.p1_hits;
    let hit_rate = if total_shots > 0 {
        total_hits as f32 / total_shots as f32
    } else {
        0.0
    };

    // --- Time to first hit ---
    let first_hit_tick = if total_hits > 0 {
        // Find first frame where either fighter has taken damage
        let mut first_tick = None;
        for frame in frames {
            if frame.fighters[0].hp < MAX_HP || frame.fighters[1].hp < MAX_HP {
                first_tick = Some(frame.tick);
                break;
            }
        }
        first_tick
    } else {
        None
    };
    let time_to_first_hit = first_hit_tick.map(|t| t as f32 / TICK_RATE as f32);

    // --- Average combat speed ---
    let mut combat_speed_sum = 0.0f32;
    let mut combat_speed_count = 0u32;

    for frame in frames {
        let dx = frame.fighters[0].x - frame.fighters[1].x;
        let dy = frame.fighters[0].y - frame.fighters[1].y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 200.0 {
            for f in &frame.fighters {
                if f.alive {
                    combat_speed_sum += f.speed;
                    combat_speed_count += 1;
                }
            }
        }
    }

    let avg_combat_speed = if combat_speed_count > 0 {
        combat_speed_sum / combat_speed_count as f32
    } else {
        0.0
    };

    // --- Elimination rate ---
    let elimination_rate = match replay.result.reason {
        MatchEndReason::Elimination => 1.0,
        MatchEndReason::Timeout => 0.0,
    };

    // --- Dynamism score (weighted composite 0-100) ---
    // Lower circling = better (inverted)
    let circ_score = (1.0 - circling_index) * 25.0;
    // More speed variance = better (capped at 50 m/s std dev)
    let speed_score = (speed_variance / 50.0).min(1.0) * 15.0;
    // More altitude range = better (capped at 400m)
    let alt_score = (altitude_range / 400.0).min(1.0) * 10.0;
    // More engagements = better (capped at 20)
    let engage_score = (engagement_count as f32 / 20.0).min(1.0) * 15.0;
    // Higher hit rate = better (capped at 0.3)
    let hit_score = (hit_rate / 0.3).min(1.0) * 10.0;
    // Elimination is better
    let elim_score = elimination_rate * 15.0;
    // Faster first hit = better (inverted, capped at 30s)
    let first_hit_score = time_to_first_hit
        .map(|t| (1.0 - (t / 30.0).min(1.0)) * 10.0)
        .unwrap_or(0.0);

    let dynamism_score = circ_score + speed_score + alt_score + engage_score
        + hit_score + elim_score + first_hit_score;

    BattleMetrics {
        circling_index,
        speed_variance,
        altitude_range,
        engagement_count,
        hit_rate,
        time_to_first_hit,
        avg_combat_speed,
        elimination_rate,
        dynamism_score,
    }
}
