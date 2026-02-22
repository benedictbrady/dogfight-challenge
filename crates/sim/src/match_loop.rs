use dogfight_shared::*;

use crate::physics::SimState;
use crate::policy::Policy;

/// Run a deterministic match between two policies.
pub fn run_match(config: &MatchConfig, p0: &mut dyn Policy, p1: &mut dyn Policy) -> Replay {
    let mut state = SimState::new_with_seed_and_config(config.seed, config.randomize_spawns, config.sim_config);
    let mut frames = Vec::new();
    let mut action0 = Action::none();
    let mut action1 = Action::none();

    // Capture initial frame
    frames.push(state.snapshot());

    for tick in 0..config.max_ticks {
        // Update actions at fixed 12Hz rate (every CONTROL_PERIOD ticks)
        if tick % CONTROL_PERIOD == 0 {
            let obs0 = state.observe(0);
            action0 = p0.act(&obs0);
            let obs1 = state.observe(1);
            action1 = p1.act(&obs1);
        }

        state.step(&[action0, action1]);

        // Record frame every FRAME_INTERVAL ticks
        if state.tick % FRAME_INTERVAL == 0 {
            frames.push(state.snapshot());
        }

        if state.is_terminal() {
            // Capture final frame
            if state.tick % FRAME_INTERVAL != 0 {
                frames.push(state.snapshot());
            }
            break;
        }
    }

    let (outcome, reason) = state.outcome();

    Replay {
        config: config.clone(),
        frames,
        result: MatchResult {
            outcome,
            reason,
            final_tick: state.tick,
            stats: state.stats,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opponents::ChaserPolicy;

    #[test]
    fn test_match_completes() {
        let config = MatchConfig::default();
        let mut p0 = ChaserPolicy::new();
        let mut p1 = ChaserPolicy::new();

        let replay = run_match(&config, &mut p0, &mut p1);

        assert!(!replay.frames.is_empty());
        // Match should end (either elimination or timeout)
        assert!(replay.result.final_tick <= MAX_TICKS);
    }

    #[test]
    fn test_match_records_frames() {
        let config = MatchConfig {
            max_ticks: 120, // 1 second
            ..Default::default()
        };
        let mut p0 = ChaserPolicy::new();
        let mut p1 = ChaserPolicy::new();

        let replay = run_match(&config, &mut p0, &mut p1);

        // Should have ~30 frames per second + initial
        assert!(replay.frames.len() >= 30);
    }
}
