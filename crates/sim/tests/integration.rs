use dogfight_shared::*;
use dogfight_sim::opponents::{ChaserPolicy, DogfighterPolicy};
use dogfight_sim::{run_match, DoNothingPolicy};

#[test]
fn test_chaser_beats_do_nothing() {
    let config = MatchConfig {
        seed: 42,
        p0_name: "chaser".into(),
        p1_name: "do_nothing".into(),
        ..Default::default()
    };
    let mut p0 = ChaserPolicy;
    let mut p1 = DoNothingPolicy;

    let replay = run_match(&config, &mut p0, &mut p1);

    assert_eq!(
        replay.result.outcome,
        MatchOutcome::Player0Win,
        "Chaser should beat DoNothing. Got {:?} at tick {} with p0_hp={} p1_hp={}",
        replay.result.outcome,
        replay.result.final_tick,
        replay.result.stats.p0_hp,
        replay.result.stats.p1_hp,
    );
}

#[test]
fn test_dogfighter_beats_do_nothing() {
    let config = MatchConfig {
        seed: 42,
        p0_name: "dogfighter".into(),
        p1_name: "do_nothing".into(),
        ..Default::default()
    };
    let mut p0 = DogfighterPolicy::new();
    let mut p1 = DoNothingPolicy;

    let replay = run_match(&config, &mut p0, &mut p1);

    assert_eq!(
        replay.result.outcome,
        MatchOutcome::Player0Win,
        "Dogfighter should beat DoNothing. Got {:?} at tick {} with p0_hp={} p1_hp={}",
        replay.result.outcome,
        replay.result.final_tick,
        replay.result.stats.p0_hp,
        replay.result.stats.p1_hp,
    );
}

#[test]
fn test_dogfighter_beats_chaser() {
    // Run multiple seeds - dogfighter should win majority
    let mut dogfighter_wins = 0;
    let mut chaser_wins = 0;
    let mut draws = 0;

    for seed in 0..10 {
        let config = MatchConfig {
            seed,
            p0_name: "dogfighter".into(),
            p1_name: "chaser".into(),
            ..Default::default()
        };
        let mut p0 = DogfighterPolicy::new();
        let mut p1 = ChaserPolicy;

        let replay = run_match(&config, &mut p0, &mut p1);

        match replay.result.outcome {
            MatchOutcome::Player0Win => dogfighter_wins += 1,
            MatchOutcome::Player1Win => chaser_wins += 1,
            MatchOutcome::Draw => draws += 1,
        }
    }

    println!(
        "Dogfighter vs Chaser (10 seeds): DF={}, C={}, D={}",
        dogfighter_wins, chaser_wins, draws
    );

    // Dogfighter should win at least 4 out of 10 (it's the smarter policy)
    assert!(
        dogfighter_wins >= 4,
        "Dogfighter should beat Chaser most of the time. Wins: {}/10",
        dogfighter_wins
    );
}

#[test]
fn test_deterministic_replays() {
    // Same seed should produce identical results
    let config = MatchConfig {
        seed: 123,
        p0_name: "chaser".into(),
        p1_name: "dogfighter".into(),
        ..Default::default()
    };

    let replay1 = {
        let mut p0 = ChaserPolicy;
        let mut p1 = DogfighterPolicy::new();
        run_match(&config, &mut p0, &mut p1)
    };

    let replay2 = {
        let mut p0 = ChaserPolicy;
        let mut p1 = DogfighterPolicy::new();
        run_match(&config, &mut p0, &mut p1)
    };

    assert_eq!(replay1.result.final_tick, replay2.result.final_tick);
    assert_eq!(replay1.result.outcome, replay2.result.outcome);
    assert_eq!(replay1.result.stats.p0_hp, replay2.result.stats.p0_hp);
    assert_eq!(replay1.result.stats.p1_hp, replay2.result.stats.p1_hp);
    assert_eq!(replay1.frames.len(), replay2.frames.len());
}

#[test]
fn test_match_completion_time() {
    // Matches with active opponents should end faster than 7200 ticks (usually via elimination)
    let config = MatchConfig {
        seed: 42,
        p0_name: "chaser".into(),
        p1_name: "chaser".into(),
        ..Default::default()
    };
    let mut p0 = ChaserPolicy;
    let mut p1 = ChaserPolicy;

    let replay = run_match(&config, &mut p0, &mut p1);

    // Two chasers should resolve before timeout
    assert!(
        replay.result.final_tick < MAX_TICKS,
        "Two chasers should not draw to timeout. Ended at tick {} with p0_hp={} p1_hp={}",
        replay.result.final_tick,
        replay.result.stats.p0_hp,
        replay.result.stats.p1_hp,
    );
}

#[test]
fn test_replay_serialization() {
    let config = MatchConfig {
        seed: 1,
        p0_name: "chaser".into(),
        p1_name: "do_nothing".into(),
        max_ticks: 240, // 2 seconds
        ..Default::default()
    };
    let mut p0 = ChaserPolicy;
    let mut p1 = DoNothingPolicy;

    let replay = run_match(&config, &mut p0, &mut p1);

    // Should serialize to JSON without error
    let json = serde_json::to_string(&replay).expect("replay should serialize");
    assert!(json.len() > 100);

    // Should deserialize back
    let replay2: Replay = serde_json::from_str(&json).expect("replay should deserialize");
    assert_eq!(replay.result.final_tick, replay2.result.final_tick);
}
