use dogfight_shared::*;
use dogfight_sim::opponents::{AcePolicy, BrawlerPolicy, ChaserPolicy, DogfighterPolicy};
use dogfight_sim::{run_match, DoNothingPolicy};

#[test]
fn test_chaser_beats_do_nothing() {
    let config = MatchConfig {
        seed: 42,
        p0_name: "chaser".into(),
        p1_name: "do_nothing".into(),
        ..Default::default()
    };
    let mut p0 = ChaserPolicy::new();
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
fn test_ace_beats_do_nothing() {
    let config = MatchConfig {
        seed: 42,
        p0_name: "ace".into(),
        p1_name: "do_nothing".into(),
        ..Default::default()
    };
    let mut p0 = AcePolicy::new();
    let mut p1 = DoNothingPolicy;

    let replay = run_match(&config, &mut p0, &mut p1);

    assert_eq!(
        replay.result.outcome,
        MatchOutcome::Player0Win,
        "Ace should beat DoNothing. Got {:?} at tick {} with p0_hp={} p1_hp={}",
        replay.result.outcome,
        replay.result.final_tick,
        replay.result.stats.p0_hp,
        replay.result.stats.p1_hp,
    );
}

#[test]
fn test_brawler_beats_do_nothing() {
    let config = MatchConfig {
        seed: 42,
        p0_name: "brawler".into(),
        p1_name: "do_nothing".into(),
        ..Default::default()
    };
    let mut p0 = BrawlerPolicy::new();
    let mut p1 = DoNothingPolicy;

    let replay = run_match(&config, &mut p0, &mut p1);

    assert_eq!(
        replay.result.outcome,
        MatchOutcome::Player0Win,
        "Brawler should beat DoNothing. Got {:?} at tick {} with p0_hp={} p1_hp={}",
        replay.result.outcome,
        replay.result.final_tick,
        replay.result.stats.p0_hp,
        replay.result.stats.p1_hp,
    );
}

#[test]
fn test_chaser_vs_dogfighter() {
    // With rear-aspect armor, chaser's pressure style dominates dogfighter
    let mut ch_wins = 0;

    for seed in 0..10 {
        let config = MatchConfig {
            seed,
            p0_name: "chaser".into(),
            p1_name: "dogfighter".into(),
            ..Default::default()
        };
        let mut p0 = ChaserPolicy::new();
        let mut p1 = DogfighterPolicy::new();

        let replay = run_match(&config, &mut p0, &mut p1);
        if replay.result.outcome == MatchOutcome::Player0Win {
            ch_wins += 1;
        }
    }

    println!("Chaser vs Dogfighter (10 seeds): CH wins={}", ch_wins);
    assert!(
        ch_wins >= 5,
        "Chaser should beat Dogfighter majority. CH wins: {}/10",
        ch_wins
    );
}

#[test]
fn test_deterministic_replays() {
    let config = MatchConfig {
        seed: 123,
        p0_name: "chaser".into(),
        p1_name: "dogfighter".into(),
        ..Default::default()
    };

    let replay1 = {
        let mut p0 = ChaserPolicy::new();
        let mut p1 = DogfighterPolicy::new();
        run_match(&config, &mut p0, &mut p1)
    };

    let replay2 = {
        let mut p0 = ChaserPolicy::new();
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
fn test_ace_deterministic() {
    let config = MatchConfig {
        seed: 77,
        p0_name: "ace".into(),
        p1_name: "brawler".into(),
        ..Default::default()
    };

    let replay1 = {
        let mut p0 = AcePolicy::new();
        let mut p1 = BrawlerPolicy::new();
        run_match(&config, &mut p0, &mut p1)
    };

    let replay2 = {
        let mut p0 = AcePolicy::new();
        let mut p1 = BrawlerPolicy::new();
        run_match(&config, &mut p0, &mut p1)
    };

    assert_eq!(replay1.result.final_tick, replay2.result.final_tick);
    assert_eq!(replay1.result.outcome, replay2.result.outcome);
}

#[test]
fn test_match_completion_time() {
    // With 5 HP + rear-aspect armor, matches take longer.
    // Verify brawler vs ace finishes by elimination (brawler's perpendicular
    // style is most effective with rear-aspect armor).
    let config = MatchConfig {
        seed: 42,
        p0_name: "brawler".into(),
        p1_name: "ace".into(),
        ..Default::default()
    };
    let mut p0 = BrawlerPolicy::new();
    let mut p1 = AcePolicy::new();

    let replay = run_match(&config, &mut p0, &mut p1);

    assert!(
        replay.result.final_tick < MAX_TICKS,
        "Brawler vs Ace should not draw to timeout. Ended at tick {} with p0_hp={} p1_hp={}",
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
        max_ticks: 240,
        ..Default::default()
    };
    let mut p0 = ChaserPolicy::new();
    let mut p1 = DoNothingPolicy;

    let replay = run_match(&config, &mut p0, &mut p1);

    let json = serde_json::to_string(&replay).expect("replay should serialize");
    assert!(json.len() > 100);

    let replay2: Replay = serde_json::from_str(&json).expect("replay should deserialize");
    assert_eq!(replay.result.final_tick, replay2.result.final_tick);
}

/// Run a multi-seed matchup and return (p0_wins, p1_wins, draws)
fn run_matchup(p0_name: &str, p1_name: &str, seeds: u64) -> (u32, u32, u32) {
    let mut p0_wins = 0u32;
    let mut p1_wins = 0u32;
    let mut draws = 0u32;

    for seed in 0..seeds {
        let config = MatchConfig {
            seed,
            p0_name: p0_name.into(),
            p1_name: p1_name.into(),
            ..Default::default()
        };

        let mut p0: Box<dyn dogfight_sim::Policy> = match p0_name {
            "chaser" => Box::new(ChaserPolicy::new()),
            "dogfighter" => Box::new(DogfighterPolicy::new()),
            "ace" => Box::new(AcePolicy::new()),
            "brawler" => Box::new(BrawlerPolicy::new()),
            _ => Box::new(DoNothingPolicy),
        };
        let mut p1: Box<dyn dogfight_sim::Policy> = match p1_name {
            "chaser" => Box::new(ChaserPolicy::new()),
            "dogfighter" => Box::new(DogfighterPolicy::new()),
            "ace" => Box::new(AcePolicy::new()),
            "brawler" => Box::new(BrawlerPolicy::new()),
            _ => Box::new(DoNothingPolicy),
        };

        let replay = run_match(&config, p0.as_mut(), p1.as_mut());
        match replay.result.outcome {
            MatchOutcome::Player0Win => p0_wins += 1,
            MatchOutcome::Player1Win => p1_wins += 1,
            MatchOutcome::Draw => draws += 1,
        }
    }

    (p0_wins, p1_wins, draws)
}

#[test]
fn test_all_policies_beat_do_nothing_multi_seed() {
    for policy in &["chaser", "dogfighter", "ace", "brawler"] {
        let (wins, _, _) = run_matchup(policy, "do_nothing", 5);
        assert!(
            wins >= 4,
            "{} should beat do_nothing consistently. Wins: {}/5",
            policy,
            wins
        );
    }
}

#[test]
fn test_matchup_balance_overview() {
    // Verify expected matchup directions with rear-aspect armor.
    // The sim is fully deterministic (no RNG), so all seeds produce the same winner.
    //
    // New balance hierarchy (post rear-aspect armor + 5HP):
    //   brawler > ace > chaser > dogfighter
    //   brawler > chaser
    //   ace > dogfighter
    //   chaser > dogfighter
    //
    // Brawler's perpendicular fighting style benefits most from rear-aspect armor.
    let expected_winners: Vec<(&str, &str, &str)> = vec![
        ("ace", "chaser", "ace"),               // ace beats chaser
        ("brawler", "ace", "brawler"),          // brawler beats ace
        ("brawler", "chaser", "brawler"),       // brawler beats chaser
        ("chaser", "dogfighter", "chaser"),     // chaser beats dogfighter
        ("ace", "dogfighter", "ace"),           // ace beats dogfighter
    ];

    for (p0, p1, expected) in &expected_winners {
        let (w0, _w1, _d) = run_matchup(p0, p1, 1);
        let winner = if w0 == 1 { p0 } else { p1 };
        println!("{} vs {}: winner = {}", p0, p1, winner);
        assert_eq!(
            winner, expected,
            "{} vs {}: expected {} to win, but {} won",
            p0, p1, expected, winner
        );
    }
}
