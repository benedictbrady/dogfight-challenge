use std::collections::HashMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use dogfight_shared::*;
use dogfight_sim::opponents::{AcePolicy, BrawlerPolicy, ChaserPolicy, DogfighterPolicy};
use dogfight_sim::{run_match, DoNothingPolicy, Policy};

#[derive(Parser)]
#[command(name = "dogfight", about = "Dogfight challenge CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a match between two policies
    Run {
        /// Policy for player 0 (chaser, dogfighter, do_nothing, or .onnx path)
        #[arg(long)]
        p0: String,

        /// Policy for player 1 (chaser, dogfighter, do_nothing, or .onnx path)
        #[arg(long)]
        p1: String,

        /// Random seed for the match
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Output path for replay JSON
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Validate an ONNX model
    Validate {
        /// Path to the .onnx model file
        model_path: PathBuf,
    },

    /// Start the match server
    Serve {
        /// Port to listen on
        #[arg(long, default_value_t = 3001)]
        port: u16,
    },

    /// Run a round-robin tournament between policies
    Tournament {
        /// Comma-separated list of policy names
        #[arg(long)]
        policies: String,

        /// Number of rounds per matchup
        #[arg(long)]
        rounds: u32,
    },
}

/// Resolve a policy name to a boxed Policy trait object.
///
/// Supported names:
/// - "chaser" -> ChaserPolicy
/// - "dogfighter" -> DogfighterPolicy
/// - "do_nothing" -> DoNothingPolicy
/// - A path ending in ".onnx" -> falls back to DoNothingPolicy with a warning
fn resolve_policy(name: &str) -> Box<dyn Policy> {
    match name {
        "chaser" => Box::new(ChaserPolicy::new()),
        "dogfighter" => Box::new(DogfighterPolicy::new()),
        "ace" => Box::new(AcePolicy::new()),
        "brawler" => Box::new(BrawlerPolicy::new()),
        "do_nothing" => Box::new(DoNothingPolicy),
        path if path.ends_with(".onnx") => {
            eprintln!(
                "Warning: ONNX loading requires the onnxruntime library. \
                 Falling back to DoNothingPolicy for '{}'.",
                path
            );
            Box::new(DoNothingPolicy)
        }
        other => {
            eprintln!(
                "Unknown policy '{}'. Valid options: chaser, dogfighter, ace, brawler, do_nothing, or a .onnx file path.",
                other
            );
            std::process::exit(1);
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            p0,
            p1,
            seed,
            output,
        } => cmd_run(&p0, &p1, seed, output),

        Commands::Validate { model_path } => cmd_validate(&model_path),

        Commands::Serve { port } => cmd_serve(port),

        Commands::Tournament { policies, rounds } => cmd_tournament(&policies, rounds),
    }
}

fn cmd_run(p0_name: &str, p1_name: &str, seed: u64, output: Option<PathBuf>) {
    let mut p0 = resolve_policy(p0_name);
    let mut p1 = resolve_policy(p1_name);

    let config = MatchConfig {
        seed,
        p0_name: p0.name().to_string(),
        p1_name: p1.name().to_string(),
        p0_control_period: 1,
        p1_control_period: 1,
        max_ticks: MAX_TICKS,
    };

    println!(
        "Running match: {} vs {} (seed={})",
        p0.name(),
        p1.name(),
        seed
    );

    let replay = run_match(&config, p0.as_mut(), p1.as_mut());
    let result = &replay.result;

    println!();
    println!("=== Match Result ===");
    println!("Outcome:    {:?}", result.outcome);
    println!("Reason:     {:?}", result.reason);
    println!("Final tick: {} ({:.1}s)", result.final_tick, result.final_tick as f32 / TICK_RATE as f32);
    println!();
    println!("--- Stats ---");
    println!(
        "  {} (P0): HP={}, Hits={}, Shots={}",
        config.p0_name, result.stats.p0_hp, result.stats.p0_hits, result.stats.p0_shots
    );
    println!(
        "  {} (P1): HP={}, Hits={}, Shots={}",
        config.p1_name, result.stats.p1_hp, result.stats.p1_hits, result.stats.p1_shots
    );

    if let Some(path) = output {
        match serde_json::to_string_pretty(&replay) {
            Ok(json) => match std::fs::write(&path, json) {
                Ok(()) => println!("\nReplay written to {}", path.display()),
                Err(e) => eprintln!("\nFailed to write replay: {}", e),
            },
            Err(e) => eprintln!("\nFailed to serialize replay: {}", e),
        }
    }
}

fn cmd_validate(model_path: &PathBuf) {
    // TODO: Wire up dogfight-validator once onnxruntime is available
    println!(
        "ONNX validation requires onnxruntime library (model: {})",
        model_path.display()
    );
}

fn cmd_serve(port: u16) {
    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        if let Err(e) = dogfight_server::run_server(port).await {
            eprintln!("Server error: {}", e);
            std::process::exit(1);
        }
    });
}

fn cmd_tournament(policies_str: &str, rounds: u32) {
    let policy_names: Vec<&str> = policies_str.split(',').map(|s| s.trim()).collect();

    if policy_names.len() < 2 {
        eprintln!("Tournament requires at least 2 policies.");
        std::process::exit(1);
    }

    println!(
        "Tournament: {} policies, {} rounds per matchup",
        policy_names.len(),
        rounds
    );
    println!("Policies: {}", policy_names.join(", "));
    println!();

    let mut scores: HashMap<String, u32> = HashMap::new();
    for name in &policy_names {
        scores.insert(name.to_string(), 0);
    }

    for i in 0..policy_names.len() {
        for j in (i + 1)..policy_names.len() {
            let name_a = policy_names[i];
            let name_b = policy_names[j];

            println!("--- {} vs {} ---", name_a, name_b);

            let mut a_wins = 0u32;
            let mut b_wins = 0u32;
            let mut draws = 0u32;

            for round in 0..rounds {
                let mut p0 = resolve_policy(name_a);
                let mut p1 = resolve_policy(name_b);

                let config = MatchConfig {
                    seed: round as u64,
                    p0_name: p0.name().to_string(),
                    p1_name: p1.name().to_string(),
                    p0_control_period: 1,
                    p1_control_period: 1,
                    max_ticks: MAX_TICKS,
                };

                let replay = run_match(&config, p0.as_mut(), p1.as_mut());
                let result = &replay.result;

                let a_pts = result.outcome.points_with_reason(result.reason, 0);
                let b_pts = result.outcome.points_with_reason(result.reason, 1);

                *scores.get_mut(name_a).unwrap() += a_pts;
                *scores.get_mut(name_b).unwrap() += b_pts;

                match result.outcome {
                    MatchOutcome::Player0Win => a_wins += 1,
                    MatchOutcome::Player1Win => b_wins += 1,
                    MatchOutcome::Draw => draws += 1,
                }
            }

            println!(
                "  Results: {} wins={}, {} wins={}, draws={}",
                name_a, a_wins, name_b, b_wins, draws
            );
        }
    }

    // Print scoreboard
    println!();
    println!("=== Tournament Scoreboard ===");
    println!("{:<20} {:>8}", "Policy", "Points");
    println!("{:-<20} {:-<8}", "", "");

    let mut sorted_scores: Vec<(&String, &u32)> = scores.iter().collect();
    sorted_scores.sort_by(|a, b| b.1.cmp(a.1));

    for (name, pts) in sorted_scores {
        println!("{:<20} {:>8}", name, pts);
    }
}
