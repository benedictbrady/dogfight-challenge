use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use dogfight_shared::*;
use dogfight_sim::analyzer;
use dogfight_sim::opponents::{AcePolicy, BrawlerPolicy, ChaserPolicy, DogfighterPolicy};
use dogfight_sim::{run_match, Policy};
use dogfight_validator::OnnxPolicy;

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
        /// Policy for player 0 (chaser, dogfighter, ace, brawler, or .onnx path)
        #[arg(long)]
        p0: String,

        /// Policy for player 1 (chaser, dogfighter, ace, brawler, or .onnx path)
        #[arg(long)]
        p1: String,

        /// Random seed for the match
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Output path for replay JSON
        #[arg(long)]
        output: Option<PathBuf>,

        /// Use randomized spawn positions
        #[arg(long)]
        randomize: bool,
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

    /// Analyze battle dynamics across matchups
    Analyze {
        /// Comma-separated list of policy names
        #[arg(long, default_value = "chaser,dogfighter,ace,brawler")]
        policies: String,

        /// Number of seeds to run per matchup
        #[arg(long, default_value_t = 5)]
        seeds: u32,

        /// Optional label for this analysis run
        #[arg(long)]
        label: Option<String>,

        /// Use randomized spawns
        #[arg(long)]
        randomize: bool,
    },
}

/// Resolve a policy name to a boxed Policy trait object.
///
/// Supported names: chaser, dogfighter, ace, brawler, neural,
/// or a path ending in ".onnx". Exits the process on unknown policy or load failure.
fn resolve_policy(name: &str) -> Box<dyn Policy> {
    match name {
        "chaser" => Box::new(ChaserPolicy::new()),
        "dogfighter" => Box::new(DogfighterPolicy::new()),
        "ace" => Box::new(AcePolicy::new()),
        "brawler" => Box::new(BrawlerPolicy::new()),
        path if path == "neural" || path.ends_with(".onnx") => {
            let onnx_path = if path == "neural" { "policy.onnx" } else { path };
            let p = std::path::Path::new(onnx_path);
            match OnnxPolicy::load(p) {
                Ok(policy) => {
                    println!("Loaded ONNX policy from {}", p.display());
                    Box::new(policy)
                }
                Err(e) => {
                    eprintln!("Failed to load ONNX policy '{}': {e}", p.display());
                    std::process::exit(1);
                }
            }
        }
        other => {
            // Try models/ directory
            let models_path = std::path::Path::new("models").join(format!("{}.onnx", other));
            if models_path.exists() {
                match OnnxPolicy::load(&models_path) {
                    Ok(policy) => {
                        println!("Loaded ONNX policy from {}", models_path.display());
                        return Box::new(policy);
                    }
                    Err(e) => {
                        eprintln!("Failed to load ONNX policy '{}': {e}", models_path.display());
                        std::process::exit(1);
                    }
                }
            }
            eprintln!(
                "Unknown policy '{}'. Valid options: chaser, dogfighter, ace, brawler, neural, or a .onnx file path.",
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
            randomize,
        } => cmd_run(&p0, &p1, seed, output, randomize),

        Commands::Validate { model_path } => cmd_validate(&model_path),

        Commands::Serve { port } => cmd_serve(port),

        Commands::Tournament { policies, rounds } => cmd_tournament(&policies, rounds),

        Commands::Analyze {
            policies,
            seeds,
            label,
            randomize,
        } => cmd_analyze(&policies, seeds, label.as_deref(), randomize),
    }
}

fn cmd_run(p0_name: &str, p1_name: &str, seed: u64, output: Option<PathBuf>, randomize: bool) {
    let mut p0 = resolve_policy(p0_name);
    let mut p1 = resolve_policy(p1_name);

    let config = MatchConfig {
        seed,
        p0_name: p0.name().to_string(),
        p1_name: p1.name().to_string(),
        randomize_spawns: randomize,
        ..Default::default()
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

fn cmd_validate(model_path: &Path) {
    match dogfight_validator::validate_model_file(model_path) {
        Ok(report) => {
            println!("Model: {}", model_path.display());
            println!("  File size:  {} bytes", report.file_size_bytes);
            println!("  Parameters: {} / {} (max)", report.parameter_count, MAX_PARAMETERS);
            println!("  Input:      {:?}", report.input_shape);
            println!("  Output:     {:?}", report.output_shape);
            println!("  Ops:        {}", report.ops_used.join(", "));
            println!("  Status:     PASS");
        }
        Err(e) => {
            eprintln!("Validation FAILED: {e}");
            std::process::exit(1);
        }
    }
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
                    ..Default::default()
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

fn cmd_analyze(policies_str: &str, seeds: u32, label: Option<&str>, randomize: bool) {
    let policy_names: Vec<&str> = policies_str.split(',').map(|s| s.trim()).collect();

    if policy_names.len() < 2 {
        eprintln!("Analyze requires at least 2 policies.");
        std::process::exit(1);
    }

    if let Some(l) = label {
        println!("=== Battle Dynamics Analysis: {} ===", l);
    } else {
        println!("=== Battle Dynamics Analysis ===");
    }
    println!(
        "Policies: {} | Seeds: {} | Randomize: {}",
        policy_names.join(", "),
        seeds,
        randomize
    );
    println!();

    // Header
    println!(
        "{:<25} {:>5} {:>8} {:>8} {:>7} {:>5} {:>5} {:>8} {:>8}",
        "Matchup", "circ", "spd_var", "alt_rng", "engage", "hit%", "elim%", "combat_v", "dynam"
    );
    println!("{:-<85}", "");

    let mut total_dynamism = 0.0f32;
    let mut matchup_count = 0u32;

    for i in 0..policy_names.len() {
        for j in (i + 1)..policy_names.len() {
            let name_a = policy_names[i];
            let name_b = policy_names[j];

            let mut agg_circ = 0.0f32;
            let mut agg_spd = 0.0f32;
            let mut agg_alt = 0.0f32;
            let mut agg_engage = 0u32;
            let mut agg_hit = 0.0f32;
            let mut agg_elim = 0.0f32;
            let mut agg_combat = 0.0f32;
            let mut agg_dyn = 0.0f32;

            for seed in 0..seeds {
                let mut p0 = resolve_policy(name_a);
                let mut p1 = resolve_policy(name_b);

                let config = MatchConfig {
                    seed: seed as u64,
                    p0_name: p0.name().to_string(),
                    p1_name: p1.name().to_string(),
                    randomize_spawns: randomize,
                    ..Default::default()
                };

                let replay = run_match(&config, p0.as_mut(), p1.as_mut());
                let m = analyzer::analyze(&replay);

                agg_circ += m.circling_index;
                agg_spd += m.speed_variance;
                agg_alt += m.altitude_range;
                agg_engage += m.engagement_count;
                agg_hit += m.hit_rate;
                agg_elim += m.elimination_rate;
                agg_combat += m.avg_combat_speed;
                agg_dyn += m.dynamism_score;
            }

            let n = seeds as f32;
            let matchup_label = format!("{} v {}", name_a, name_b);

            println!(
                "{:<25} {:>5.2} {:>8.1} {:>8.0} {:>7.1} {:>5.2} {:>5.2} {:>8.1} {:>8.1}",
                matchup_label,
                agg_circ / n,
                agg_spd / n,
                agg_alt / n,
                agg_engage as f32 / n,
                agg_hit / n,
                agg_elim / n,
                agg_combat / n,
                agg_dyn / n,
            );

            total_dynamism += agg_dyn / n;
            matchup_count += 1;
        }
    }

    println!("{:-<85}", "");
    println!(
        "Average dynamism score: {:.1}/100",
        total_dynamism / matchup_count as f32
    );
}
