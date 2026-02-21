use std::io::Write;
use std::path::PathBuf;

use rayon::prelude::*;

use dogfight_shared::*;
use dogfight_sim::analyzer::{self, BattleMetrics};
use dogfight_sim::opponents::{AcePolicy, BrawlerPolicy, ChaserPolicy, DogfighterPolicy};
use dogfight_sim::{run_match, Policy};

/// A sweepable physics parameter with its name, range, and accessor.
struct SweepParam {
    name: &'static str,
    min: f64,
    default: f64,
    max: f64,
    /// Apply this parameter value to a SimConfig.
    apply: fn(&mut SimConfig, f64),
}

const SWEEP_PARAMS: &[SweepParam] = &[
    SweepParam {
        name: "gravity",
        min: 60.0,
        default: 130.0,
        max: 200.0,
        apply: |c, v| c.gravity = v as f32,
    },
    SweepParam {
        name: "drag_coeff",
        min: 0.5,
        default: 0.9,
        max: 1.3,
        apply: |c, v| c.drag_coeff = v as f32,
    },
    SweepParam {
        name: "turn_bleed_coeff",
        min: 0.05,
        default: 0.25,
        max: 0.45,
        apply: |c, v| c.turn_bleed_coeff = v as f32,
    },
    SweepParam {
        name: "max_speed",
        min: 140.0,
        default: 250.0,
        max: 320.0,
        apply: |c, v| c.max_speed = v as f32,
    },
    SweepParam {
        name: "min_speed",
        min: 10.0,
        default: 20.0,
        max: 40.0,
        apply: |c, v| c.min_speed = v as f32,
    },
    SweepParam {
        name: "max_thrust",
        min: 100.0,
        default: 180.0,
        max: 260.0,
        apply: |c, v| c.max_thrust = v as f32,
    },
    SweepParam {
        name: "bullet_speed",
        min: 300.0,
        default: 400.0,
        max: 500.0,
        apply: |c, v| c.bullet_speed = v as f32,
    },
    SweepParam {
        name: "gun_cooldown_ticks",
        min: 45.0,
        default: 90.0,
        max: 135.0,
        apply: |c, v| c.gun_cooldown_ticks = v as u32,
    },
    SweepParam {
        name: "bullet_lifetime_ticks",
        min: 30.0,
        default: 60.0,
        max: 90.0,
        apply: |c, v| c.bullet_lifetime_ticks = v as u32,
    },
    SweepParam {
        name: "max_hp",
        min: 3.0,
        default: 5.0,
        max: 10.0,
        apply: |c, v| c.max_hp = v as u8,
    },
    SweepParam {
        name: "max_turn_rate",
        min: 1.5,
        default: 4.0,
        max: 5.5,
        apply: |c, v| c.max_turn_rate = v as f32,
    },
    SweepParam {
        name: "min_turn_rate",
        min: 0.4,
        default: 0.8,
        max: 1.2,
        apply: |c, v| c.min_turn_rate = v as f32,
    },
    SweepParam {
        name: "rear_aspect_cone",
        min: 0.0,
        default: 0.785,
        max: 1.57,
        apply: |c, v| c.rear_aspect_cone = v as f32,
    },
];

fn resolve_policy(name: &str) -> Box<dyn Policy> {
    match name {
        "chaser" => Box::new(ChaserPolicy::new()),
        "dogfighter" => Box::new(DogfighterPolicy::new()),
        "ace" => Box::new(AcePolicy::new()),
        "brawler" => Box::new(BrawlerPolicy::new()),
        _ => {
            eprintln!("Unknown policy '{}' for sweep. Valid: chaser, dogfighter, ace, brawler", name);
            std::process::exit(1);
        }
    }
}

/// Aggregated metrics for one parameter value across all matchups and seeds.
struct AggResult {
    value: f64,
    mean_interestingness: f32,
    mean_dynamism: f32,
    mean_lead_changes: f32,
    mean_duration_quality: f32,
    mean_elim_rate: f32,
    match_count: u32,
}

/// A single match job to be run in parallel.
struct MatchJob {
    p0_name: String,
    p1_name: String,
    seed: u64,
    sim_config: SimConfig,
    randomize: bool,
}

fn run_job(job: &MatchJob) -> BattleMetrics {
    let mut p0 = resolve_policy(&job.p0_name);
    let mut p1 = resolve_policy(&job.p1_name);

    let config = MatchConfig {
        seed: job.seed,
        p0_name: job.p0_name.clone(),
        p1_name: job.p1_name.clone(),
        sim_config: job.sim_config,
        randomize_spawns: job.randomize,
        ..Default::default()
    };

    let replay = run_match(&config, p0.as_mut(), p1.as_mut());
    analyzer::analyze(&replay)
}

fn generate_matchup_pairs(policies: &[&str]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for i in 0..policies.len() {
        for j in (i + 1)..policies.len() {
            pairs.push((policies[i].to_string(), policies[j].to_string()));
        }
    }
    pairs
}

fn sweep_param(
    param: &SweepParam,
    steps: usize,
    seeds: u32,
    policies: &[&str],
    randomize: bool,
) -> Vec<AggResult> {
    let pairs = generate_matchup_pairs(policies);

    // Generate linearly-spaced values
    let values: Vec<f64> = if steps == 1 {
        vec![param.default]
    } else {
        (0..steps)
            .map(|i| param.min + (param.max - param.min) * i as f64 / (steps - 1) as f64)
            .collect()
    };

    values
        .iter()
        .map(|&val| {
            // Build all jobs for this value
            let jobs: Vec<MatchJob> = pairs
                .iter()
                .flat_map(|(p0, p1)| {
                    (0..seeds).map(move |s| {
                        let mut sim_config = SimConfig::default();
                        (param.apply)(&mut sim_config, val);
                        MatchJob {
                            p0_name: p0.clone(),
                            p1_name: p1.clone(),
                            seed: s as u64,
                            sim_config,
                            randomize,
                        }
                    })
                })
                .collect();

            let metrics: Vec<BattleMetrics> =
                jobs.par_iter().map(|job| run_job(job)).collect();

            let n = metrics.len() as f32;
            let mean_interestingness =
                metrics.iter().map(|m| m.interestingness_score).sum::<f32>() / n;
            let mean_dynamism = metrics.iter().map(|m| m.dynamism_score).sum::<f32>() / n;
            let mean_lead_changes =
                metrics.iter().map(|m| m.lead_changes as f32).sum::<f32>() / n;
            let mean_duration_quality =
                metrics.iter().map(|m| m.duration_quality).sum::<f32>() / n;
            let mean_elim_rate =
                metrics.iter().map(|m| m.elimination_rate).sum::<f32>() / n;

            AggResult {
                value: val,
                mean_interestingness,
                mean_dynamism,
                mean_lead_changes,
                mean_duration_quality,
                mean_elim_rate,
                match_count: metrics.len() as u32,
            }
        })
        .collect()
}

fn print_param_table(param_name: &str, results: &[AggResult]) {
    println!("\n--- {} ---", param_name);
    println!(
        "{:>12} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "value", "intrstng", "dynamism", "lead_ch", "dur_q", "elim%"
    );
    println!("{:-<60}", "");

    let best_idx = results
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.mean_interestingness
                .partial_cmp(&b.mean_interestingness)
                .unwrap()
        })
        .map(|(i, _)| i);

    for (i, r) in results.iter().enumerate() {
        let marker = if Some(i) == best_idx { " *" } else { "" };
        println!(
            "{:>12.3} {:>8.1} {:>8.1} {:>8.2} {:>8.3} {:>6.2}{}",
            r.value,
            r.mean_interestingness,
            r.mean_dynamism,
            r.mean_lead_changes,
            r.mean_duration_quality,
            r.mean_elim_rate,
            marker,
        );
    }
}

/// Evaluate a SimConfig across all matchup pairs and seeds, returning mean interestingness.
fn eval_config(
    config: SimConfig,
    policies: &[&str],
    seeds: u32,
    randomize: bool,
) -> f32 {
    let pairs = generate_matchup_pairs(policies);
    let jobs: Vec<MatchJob> = pairs
        .iter()
        .flat_map(|(p0, p1)| {
            (0..seeds).map(move |s| MatchJob {
                p0_name: p0.clone(),
                p1_name: p1.clone(),
                seed: s as u64,
                sim_config: config,
                randomize,
            })
        })
        .collect();

    let metrics: Vec<BattleMetrics> = jobs.par_iter().map(|j| run_job(j)).collect();
    metrics.iter().map(|m| m.interestingness_score).sum::<f32>() / metrics.len() as f32
}

fn write_csv(path: &std::path::Path, all_results: &[(&str, Vec<AggResult>)]) {
    let mut file = std::fs::File::create(path).expect("Failed to create CSV file");
    writeln!(
        file,
        "parameter,value,interestingness,dynamism,lead_changes,duration_quality,elimination_rate,match_count"
    )
    .unwrap();

    for (param_name, results) in all_results {
        for r in results {
            writeln!(
                file,
                "{},{:.4},{:.2},{:.2},{:.3},{:.4},{:.3},{}",
                param_name,
                r.value,
                r.mean_interestingness,
                r.mean_dynamism,
                r.mean_lead_changes,
                r.mean_duration_quality,
                r.mean_elim_rate,
                r.match_count,
            )
            .unwrap();
        }
    }
    println!("\nCSV written to {}", path.display());
}

pub fn cmd_sweep(
    param_filter: Option<&str>,
    steps: usize,
    seeds: u32,
    policies_str: &str,
    output: Option<PathBuf>,
    validate: bool,
    randomize: bool,
    optimize: bool,
) {
    let policies: Vec<&str> = policies_str.split(',').map(|s| s.trim()).collect();

    if policies.len() < 2 {
        eprintln!("Sweep requires at least 2 policies.");
        std::process::exit(1);
    }

    let pairs = generate_matchup_pairs(&policies);
    let total_per_param = pairs.len() * seeds as usize * steps;

    // Filter to requested parameter(s)
    let params_to_sweep: Vec<&SweepParam> = if let Some(name) = param_filter {
        let found = SWEEP_PARAMS.iter().find(|p| p.name == name);
        match found {
            Some(p) => vec![p],
            None => {
                eprintln!(
                    "Unknown parameter '{}'. Available: {}",
                    name,
                    SWEEP_PARAMS
                        .iter()
                        .map(|p| p.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                std::process::exit(1);
            }
        }
    } else {
        SWEEP_PARAMS.iter().collect()
    };

    let total_matches = params_to_sweep.len() * total_per_param;
    println!(
        "=== Physics Sweep ===\nPolicies: {} | Steps: {} | Seeds: {} | Randomize: {}\nParams: {} | Total matches: {}",
        policies.join(", "),
        steps,
        seeds,
        randomize,
        params_to_sweep.len(),
        total_matches,
    );

    let start = std::time::Instant::now();

    let mut all_results: Vec<(&str, Vec<AggResult>)> = Vec::new();
    let mut best_per_param: Vec<(&str, f64, f32)> = Vec::new();

    for param in &params_to_sweep {
        let results = sweep_param(param, steps, seeds, &policies, randomize);

        // Find best value
        let best = results
            .iter()
            .max_by(|a, b| {
                a.mean_interestingness
                    .partial_cmp(&b.mean_interestingness)
                    .unwrap()
            })
            .unwrap();

        best_per_param.push((param.name, best.value, best.mean_interestingness));
        print_param_table(param.name, &results);
        all_results.push((param.name, results));
    }

    let elapsed = start.elapsed();
    println!("\n=== Summary ({:.1}s) ===", elapsed.as_secs_f32());
    println!("{:<25} {:>12} {:>10}", "Parameter", "Best Value", "Score");
    println!("{:-<50}", "");
    for (name, value, score) in &best_per_param {
        println!("{:<25} {:>12.3} {:>10.1}", name, value, score);
    }

    // Write CSV if requested
    if let Some(path) = &output {
        write_csv(path, &all_results);
    }

    // Validate mode: assemble best-per-param config and compare to default
    if validate {
        println!("\n=== Validation: Best-per-param config vs Default ===");
        let mut best_config = SimConfig::default();
        for (name, value, _) in &best_per_param {
            let param = SWEEP_PARAMS.iter().find(|p| p.name == *name).unwrap();
            (param.apply)(&mut best_config, *value);
        }

        let validation_seeds = 20u32;
        let pairs = generate_matchup_pairs(&policies);

        // Run with default config
        let default_jobs: Vec<MatchJob> = pairs
            .iter()
            .flat_map(|(p0, p1)| {
                (0..validation_seeds).map(move |s| MatchJob {
                    p0_name: p0.clone(),
                    p1_name: p1.clone(),
                    seed: s as u64,
                    sim_config: SimConfig::default(),
                    randomize,
                })
            })
            .collect();

        let default_metrics: Vec<BattleMetrics> =
            default_jobs.par_iter().map(|j| run_job(j)).collect();
        let default_interest = default_metrics
            .iter()
            .map(|m| m.interestingness_score)
            .sum::<f32>()
            / default_metrics.len() as f32;

        // Run with best config
        let best_jobs: Vec<MatchJob> = pairs
            .iter()
            .flat_map(|(p0, p1)| {
                (0..validation_seeds).map(move |s| MatchJob {
                    p0_name: p0.clone(),
                    p1_name: p1.clone(),
                    seed: s as u64,
                    sim_config: best_config,
                    randomize,
                })
            })
            .collect();

        let best_metrics: Vec<BattleMetrics> =
            best_jobs.par_iter().map(|j| run_job(j)).collect();
        let best_interest = best_metrics
            .iter()
            .map(|m| m.interestingness_score)
            .sum::<f32>()
            / best_metrics.len() as f32;

        println!(
            "Default physics:  interestingness = {:.1}",
            default_interest
        );
        println!(
            "Best-per-param:   interestingness = {:.1}",
            best_interest
        );
        let delta = best_interest - default_interest;
        println!(
            "Delta:            {:+.1} ({:+.1}%)",
            delta,
            delta / default_interest * 100.0
        );
    }

    // Greedy forward-selection: add one param at a time, keep only if it improves score
    if optimize {
        println!("\n=== Greedy Forward-Selection Optimization ===");
        let opt_seeds = 20u32;

        let baseline = eval_config(SimConfig::default(), &policies, opt_seeds, randomize);
        println!("Baseline (default physics): {:.1}", baseline);

        let mut current_config = SimConfig::default();
        let mut current_score = baseline;
        let mut applied: Vec<(&str, f64)> = Vec::new();

        // Sort params by individual improvement (descending)
        let mut candidates: Vec<(&str, f64)> = best_per_param
            .iter()
            .filter(|(name, value, _)| {
                // Skip params where best == default
                let param = SWEEP_PARAMS.iter().find(|p| p.name == *name).unwrap();
                (*value - param.default).abs() > 1e-6
            })
            .map(|(name, value, _)| (*name, *value))
            .collect();

        // Sort by individual score (highest first)
        candidates.sort_by(|a, b| {
            let a_score = best_per_param.iter().find(|(n, _, _)| *n == a.0).unwrap().2;
            let b_score = best_per_param.iter().find(|(n, _, _)| *n == b.0).unwrap().2;
            b_score.partial_cmp(&a_score).unwrap()
        });

        for (name, value) in &candidates {
            let param = SWEEP_PARAMS.iter().find(|p| p.name == *name).unwrap();
            let mut test_config = current_config;
            (param.apply)(&mut test_config, *value);

            let score = eval_config(test_config, &policies, opt_seeds, randomize);
            let delta = score - current_score;

            if delta > 0.5 {
                println!(
                    "  + {:<25} = {:>8.3}  score: {:.1} ({:+.1}) KEPT",
                    name, value, score, delta
                );
                current_config = test_config;
                current_score = score;
                applied.push((name, *value));
            } else {
                println!(
                    "  - {:<25} = {:>8.3}  score: {:.1} ({:+.1}) skipped",
                    name, value, score, delta
                );
            }
        }

        println!("\n=== Optimized Config ===");
        println!(
            "Score: {:.1} (baseline: {:.1}, {:+.1}%)",
            current_score,
            baseline,
            (current_score - baseline) / baseline * 100.0,
        );
        if applied.is_empty() {
            println!("No improvements found — default physics is optimal.");
        } else {
            println!("Applied changes:");
            for (name, value) in &applied {
                let param = SWEEP_PARAMS.iter().find(|p| p.name == *name).unwrap();
                println!(
                    "  {:<25} {:>8.3} → {:>8.3}",
                    name, param.default, value
                );
            }
        }
    }
}
