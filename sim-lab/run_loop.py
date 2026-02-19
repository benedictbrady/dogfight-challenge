"""Full pipeline orchestrator for sim-vs-real evaluation iterations."""

import argparse
import json
import os
import shutil
import statistics
from pathlib import Path

import numpy as np

from sim_interface import (
    EvalResult,
    IterationSnapshot,
    PendulumParams,
    Trajectory,
)
import candidate_sim
import diagnostics
import metrics
import scenarios
import visualize


SIM_LAB_DIR = Path(__file__).parent
ITERATIONS_DIR = SIM_LAB_DIR / "iterations"


def _next_iteration_number() -> int:
    """Auto-detect the next iteration number from existing directories."""
    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    for d in ITERATIONS_DIR.iterdir():
        if d.is_dir() and d.name.startswith("iter_"):
            try:
                existing.append(int(d.name.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return max(existing, default=-1) + 1


def _evaluate_run(run_data, params: PendulumParams) -> tuple[EvalResult, Trajectory]:
    """Simulate one run and compute evaluation metrics."""
    real_traj = run_data.trajectory
    initial_state = real_traj.state_at(0)
    t_end = float(real_traj.t[-1] - real_traj.t[0])
    dt = real_traj.dt  # Match real data timestep

    sim_traj = candidate_sim.simulate(initial_state, params, t_end, dt)

    div_time = metrics.divergence_time(sim_traj, real_traj)
    angle_err = metrics.angle_rmse(sim_traj, real_traj)
    energy_err = metrics.energy_rmse(sim_traj, real_traj, params)
    theta1_err, theta2_err = metrics.per_arm_rmse(sim_traj, real_traj)

    result = EvalResult(
        run_id=run_data.run_id,
        divergence_time=div_time,
        angle_rmse=angle_err,
        energy_rmse=energy_err,
        theta1_rmse=theta1_err,
        theta2_rmse=theta2_err,
        sim_duration=float(sim_traj.t[-1] - sim_traj.t[0]),
        real_duration=t_end,
    )
    return result, sim_traj


def _save_iteration_snapshot(
    iteration: int,
    eval_results: list[EvalResult],
    sim_trajs: dict[str, Trajectory],
    real_trajs: dict[str, Trajectory],
    params: PendulumParams,
    report: str,
):
    """Save all artifacts for one iteration to iterations/iter_NNN/."""
    iter_dir = ITERATIONS_DIR / f"iter_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Copy current candidate_sim.py
    src = SIM_LAB_DIR / "candidate_sim.py"
    if src.exists():
        shutil.copy2(src, iter_dir / "candidate_sim.py")

    # Save metrics as JSON
    metrics_data = {
        "iteration": iteration,
        "results": [],
        "summary": {},
    }
    div_times = []
    for r in eval_results:
        metrics_data["results"].append({
            "run_id": r.run_id,
            "divergence_time": r.divergence_time,
            "angle_rmse": r.angle_rmse,
            "energy_rmse": r.energy_rmse,
            "theta1_rmse": r.theta1_rmse,
            "theta2_rmse": r.theta2_rmse,
            "sim_duration": r.sim_duration,
            "real_duration": r.real_duration,
            "notes": r.notes,
        })
        div_times.append(r.divergence_time)

    metrics_data["summary"] = {
        "mean_divergence_time": statistics.mean(div_times) if div_times else 0,
        "median_divergence_time": statistics.median(div_times) if div_times else 0,
        "min_divergence_time": min(div_times) if div_times else 0,
        "max_divergence_time": max(div_times) if div_times else 0,
        "mean_angle_rmse": statistics.mean(r.angle_rmse for r in eval_results) if eval_results else 0,
    }

    with open(iter_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Save diagnostic report
    with open(iter_dir / "diagnostic_report.md", "w") as f:
        f.write(report)

    # Generate GIFs for best and worst runs (by divergence time)
    if eval_results:
        sorted_results = sorted(eval_results, key=lambda r: r.divergence_time)
        worst = sorted_results[0]   # Lowest divergence time = worst
        best = sorted_results[-1]   # Highest divergence time = best

        for tag, result in [("best", best), ("worst", worst)]:
            rid = result.run_id
            if rid in sim_trajs and rid in real_trajs:
                gif_path = str(iter_dir / f"{tag}_{rid}_overlay.gif")
                visualize.animate_overlay(
                    sim_trajs[rid], real_trajs[rid], params, gif_path,
                )
                energy_path = str(iter_dir / f"{tag}_{rid}_energy.png")
                visualize.plot_energy(
                    sim_trajs[rid], real_trajs[rid], params, energy_path,
                )

    print(f"  Iteration snapshot saved to: {iter_dir}")
    return metrics_data


def _print_summary(iteration: int, eval_results: list[EvalResult]):
    """Print a formatted summary table to stdout."""
    print(f"\n{'='*70}")
    print(f"  ITERATION {iteration} SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Run ID':<12} {'Div Time':>10} {'Angle RMSE':>12} {'Energy RMSE':>13} {'th1 RMSE':>10} {'th2 RMSE':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*13} {'-'*10} {'-'*10}")
    for r in sorted(eval_results, key=lambda x: x.run_id):
        print(
            f"  {r.run_id:<12} {r.divergence_time:>10.3f} {r.angle_rmse:>12.4f} "
            f"{r.energy_rmse:>13.4f} {r.theta1_rmse:>10.4f} {r.theta2_rmse:>10.4f}"
        )

    div_times = [r.divergence_time for r in eval_results]
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*13} {'-'*10} {'-'*10}")
    print(
        f"  {'MEAN':<12} {statistics.mean(div_times):>10.3f} "
        f"{statistics.mean(r.angle_rmse for r in eval_results):>12.4f} "
        f"{statistics.mean(r.energy_rmse for r in eval_results):>13.4f} "
        f"{statistics.mean(r.theta1_rmse for r in eval_results):>10.4f} "
        f"{statistics.mean(r.theta2_rmse for r in eval_results):>10.4f}"
    )
    print(f"  Divergence time range: [{min(div_times):.3f}, {max(div_times):.3f}]")
    print(f"{'='*70}\n")


def run_iteration(iteration: int, params: PendulumParams | None = None):
    """Execute one full evaluation iteration.

    1. Load test runs
    2. Simulate each with candidate_sim
    3. Compute metrics
    4. Generate diagnostics report
    5. Generate visualizations (best/worst)
    6. Save iteration snapshot
    7. Print summary
    """
    if params is None:
        params = PendulumParams()

    print(f"--- Iteration {iteration} ---")

    # Load test scenarios
    test_runs = scenarios.get_test_runs()
    if not test_runs:
        print("  No test runs found. Make sure data is downloaded.")
        return None

    print(f"  Evaluating {len(test_runs)} test runs...")

    # Evaluate each run
    eval_results: list[EvalResult] = []
    sim_trajs: dict[str, Trajectory] = {}
    real_trajs: dict[str, Trajectory] = {}

    for run_data in test_runs:
        print(f"    {run_data.run_id}...", end=" ", flush=True)
        result, sim_traj = _evaluate_run(run_data, params)
        eval_results.append(result)
        sim_trajs[run_data.run_id] = sim_traj
        real_trajs[run_data.run_id] = run_data.trajectory
        print(f"div={result.divergence_time:.3f}s, rmse={result.angle_rmse:.4f}")

    # Generate diagnostics report (diagnostics expects lists, not dicts)
    print("  Generating diagnostics report...")
    sim_traj_list = [sim_trajs[r.run_id] for r in eval_results]
    real_traj_list = [real_trajs[r.run_id] for r in eval_results]
    report = diagnostics.generate_report(eval_results, sim_traj_list, real_traj_list, params)

    # Save iteration snapshot
    metrics_data = _save_iteration_snapshot(
        iteration, eval_results, sim_trajs, real_trajs, params, report,
    )

    # Print summary
    _print_summary(iteration, eval_results)

    # Build IterationSnapshot for return
    div_times = [r.divergence_time for r in eval_results]
    snapshot = IterationSnapshot(
        iteration=iteration,
        eval_results=eval_results,
        mean_divergence_time=statistics.mean(div_times),
        median_divergence_time=statistics.median(div_times),
        mean_angle_rmse=statistics.mean(r.angle_rmse for r in eval_results),
        diagnostic_report=report,
    )
    return snapshot


def load_iteration_history() -> list[dict]:
    """Load metrics summaries from all past iterations for convergence plotting."""
    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("iter_"):
            metrics_path = d / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)
                summary = data.get("summary", {})
                history.append({
                    "iteration": data.get("iteration", 0),
                    "mean_divergence_time": summary.get("mean_divergence_time", 0),
                    "min_divergence_time": summary.get("min_divergence_time", 0),
                    "max_divergence_time": summary.get("max_divergence_time", 0),
                })
    return history


def main():
    parser = argparse.ArgumentParser(description="Run sim-vs-real evaluation loop")
    parser.add_argument(
        "--iteration", type=int, default=None,
        help="Iteration number (auto-detect if not given)",
    )
    args = parser.parse_args()

    iteration = args.iteration if args.iteration is not None else _next_iteration_number()

    snapshot = run_iteration(iteration)

    if snapshot is not None:
        # Update convergence plot with all past iterations
        history = load_iteration_history()
        if history:
            conv_path = str(ITERATIONS_DIR / "convergence.png")
            visualize.plot_convergence(history, conv_path)

    print("Done.")


if __name__ == "__main__":
    main()
