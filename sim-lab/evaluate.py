#!/usr/bin/env python3
"""Evaluate candidate simulator against real experimental data.

Usage:
    python evaluate.py [--runs test|calibration|all] [--output metrics.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sim_interface import EvalResult, PendulumParams, RunData, Trajectory
from candidate_sim import simulate
from metrics import angle_rmse, divergence_time, energy_rmse, per_arm_rmse
from scenarios import get_calibration_runs, get_test_runs, get_all_runs


def evaluate_run(
    run_data: RunData,
    params: PendulumParams,
) -> tuple[EvalResult, Trajectory]:
    """Run simulator from real initial conditions and compare."""
    real = run_data.trajectory
    initial = real.state_at(0)
    t_end = float(real.t[-1] - real.t[0])
    dt = real.dt

    sim_traj = simulate(initial, params, t_end, dt)

    div_t = divergence_time(sim_traj, real)
    a_rmse = angle_rmse(sim_traj, real)
    e_rmse = energy_rmse(sim_traj, real, params)
    t1_rmse, t2_rmse = per_arm_rmse(sim_traj, real)

    result = EvalResult(
        run_id=run_data.run_id,
        divergence_time=div_t,
        angle_rmse=a_rmse,
        energy_rmse=e_rmse,
        theta1_rmse=t1_rmse,
        theta2_rmse=t2_rmse,
        sim_duration=t_end,
        real_duration=t_end,
    )
    return result, sim_traj


def run_evaluation(
    run_data_list: list[RunData],
    params: PendulumParams,
) -> tuple[list[EvalResult], list[Trajectory], list[Trajectory]]:
    """Evaluate all specified runs."""
    results = []
    sim_trajs = []
    real_trajs = []

    for run_data in run_data_list:
        result, sim_traj = evaluate_run(run_data, params)
        results.append(result)
        sim_trajs.append(sim_traj)
        real_trajs.append(run_data.trajectory)

        print(
            f"  {run_data.run_id}: div_time={result.divergence_time:.2f}s  "
            f"angle_rmse={result.angle_rmse:.4f}  "
            f"energy_rmse={result.energy_rmse:.4f}"
        )

    return results, sim_trajs, real_trajs


def results_to_dict(results: list[EvalResult]) -> dict:
    """Serialize evaluation results to JSON-compatible dict."""
    per_run = []
    for r in results:
        per_run.append({
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

    div_times = [r.divergence_time for r in results]
    angle_rmses = [r.angle_rmse for r in results]

    return {
        "per_run": per_run,
        "aggregate": {
            "n_runs": len(results),
            "mean_divergence_time": float(np.mean(div_times)) if div_times else 0,
            "median_divergence_time": float(np.median(div_times)) if div_times else 0,
            "mean_angle_rmse": float(np.mean(angle_rmses)) if angle_rmses else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate candidate simulator")
    parser.add_argument(
        "--runs",
        choices=["test", "calibration", "all"],
        default="test",
        help="Which runs to evaluate (default: test)",
    )
    parser.add_argument(
        "--output",
        default="metrics.json",
        help="Output file for metrics (default: metrics.json)",
    )
    args = parser.parse_args()

    # Load runs via scenarios.py
    if args.runs == "test":
        run_data_list = get_test_runs()
    elif args.runs == "calibration":
        run_data_list = get_calibration_runs()
    else:
        run_data_list = get_all_runs()

    print(f"Evaluating {len(run_data_list)} runs ({args.runs})...")

    # Default params — ideal pendulum, no dissipation
    params = PendulumParams()

    results, _, _ = run_evaluation(run_data_list, params)

    if not results:
        print("No runs evaluated successfully.", file=sys.stderr)
        sys.exit(1)

    output = results_to_dict(results)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nMetrics saved to {output_path}")

    agg = output["aggregate"]
    print(f"\n=== Aggregate ===")
    print(f"  Runs evaluated: {agg['n_runs']}")
    print(f"  Mean divergence time:   {agg['mean_divergence_time']:.2f}s")
    print(f"  Median divergence time: {agg['median_divergence_time']:.2f}s")
    print(f"  Mean angle RMSE:        {agg['mean_angle_rmse']:.4f} rad")


if __name__ == "__main__":
    main()
