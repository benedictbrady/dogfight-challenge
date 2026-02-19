"""Visualization tools for double pendulum sim-vs-real comparison."""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from sim_interface import PendulumParams, Trajectory


def _pendulum_xy(theta1: float, theta2: float, L1: float, L2: float):
    """Convert angles to (x, y) positions of bob1 and bob2, pivot at origin."""
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return (x1, y1), (x2, y2)


def animate_overlay(
    sim_traj: Trajectory,
    real_traj: Trajectory,
    params: PendulumParams,
    output_path: str,
    fps: int = 30,
    max_frames: int = 300,
):
    """Create a GIF overlaying sim (red) and real (green) pendulums.

    Args:
        sim_traj: Simulated trajectory.
        real_traj: Real (ground-truth) trajectory.
        params: Pendulum physical parameters (for arm lengths).
        output_path: Path to write the .gif file.
        fps: Frames per second in the output GIF.
        max_frames: Maximum number of frames to render.
    """
    n_frames = min(len(sim_traj), len(real_traj), max_frames)
    # Subsample if the trajectory is longer than max_frames
    total_avail = min(len(sim_traj), len(real_traj))
    indices = np.linspace(0, total_avail - 1, n_frames, dtype=int)

    L1, L2 = params.L1, params.L2
    arm_span = L1 + L2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-arm_span * 1.3, arm_span * 1.3)
    ax.set_ylim(-arm_span * 1.3, arm_span * 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f0f0f0")
    ax.set_title("Double Pendulum: Sim (red) vs Real (green)")

    # Pivot point
    pivot_dot, = ax.plot(0, 0, "ko", markersize=8, zorder=5)

    # Real pendulum (green)
    real_arm1, = ax.plot([], [], "g-", linewidth=2.5, alpha=0.8, label="Real")
    real_arm2, = ax.plot([], [], "g-", linewidth=2.5, alpha=0.8)
    real_bob1, = ax.plot([], [], "go", markersize=10, alpha=0.8)
    real_bob2, = ax.plot([], [], "go", markersize=12, alpha=0.8)

    # Sim pendulum (red)
    sim_arm1, = ax.plot([], [], "r-", linewidth=2.5, alpha=0.8, label="Sim")
    sim_arm2, = ax.plot([], [], "r-", linewidth=2.5, alpha=0.8)
    sim_bob1, = ax.plot([], [], "ro", markersize=10, alpha=0.8)
    sim_bob2, = ax.plot([], [], "ro", markersize=12, alpha=0.8)

    # Text overlays
    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=11, verticalalignment="top", fontfamily="monospace",
    )
    error_text = ax.text(
        0.02, 0.90, "", transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        color="darkred",
    )
    ax.legend(loc="upper right")

    def _update(frame_num):
        idx = indices[frame_num]

        # Real
        r1, r2 = _pendulum_xy(
            real_traj.theta1[idx], real_traj.theta2[idx], L1, L2
        )
        real_arm1.set_data([0, r1[0]], [0, r1[1]])
        real_arm2.set_data([r1[0], r2[0]], [r1[1], r2[1]])
        real_bob1.set_data([r1[0]], [r1[1]])
        real_bob2.set_data([r2[0]], [r2[1]])

        # Sim
        s1, s2 = _pendulum_xy(
            sim_traj.theta1[idx], sim_traj.theta2[idx], L1, L2
        )
        sim_arm1.set_data([0, s1[0]], [0, s1[1]])
        sim_arm2.set_data([s1[0], s2[0]], [s1[1], s2[1]])
        sim_bob1.set_data([s1[0]], [s1[1]])
        sim_bob2.set_data([s2[0]], [s2[1]])

        # Angle error (sum of absolute differences in both angles)
        d_theta1 = abs(sim_traj.theta1[idx] - real_traj.theta1[idx])
        d_theta2 = abs(sim_traj.theta2[idx] - real_traj.theta2[idx])
        angle_err = d_theta1 + d_theta2

        t = real_traj.t[idx]
        time_text.set_text(f"t = {t:.2f} s")
        error_text.set_text(
            f"angle err: {np.degrees(angle_err):.1f} deg "
            f"(th1={np.degrees(d_theta1):.1f}, th2={np.degrees(d_theta2):.1f})"
        )

        return (
            real_arm1, real_arm2, real_bob1, real_bob2,
            sim_arm1, sim_arm2, sim_bob1, sim_bob2,
            time_text, error_text,
        )

    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 / fps, blit=True
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Saved overlay GIF: {output_path}")


def plot_energy(
    sim_traj: Trajectory,
    real_traj: Trajectory,
    params: PendulumParams,
    output_path: str,
):
    """Plot total mechanical energy over time for sim and real trajectories."""
    sim_energy = sim_traj.energy(params)
    real_energy = real_traj.energy(params)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(real_traj.t, real_energy, "g-", linewidth=1.5, label="Real", alpha=0.8)
    ax.plot(sim_traj.t, sim_energy, "r-", linewidth=1.5, label="Sim", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total Energy (J)")
    ax.set_title("Energy Comparison: Sim vs Real")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved energy plot: {output_path}")


def plot_convergence(iteration_data: list[dict], output_path: str):
    """Plot divergence time vs iteration number with error bars.

    Args:
        iteration_data: List of dicts with keys:
            - "iteration": int
            - "mean_divergence_time": float
            - "min_divergence_time": float
            - "max_divergence_time": float
        output_path: Path to save PNG.
    """
    if not iteration_data:
        print("  No iteration data for convergence plot, skipping.")
        return

    iters = [d["iteration"] for d in iteration_data]
    means = [d["mean_divergence_time"] for d in iteration_data]
    mins = [d["min_divergence_time"] for d in iteration_data]
    maxs = [d["max_divergence_time"] for d in iteration_data]

    lower_err = [m - lo for m, lo in zip(means, mins)]
    upper_err = [hi - m for m, hi in zip(means, maxs)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        iters, means, yerr=[lower_err, upper_err],
        fmt="o-", capsize=5, linewidth=2, markersize=8,
        color="steelblue", ecolor="lightcoral",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Divergence Time (s)")
    ax.set_title("Sim Accuracy Convergence Over Iterations")
    ax.grid(True, alpha=0.3)
    if len(iters) > 1:
        ax.set_xticks(iters)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved convergence plot: {output_path}")


def _load_trajectory_npz(path: str) -> Trajectory:
    """Load a trajectory from a .npz file."""
    data = np.load(path)
    return Trajectory(
        t=data["t"],
        theta1=data["theta1"],
        theta2=data["theta2"],
        omega1=data["omega1"],
        omega2=data["omega2"],
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize sim vs real comparison")
    parser.add_argument("--run-data", required=True, help="Path to real trajectory .npz")
    parser.add_argument("--sim-data", required=True, help="Path to sim trajectory .npz")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--L1", type=float, default=1.0, help="First arm length")
    parser.add_argument("--L2", type=float, default=1.0, help="Second arm length")
    args = parser.parse_args()

    real_traj = _load_trajectory_npz(args.run_data)
    sim_traj = _load_trajectory_npz(args.sim_data)
    params = PendulumParams(L1=args.L1, L2=args.L2)

    os.makedirs(args.output, exist_ok=True)

    animate_overlay(
        sim_traj, real_traj, params,
        os.path.join(args.output, "overlay.gif"),
    )
    plot_energy(
        sim_traj, real_traj, params,
        os.path.join(args.output, "energy.png"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
