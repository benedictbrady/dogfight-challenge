"""Trajectory comparison metrics for sim vs real data."""

import numpy as np

from sim_interface import PendulumParams, Trajectory


def _truncate(sim: Trajectory, real: Trajectory) -> tuple[np.ndarray, np.ndarray, int]:
    """Find the common length and return truncation info."""
    n = min(len(sim), len(real))
    return sim, real, n


def divergence_time(
    sim: Trajectory,
    real: Trajectory,
    threshold: float = np.pi / 4,
) -> float:
    """Seconds until combined angle error exceeds threshold.

    Returns the time at which sqrt(dtheta1^2 + dtheta2^2) first exceeds
    the threshold. If it never exceeds, returns the trajectory duration.
    """
    _, _, n = _truncate(sim, real)
    err1 = sim.theta1[:n] - real.theta1[:n]
    err2 = sim.theta2[:n] - real.theta2[:n]
    combined = np.sqrt(err1**2 + err2**2)

    exceed = np.where(combined > threshold)[0]
    if len(exceed) == 0:
        return float(sim.t[n - 1])
    return float(sim.t[exceed[0]])


def angle_rmse(sim: Trajectory, real: Trajectory) -> float:
    """Combined angle RMSE across both arms (radians)."""
    _, _, n = _truncate(sim, real)
    err1 = sim.theta1[:n] - real.theta1[:n]
    err2 = sim.theta2[:n] - real.theta2[:n]
    return float(np.sqrt(np.mean(err1**2 + err2**2)))


def energy_rmse(sim: Trajectory, real: Trajectory, params: PendulumParams) -> float:
    """RMSE between energy trajectories."""
    _, _, n = _truncate(sim, real)
    e_sim = sim.energy(params)[:n]
    e_real = real.energy(params)[:n]
    return float(np.sqrt(np.mean((e_sim - e_real) ** 2)))


def per_arm_rmse(sim: Trajectory, real: Trajectory) -> tuple[float, float]:
    """Per-arm angle RMSE: (theta1_rmse, theta2_rmse)."""
    _, _, n = _truncate(sim, real)
    rmse1 = float(np.sqrt(np.mean((sim.theta1[:n] - real.theta1[:n]) ** 2)))
    rmse2 = float(np.sqrt(np.mean((sim.theta2[:n] - real.theta2[:n]) ** 2)))
    return rmse1, rmse2
