"""Shared types for the double pendulum simulator."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PendulumParams:
    """Physical parameters of a double pendulum."""
    L1: float = 1.0          # Length of first arm (meters or pixel-scale)
    L2: float = 1.0          # Length of second arm
    m1: float = 1.0          # Mass of first bob
    m2: float = 1.0          # Mass of second bob
    g: float = 9.81          # Gravitational acceleration
    # Dissipation parameters (initially zero — AI discovers these)
    b1: float = 0.0          # Viscous friction coefficient, joint 1
    b2: float = 0.0          # Viscous friction coefficient, joint 2
    c1: float = 0.0          # Coulomb friction coefficient, joint 1
    c2: float = 0.0          # Coulomb friction coefficient, joint 2
    cd1: float = 0.0         # Air drag coefficient, arm 1
    cd2: float = 0.0         # Air drag coefficient, arm 2


@dataclass
class PendulumState:
    """Instantaneous state of a double pendulum."""
    theta1: float = 0.0      # Angle of first arm (rad, from vertical)
    theta2: float = 0.0      # Angle of second arm (rad, from vertical)
    omega1: float = 0.0      # Angular velocity of first arm (rad/s)
    omega2: float = 0.0      # Angular velocity of second arm (rad/s)

    def to_array(self) -> np.ndarray:
        return np.array([self.theta1, self.theta2, self.omega1, self.omega2])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "PendulumState":
        return cls(theta1=arr[0], theta2=arr[1], omega1=arr[2], omega2=arr[3])


@dataclass
class Trajectory:
    """Time series of pendulum states."""
    t: np.ndarray             # Time values (N,)
    theta1: np.ndarray        # First arm angle over time (N,)
    theta2: np.ndarray        # Second arm angle over time (N,)
    omega1: np.ndarray        # First arm angular velocity (N,)
    omega2: np.ndarray        # Second arm angular velocity (N,)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> float:
        """Average timestep."""
        return float(np.mean(np.diff(self.t)))

    def state_at(self, idx: int) -> PendulumState:
        return PendulumState(
            theta1=float(self.theta1[idx]),
            theta2=float(self.theta2[idx]),
            omega1=float(self.omega1[idx]),
            omega2=float(self.omega2[idx]),
        )

    def energy(self, params: PendulumParams) -> np.ndarray:
        """Compute total mechanical energy at each timestep."""
        L1, L2, m1, m2, g = params.L1, params.L2, params.m1, params.m2, params.g
        # Positions of bobs
        x1 = L1 * np.sin(self.theta1)
        y1 = -L1 * np.cos(self.theta1)
        x2 = x1 + L2 * np.sin(self.theta2)
        y2 = y1 - L2 * np.cos(self.theta2)
        # Velocities
        vx1 = L1 * self.omega1 * np.cos(self.theta1)
        vy1 = L1 * self.omega1 * np.sin(self.theta1)
        vx2 = vx1 + L2 * self.omega2 * np.cos(self.theta2)
        vy2 = vy1 + L2 * self.omega2 * np.sin(self.theta2)
        # Kinetic energy
        KE = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
        # Potential energy (reference: pivot point)
        PE = m1 * g * y1 + m2 * g * y2
        return KE + PE


@dataclass
class RunData:
    """Preprocessed data from a single experimental run."""
    run_id: str               # e.g. "run_01"
    trajectory: Trajectory    # Angles + velocities extracted from video
    L1_px: float              # First arm length in pixels
    L2_px: float              # Second arm length in pixels
    fps: float                # Frame rate of the recording


@dataclass
class EvalResult:
    """Evaluation metrics for a single sim run vs real data."""
    run_id: str
    divergence_time: float          # Seconds until angle error > threshold
    angle_rmse: float               # Overall angle RMSE (radians)
    energy_rmse: float              # Energy trajectory RMSE
    theta1_rmse: float              # Per-arm angle RMSE
    theta2_rmse: float              # Per-arm angle RMSE
    sim_duration: float             # Total sim time
    real_duration: float            # Total real data time
    notes: str = ""                 # Any diagnostic notes


@dataclass
class IterationSnapshot:
    """Record of one improvement iteration."""
    iteration: int
    eval_results: list              # List of EvalResult
    mean_divergence_time: float
    median_divergence_time: float
    mean_angle_rmse: float
    diagnostic_report: str
    code_diff: str = ""             # What changed in candidate_sim.py
