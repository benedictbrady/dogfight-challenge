"""Candidate double pendulum simulator.

Implements the standard Lagrangian equations of motion for an ideal double
pendulum. Deliberately omits friction, drag, and all dissipation — the AI
improvement loop discovers these missing physics in later iterations.
"""

import numpy as np
from scipy.integrate import solve_ivp

from sim_interface import PendulumParams, PendulumState, Trajectory


def equations_of_motion(t: float, y: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Standard Lagrangian double pendulum equations (no dissipation).

    State vector y = [theta1, theta2, omega1, omega2].
    Returns dy/dt.
    """
    theta1, theta2, omega1, omega2 = y
    L1, L2, m1, m2, g = params.L1, params.L2, params.m1, params.m2, params.g

    delta = theta2 - theta1
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)

    # Mass matrix denominator
    denom = m1 + m2 * (1 - cos_d**2)

    # Angular accelerations from Lagrangian mechanics
    alpha1 = (
        m2 * L1 * omega1**2 * sin_d * cos_d
        + m2 * g * np.sin(theta2) * cos_d
        + m2 * L2 * omega2**2 * sin_d
        - (m1 + m2) * g * np.sin(theta1)
    ) / (L1 * denom)

    alpha2 = (
        -m2 * L2 * omega2**2 * sin_d * cos_d
        + (m1 + m2) * g * np.sin(theta1) * cos_d
        - (m1 + m2) * L1 * omega1**2 * sin_d
        - (m1 + m2) * g * np.sin(theta2)
    ) / (L2 * denom)

    return np.array([omega1, omega2, alpha1, alpha2])


def simulate(
    initial_state: PendulumState,
    params: PendulumParams,
    t_end: float,
    dt: float = 1 / 438,
) -> Trajectory:
    """Simulate the double pendulum from initial conditions.

    Args:
        initial_state: Starting angles and angular velocities.
        params: Physical parameters (lengths, masses, gravity).
        t_end: Duration to simulate (seconds).
        dt: Output timestep (default 1/438 to match ~438 fps tracking data).

    Returns:
        Trajectory with uniformly-spaced samples at the requested dt.
    """
    y0 = initial_state.to_array()
    t_eval = np.arange(0, t_end, dt)
    if len(t_eval) == 0:
        t_eval = np.array([0.0])

    sol = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, params),
        t_span=(0, t_end),
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return Trajectory(
        t=sol.t,
        theta1=sol.y[0],
        theta2=sol.y[1],
        omega1=sol.y[2],
        omega2=sol.y[3],
    )
