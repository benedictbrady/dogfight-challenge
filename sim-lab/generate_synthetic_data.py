#!/usr/bin/env python3
"""Generate synthetic 'real' double pendulum data with known dissipation.

When the IBM CDN is unavailable, this creates ground-truth data that has the
same structure as the real dataset (CSV pixel coordinates at 438fps).

The hidden truth parameters include viscous friction, air drag, and Coulomb
friction — the optimizer must discover these by fitting the ideal simulator.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_DIR = os.path.join(DATA_DIR, "synthetic_runs")

# --- Hidden ground truth parameters (the optimizer doesn't see these) ---
TRUE_L1 = 170.0      # pixels
TRUE_L2 = 135.0      # pixels
TRUE_M1 = 1.0
TRUE_M2 = 0.8
TRUE_G = 9.81
TRUE_B1 = 0.12       # viscous friction, joint 1
TRUE_B2 = 0.08       # viscous friction, joint 2
TRUE_C1 = 0.03       # Coulomb friction, joint 1
TRUE_C2 = 0.02       # Coulomb friction, joint 2
TRUE_CD1 = 0.005     # air drag, arm 1
TRUE_CD2 = 0.004     # air drag, arm 2

FPS = 438.0
PIVOT_X = 320.0       # pivot pixel position
PIVOT_Y = 100.0
NOISE_STD = 0.5       # pixel noise from tracking

# 10 different initial conditions (varying energy levels and configurations)
INITIAL_CONDITIONS = [
    # (theta1, theta2, omega1, omega2)
    (2.0,  2.5, 0.0, 0.0),      # High energy, both arms out
    (1.5, -1.0, 0.0, 0.0),      # Moderate energy, opposing
    (2.8,  1.0, 0.0, 0.0),      # Very high theta1
    (0.5,  3.0, 0.0, 0.0),      # High theta2
    (1.0,  1.0, 2.0, -1.0),     # Moderate angles, nonzero velocity
    (-2.0,  2.0, 0.0, 0.0),     # Mirror symmetric
    (1.2,  1.8, -0.5, 0.5),     # Mixed
    (2.5,  0.3, 0.0, 0.0),      # Mostly arm 1
    (0.3,  2.5, 0.0, 0.0),      # Mostly arm 2
    (1.8, -1.8, 1.0, 1.0),      # High energy with velocity
]
DURATIONS = [8.0] * 10  # seconds each


def _true_eom(t, y):
    """Double pendulum equations with the hidden dissipation terms."""
    theta1, theta2, omega1, omega2 = y
    L1, L2 = TRUE_L1 / 100.0, TRUE_L2 / 100.0  # Convert to 'meters' scale
    m1, m2, g = TRUE_M1, TRUE_M2, TRUE_G

    delta = theta2 - theta1
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)

    denom = m1 + m2 * (1 - cos_d**2)

    # Ideal Lagrangian accelerations
    alpha1_ideal = (
        m2 * L1 * omega1**2 * sin_d * cos_d
        + m2 * g * np.sin(theta2) * cos_d
        + m2 * L2 * omega2**2 * sin_d
        - (m1 + m2) * g * np.sin(theta1)
    ) / (L1 * denom)

    alpha2_ideal = (
        -m2 * L2 * omega2**2 * sin_d * cos_d
        + (m1 + m2) * g * np.sin(theta1) * cos_d
        - (m1 + m2) * L1 * omega1**2 * sin_d
        - (m1 + m2) * g * np.sin(theta2)
    ) / (L2 * denom)

    # Dissipation terms
    visc1 = -TRUE_B1 * omega1
    visc2 = -TRUE_B2 * omega2
    coulomb1 = -TRUE_C1 * np.sign(omega1) if abs(omega1) > 1e-6 else 0.0
    coulomb2 = -TRUE_C2 * np.sign(omega2) if abs(omega2) > 1e-6 else 0.0
    drag1 = -TRUE_CD1 * abs(omega1) * omega1
    drag2 = -TRUE_CD2 * abs(omega2) * omega2

    alpha1 = alpha1_ideal + visc1 + coulomb1 + drag1
    alpha2 = alpha2_ideal + visc2 + coulomb2 + drag2

    return np.array([omega1, omega2, alpha1, alpha2])


def _angles_to_pixels(theta1_arr, theta2_arr):
    """Convert angles to pixel coordinates (pivot at PIVOT_X, PIVOT_Y)."""
    # In pixel space, y increases downward, angles from vertical-downward
    x1 = PIVOT_X + TRUE_L1 * np.sin(theta1_arr)
    y1 = PIVOT_Y + TRUE_L1 * np.cos(theta1_arr)  # cos because angle from downward
    x2 = x1 + TRUE_L2 * np.sin(theta2_arr)
    y2 = y1 + TRUE_L2 * np.cos(theta2_arr)
    return x1, y1, x2, y2


def generate_run(ic, duration, seed):
    """Generate one synthetic run."""
    rng = np.random.RandomState(seed)
    y0 = np.array(ic)
    dt = 1.0 / FPS
    t_eval = np.arange(0, duration, dt)

    sol = solve_ivp(
        fun=_true_eom,
        t_span=(0, duration),
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    theta1 = sol.y[0]
    theta2 = sol.y[1]
    x1, y1, x2, y2 = _angles_to_pixels(theta1, theta2)

    # Add tracking noise
    noise = rng.normal(0, NOISE_STD, size=(6, len(theta1)))
    x_red = np.full_like(theta1, PIVOT_X) + noise[0]
    y_red = np.full_like(theta1, PIVOT_Y) + noise[1]
    x_green = x1 + noise[2]
    y_green = y1 + noise[3]
    x_blue = x2 + noise[4]
    y_blue = y2 + noise[5]

    return x_red, y_red, x_green, y_green, x_blue, y_blue


def generate_all():
    """Generate all synthetic runs and save as CSVs."""
    os.makedirs(CSV_DIR, exist_ok=True)

    for i, (ic, dur) in enumerate(zip(INITIAL_CONDITIONS, DURATIONS)):
        run_id = f"dp_run_{i:02d}"
        print(f"  Generating {run_id}: IC={ic}, duration={dur}s...")

        x_red, y_red, x_green, y_green, x_blue, y_blue = generate_run(
            ic, dur, seed=1000 + i
        )

        csv_path = os.path.join(CSV_DIR, f"{run_id}.csv")
        header = "x_red,y_red,x_green,y_green,x_blue,y_blue"
        data = np.column_stack([x_red, y_red, x_green, y_green, x_blue, y_blue])
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

        print(f"    -> {len(x_red)} frames, {dur}s at {FPS}fps")

    print(f"\nGenerated {len(INITIAL_CONDITIONS)} synthetic runs in {CSV_DIR}")
    print(f"Ground truth dissipation: b1={TRUE_B1}, b2={TRUE_B2}, "
          f"c1={TRUE_C1}, c2={TRUE_C2}, cd1={TRUE_CD1}, cd2={TRUE_CD2}")


if __name__ == "__main__":
    generate_all()
