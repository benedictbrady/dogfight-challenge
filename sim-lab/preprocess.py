#!/usr/bin/env python3
"""Convert pixel positions from the Double Pendulum dataset to angles + angular velocities.

The IBM dataset provides CSV files with columns:
  x_red, y_red, x_green, y_green, x_blue, y_blue

where:
  Red = pivot (fixed point at top)
  Green = joint between first and second arm
  Blue = tip of second arm

This script:
  1. Reads each CSV run
  2. Computes arm lengths L1 (red->green) and L2 (green->blue) in pixels
  3. Converts pixel coordinates to angles theta1, theta2 from vertical-downward
  4. Computes angular velocities omega1, omega2 via Savitzky-Golay smoothed derivatives
  5. Saves each run as .npz in data/processed/
"""

import glob
import os
import sys

import numpy as np
from scipy.signal import savgol_filter

# Dataset parameters
FPS = 438.0  # Approximate frame rate
SAVGOL_WINDOW = 15  # Must be odd; ~34ms window at 438fps
SAVGOL_ORDER = 3

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def find_csv_dir():
    """Find the directory containing CSV files after extraction."""
    for entry in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, entry)
        if os.path.isdir(full) and entry != "processed":
            csvs = sorted(glob.glob(os.path.join(full, "*.csv")))
            if csvs:
                return full, csvs
    # Also check DATA_DIR itself
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if csvs:
        return DATA_DIR, csvs
    return None, []


def load_csv(path):
    """Load a CSV file with pixel coordinates. Returns dict of arrays."""
    # The CSVs have a header row: x_red,y_red,x_green,y_green,x_blue,y_blue
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {
        "x_red": data["x_red"],
        "y_red": data["y_red"],
        "x_green": data["x_green"],
        "y_green": data["y_green"],
        "x_blue": data["x_blue"],
        "y_blue": data["y_blue"],
    }


def pixel_to_angles(coords):
    """Convert pixel coordinates to angles from vertical-downward.

    In pixel space, y increases downward. We define angles measured from the
    downward vertical (positive y direction in pixel space).

    theta1 = angle of arm 1 (pivot -> joint) from downward vertical
    theta2 = angle of arm 2 (joint -> tip) from downward vertical

    Using atan2 convention: theta = atan2(dx, dy) where dx is horizontal
    displacement and dy is downward displacement. This gives 0 when pointing
    straight down.
    """
    # Displacements for arm 1: pivot (red) -> joint (green)
    dx1 = coords["x_green"] - coords["x_red"]
    dy1 = coords["y_green"] - coords["y_red"]  # positive = downward in pixel space

    # Displacements for arm 2: joint (green) -> tip (blue)
    dx2 = coords["x_blue"] - coords["x_green"]
    dy2 = coords["y_blue"] - coords["y_green"]

    # Angles from downward vertical: atan2(horizontal, downward)
    theta1 = np.arctan2(dx1, dy1)
    theta2 = np.arctan2(dx2, dy2)

    # Arm lengths in pixels
    L1 = np.sqrt(dx1**2 + dy1**2)
    L2 = np.sqrt(dx2**2 + dy2**2)

    return theta1, theta2, L1, L2


def unwrap_and_smooth_angles(theta):
    """Unwrap angle discontinuities and lightly smooth."""
    return np.unwrap(theta)


def compute_angular_velocity(theta, dt):
    """Compute angular velocity using Savitzky-Golay derivative filter.

    This simultaneously smooths and differentiates, which is more robust
    than smooth-then-differentiate for noisy tracking data.
    """
    # savgol_filter with deriv=1 gives the first derivative
    # delta=dt makes it return derivative in physical units (rad/s)
    window = min(SAVGOL_WINDOW, len(theta))
    if window % 2 == 0:
        window -= 1
    if window < SAVGOL_ORDER + 1:
        # Fallback: simple finite differences
        omega = np.gradient(theta, dt)
        return omega
    omega = savgol_filter(theta, window, SAVGOL_ORDER, deriv=1, delta=dt)
    return omega


def preprocess_run(csv_path, run_id):
    """Preprocess a single run from CSV to angles + velocities."""
    coords = load_csv(csv_path)
    n_frames = len(coords["x_red"])

    # Convert to angles
    theta1_raw, theta2_raw, L1_arr, L2_arr = pixel_to_angles(coords)

    # Compute median arm lengths (should be roughly constant per run)
    L1_px = float(np.median(L1_arr))
    L2_px = float(np.median(L2_arr))

    # Report arm length variation as a quality check
    L1_std = float(np.std(L1_arr))
    L2_std = float(np.std(L2_arr))

    # Unwrap angles to avoid discontinuities at +/- pi
    theta1 = unwrap_and_smooth_angles(theta1_raw)
    theta2 = unwrap_and_smooth_angles(theta2_raw)

    # Time array
    dt = 1.0 / FPS
    t = np.arange(n_frames) * dt

    # Compute angular velocities via smoothed derivative
    omega1 = compute_angular_velocity(theta1, dt)
    omega2 = compute_angular_velocity(theta2, dt)

    return {
        "run_id": run_id,
        "t": t,
        "theta1": theta1,
        "theta2": theta2,
        "omega1": omega1,
        "omega2": omega2,
        "L1_px": L1_px,
        "L2_px": L2_px,
        "L1_std": L1_std,
        "L2_std": L2_std,
        "fps": FPS,
        "n_frames": n_frames,
    }


def preprocess_all():
    """Preprocess all CSV runs and save as .npz files."""
    csv_dir, csv_files = find_csv_dir()
    if not csv_files:
        print("No CSV files found. Run download_data.py first.")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in {csv_dir}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for i, csv_path in enumerate(csv_files):
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        run_id = f"run_{i:02d}"

        print(f"  [{i+1}/{len(csv_files)}] Processing {basename} -> {run_id}...")
        result = preprocess_run(csv_path, run_id)

        # Save as compressed npz
        out_path = os.path.join(PROCESSED_DIR, f"{run_id}.npz")
        np.savez_compressed(
            out_path,
            run_id=result["run_id"],
            t=result["t"],
            theta1=result["theta1"],
            theta2=result["theta2"],
            omega1=result["omega1"],
            omega2=result["omega2"],
            L1_px=result["L1_px"],
            L2_px=result["L2_px"],
            fps=result["fps"],
        )

        duration = result["t"][-1] - result["t"][0]
        print(
            f"    {result['n_frames']} frames, {duration:.1f}s, "
            f"L1={result['L1_px']:.1f}px (std={result['L1_std']:.2f}), "
            f"L2={result['L2_px']:.1f}px (std={result['L2_std']:.2f})"
        )

    print(f"\nDone. Processed {len(csv_files)} runs -> {PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_all()
