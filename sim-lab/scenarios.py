#!/usr/bin/env python3
"""Select calibration and test run sets from preprocessed data.

Splits the preprocessed runs into:
  - 5 calibration runs (used for tuning dissipation parameters)
  - 5 test runs (held out for final evaluation)

The split is deterministic — always the same runs in each set.
"""

import glob
import os
import sys

import numpy as np

from sim_interface import Trajectory, RunData

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Deterministic split: pick runs by index
# Use a fixed seed to shuffle, then take first 5 as calibration, next 5 as test
_SPLIT_SEED = 42
_N_CALIBRATION = 5
_N_TEST = 5


def _load_run(npz_path: str) -> RunData:
    """Load a single .npz file into a RunData object."""
    data = np.load(npz_path, allow_pickle=True)
    run_id = str(data["run_id"])
    fps = float(data["fps"])
    L1_px = float(data["L1_px"])
    L2_px = float(data["L2_px"])
    trajectory = Trajectory(
        t=data["t"],
        theta1=data["theta1"],
        theta2=data["theta2"],
        omega1=data["omega1"],
        omega2=data["omega2"],
    )
    return RunData(
        run_id=run_id,
        trajectory=trajectory,
        L1_px=L1_px,
        L2_px=L2_px,
        fps=fps,
    )


def _get_sorted_npz_files() -> list:
    """Get sorted list of processed .npz files."""
    pattern = os.path.join(PROCESSED_DIR, "run_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No processed .npz files found in {PROCESSED_DIR}.")
        print("Run preprocess.py first.")
        sys.exit(1)
    return files


def _get_split_indices(n_total: int):
    """Return (calibration_indices, test_indices) using deterministic shuffle."""
    rng = np.random.RandomState(_SPLIT_SEED)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    cal_idx = sorted(indices[:_N_CALIBRATION])
    test_idx = sorted(indices[_N_CALIBRATION : _N_CALIBRATION + _N_TEST])
    return cal_idx, test_idx


def get_all_runs() -> list:
    """Load all processed runs as RunData objects."""
    files = _get_sorted_npz_files()
    return [_load_run(f) for f in files]


def get_calibration_runs() -> list:
    """Return 5 calibration runs (for tuning dissipation parameters)."""
    files = _get_sorted_npz_files()
    cal_idx, _ = _get_split_indices(len(files))
    return [_load_run(files[i]) for i in cal_idx]


def get_test_runs() -> list:
    """Return 5 test runs (held out for final evaluation)."""
    files = _get_sorted_npz_files()
    _, test_idx = _get_split_indices(len(files))
    return [_load_run(files[i]) for i in test_idx]


def get_runs(which: str = "test") -> list:
    """Get run IDs by set name: 'test', 'calibration', or 'all'."""
    if which == "calibration":
        return [r.run_id for r in get_calibration_runs()]
    elif which == "all":
        return [r.run_id for r in get_all_runs()]
    else:
        return [r.run_id for r in get_test_runs()]


if __name__ == "__main__":
    files = _get_sorted_npz_files()
    n = len(files)
    cal_idx, test_idx = _get_split_indices(n)

    print(f"Total runs: {n}")
    print(f"Calibration runs (indices): {cal_idx}")
    print(f"Test runs (indices):        {test_idx}")

    print("\nLoading calibration runs...")
    cal_runs = get_calibration_runs()
    for r in cal_runs:
        traj = r.trajectory
        print(
            f"  {r.run_id}: {len(traj)} frames, "
            f"{traj.t[-1]:.1f}s, L1={r.L1_px:.1f}px, L2={r.L2_px:.1f}px"
        )

    print("\nLoading test runs...")
    test_runs = get_test_runs()
    for r in test_runs:
        traj = r.trajectory
        print(
            f"  {r.run_id}: {len(traj)} frames, "
            f"{traj.t[-1]:.1f}s, L1={r.L1_px:.1f}px, L2={r.L2_px:.1f}px"
        )
