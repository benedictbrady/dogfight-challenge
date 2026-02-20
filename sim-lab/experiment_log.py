"""Persistent experiment tracker for sim-lab.

Logs experiments to sim-lab/experiments/log.json as an append-only JSON list.
Provides functions to query, list, and plot experiment history.

Usage:
    from experiment_log import log_experiment, list_experiments, best_experiment

    exp_id = log_experiment(
        label="add viscous friction",
        code=open("candidate_sim.py").read(),
        params={"b1": 0.05, "b2": 0.03},
        score={"mean_div_time": 1.82, "median_div_time": 1.65, "mean_angle_rmse": 0.41},
        notes="First attempt at dissipation",
    )
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SIM_LAB_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SIM_LAB_DIR / "experiments"
LOG_PATH = EXPERIMENTS_DIR / "log.json"


def _ensure_dir():
    """Create the experiments directory if it does not exist."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_log() -> list[dict]:
    """Load the experiment log from disk. Returns empty list if file missing."""
    _ensure_dir()
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return data


def _save_log(entries: list[dict]):
    """Atomically write the experiment log via temp file + rename.

    This provides thread/process safety: either the full new file is visible
    or the old file remains. No partial writes.
    """
    _ensure_dir()
    # Write to a temp file in the same directory, then atomically rename.
    fd, tmp_path = tempfile.mkstemp(
        dir=str(EXPERIMENTS_DIR), suffix=".tmp", prefix="log_"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp_path, str(LOG_PATH))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _next_id(entries: list[dict]) -> int:
    """Return the next auto-increment experiment ID."""
    if not entries:
        return 1
    return max(e.get("id", 0) for e in entries) + 1


def log_experiment(
    label: str,
    code: str,
    params: dict,
    score: dict,
    notes: str = "",
) -> int:
    """Append an experiment to the log. Returns the experiment number.

    Args:
        label: Short human-readable description (e.g. "add viscous friction").
        code: Full source code of candidate_sim.py at time of experiment.
        params: Dict of physics parameters used (e.g. {"b1": 0.05}).
        score: Dict with at minimum: mean_div_time, median_div_time, mean_angle_rmse.
               May also contain per_run list and any other metrics.
        notes: Optional free-text notes.

    Returns:
        The integer experiment ID assigned to this entry.
    """
    # Validate required score keys
    required_keys = {"mean_div_time", "median_div_time", "mean_angle_rmse"}
    missing = required_keys - set(score.keys())
    if missing:
        raise ValueError(f"score dict missing required keys: {missing}")

    entries = _load_log()
    exp_id = _next_id(entries)

    entry = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "code_hash": hashlib.md5(code.encode("utf-8")).hexdigest(),
        "params": params,
        "score": score,
        "notes": notes,
    }

    entries.append(entry)
    _save_log(entries)
    return exp_id


def list_experiments() -> str:
    """Return a formatted table of all experiments, sorted by mean_div_time descending.

    Columns: ID, Label, Mean Div Time, Angle RMSE, Notes (truncated to 30 chars).
    """
    entries = _load_log()
    if not entries:
        return "No experiments logged yet."

    # Sort by mean_div_time descending (best first)
    entries_sorted = sorted(
        entries,
        key=lambda e: e.get("score", {}).get("mean_div_time", 0),
        reverse=True,
    )

    # Build formatted table
    header = f"{'ID':>4}  {'Label':<30}  {'Mean Div Time':>14}  {'Angle RMSE':>11}  {'Notes':<30}"
    sep = f"{'─'*4}  {'─'*30}  {'─'*14}  {'─'*11}  {'─'*30}"
    lines = [header, sep]

    for e in entries_sorted:
        exp_id = e.get("id", "?")
        label = e.get("label", "")[:30]
        score = e.get("score", {})
        mean_div = score.get("mean_div_time", 0)
        angle_rmse = score.get("mean_angle_rmse", 0)
        notes = e.get("notes", "")[:30]

        lines.append(
            f"{exp_id:>4}  {label:<30}  {mean_div:>14.3f}  {angle_rmse:>11.4f}  {notes:<30}"
        )

    return "\n".join(lines)


def best_experiment() -> Optional[dict]:
    """Return the experiment entry with the highest mean_div_time.

    Returns None if no experiments have been logged.
    """
    entries = _load_log()
    if not entries:
        return None
    return max(
        entries,
        key=lambda e: e.get("score", {}).get("mean_div_time", 0),
    )


def get_experiment(exp_id: int) -> Optional[dict]:
    """Get a specific experiment by ID.

    Returns None if no experiment with that ID exists.
    """
    entries = _load_log()
    for e in entries:
        if e.get("id") == exp_id:
            return e
    return None


def plot_progress(output_path: Optional[str] = None):
    """Plot mean_div_time over experiment number and save as PNG.

    Args:
        output_path: Where to save the PNG. Defaults to experiments/progress.png
                     relative to this module's directory.

    X-axis: experiment ID (chronological order).
    Y-axis: mean divergence time (seconds).
    Also marks the best experiment with a star.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = str(EXPERIMENTS_DIR / "progress.png")

    entries = _load_log()
    if not entries:
        print("No experiments to plot.")
        return

    # Sort by ID for chronological order
    entries_sorted = sorted(entries, key=lambda e: e.get("id", 0))

    ids = [e["id"] for e in entries_sorted]
    div_times = [e.get("score", {}).get("mean_div_time", 0) for e in entries_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ids, div_times, "o-", color="#2563eb", linewidth=1.5, markersize=6)

    # Mark the best experiment
    best_idx = div_times.index(max(div_times))
    ax.plot(
        ids[best_idx], div_times[best_idx],
        "*", color="#dc2626", markersize=16, zorder=5,
        label=f"Best: #{ids[best_idx]} ({div_times[best_idx]:.3f}s)",
    )

    ax.set_xlabel("Experiment ID")
    ax.set_ylabel("Mean Divergence Time (s)")
    ax.set_title("Sim-Lab: Experiment Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ensure integer ticks on x-axis
    ax.set_xticks(ids)

    # Ensure output directory exists
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Progress plot saved to {out}")
