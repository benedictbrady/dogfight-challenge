#!/usr/bin/env python3
"""Parse TensorBoard event files into structured metrics.json for the dashboard.

Usage:
    # Parse a single run directory
    python parse_tb.py path/to/run_dir

    # Parse all run directories under a parent directory
    python parse_tb.py --all path/to/parent_dir

Output goes to training/dashboard_data/{run_name}/metrics.json
Also copies pool.json and config.json from the run directory if they exist.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from tbparse import SummaryReader
except ImportError:
    print("ERROR: tbparse not installed. Run: pip install tbparse", file=sys.stderr)
    sys.exit(1)


# Output directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_DATA_DIR = SCRIPT_DIR / "dashboard_data"


def find_event_files(run_dir: Path) -> bool:
    """Check if a directory (or its subdirectories) contains TensorBoard event files."""
    for _ in run_dir.rglob("events.out.tfevents.*"):
        return True
    return False


def parse_run(run_dir: Path) -> dict | None:
    """Parse a single TensorBoard run directory into structured metrics.

    Returns the metrics dict, or None if no event files found.
    """
    run_dir = run_dir.resolve()
    run_name = run_dir.name

    if not run_dir.is_dir():
        print(f"  SKIP: {run_dir} is not a directory")
        return None

    if not find_event_files(run_dir):
        print(f"  SKIP: {run_name} — no TensorBoard event files found")
        return None

    print(f"  Parsing: {run_name}")

    # Read all scalar data from the run directory
    reader = SummaryReader(str(run_dir), pivot=False)
    scalars_df = reader.scalars

    if scalars_df.empty:
        print(f"  SKIP: {run_name} — no scalar data found")
        return None

    # Build the metrics structure by categorizing tags
    # Tags follow the pattern: category/metric_name (e.g., losses/policy_loss)
    categories: dict[str, dict[str, list[dict]]] = {}

    tags = scalars_df["tag"].unique()
    for tag in sorted(tags):
        # Split tag into category and metric name
        parts = tag.split("/", 1)
        if len(parts) == 2:
            category, metric = parts
        else:
            # Tags without a slash go into a "misc" category
            category = "misc"
            metric = parts[0]

        # Normalize category name for the output structure
        category = category.strip()
        metric = metric.strip()

        if category not in categories:
            categories[category] = {}

        # Extract data points for this tag
        tag_data = scalars_df[scalars_df["tag"] == tag].sort_values("step")
        points = []
        for _, row in tag_data.iterrows():
            points.append({
                "step": int(row["step"]),
                "wall_time": float(row["wall_time"]),
                "value": float(row["value"]),
            })

        categories[category][metric] = points

    # Build the output structure with expected top-level keys
    metrics = {
        "run_name": run_name,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "losses": categories.pop("losses", {}),
        "charts": categories.pop("charts", {}),
        "selfplay": categories.pop("selfplay", {}),
        "eval": categories.pop("eval", {}),
    }

    # Any remaining categories get included as-is
    for cat_name, cat_data in categories.items():
        metrics[cat_name] = cat_data

    # Set up output directory
    out_dir = DASHBOARD_DATA_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics.json
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    -> {metrics_path}")

    # Copy config.json if it exists in the run directory
    config_src = run_dir / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, out_dir / "config.json")
        print(f"    -> copied config.json")

    # Copy pool.json — check both run_dir/pool.json and run_dir/pool/pool.json
    pool_src = run_dir / "pool.json"
    if not pool_src.exists():
        pool_src = run_dir / "pool" / "pool.json"
    if pool_src.exists():
        shutil.copy2(pool_src, out_dir / "pool.json")
        print(f"    -> copied pool.json")

    # Print summary
    print_summary(run_name, metrics)

    return metrics


def print_summary(run_name: str, metrics: dict) -> None:
    """Print summary stats for a parsed run."""
    total_metrics = 0
    total_points = 0
    min_step = float("inf")
    max_step = float("-inf")

    # Iterate over all categories in the metrics dict
    skip_keys = {"run_name", "parsed_at"}
    for key, value in metrics.items():
        if key in skip_keys or not isinstance(value, dict):
            continue
        for metric_name, points in value.items():
            if not isinstance(points, list):
                continue
            total_metrics += 1
            total_points += len(points)
            if points:
                steps = [p["step"] for p in points]
                min_step = min(min_step, min(steps))
                max_step = max(max_step, max(steps))

    step_range = f"{min_step:,} - {max_step:,}" if total_metrics > 0 else "N/A"
    print(f"    Summary: {total_metrics} metrics, {total_points:,} data points, steps {step_range}")

    # Show per-category breakdown
    for key, value in metrics.items():
        if key in skip_keys or not isinstance(value, dict):
            continue
        if value:
            metric_names = list(value.keys())
            print(f"      {key}: {', '.join(metric_names)}")


def parse_all(parent_dir: Path) -> None:
    """Parse all run directories under a parent directory."""
    parent_dir = parent_dir.resolve()
    if not parent_dir.is_dir():
        print(f"ERROR: {parent_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    subdirs = sorted([d for d in parent_dir.iterdir() if d.is_dir()])
    if not subdirs:
        print(f"No subdirectories found in {parent_dir}")
        return

    print(f"Scanning {len(subdirs)} directories in {parent_dir}...")
    parsed = 0
    skipped = 0

    for subdir in subdirs:
        result = parse_run(subdir)
        if result is not None:
            parsed += 1
        else:
            skipped += 1

    print(f"\nDone: {parsed} runs parsed, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(
        description="Parse TensorBoard event files into metrics.json for the dashboard."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a run directory (or parent directory with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="parse_all_runs",
        help="Parse all subdirectories in the given path as separate runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Override output directory (default: {DASHBOARD_DATA_DIR})",
    )

    args = parser.parse_args()

    if args.output_dir:
        global DASHBOARD_DATA_DIR
        DASHBOARD_DATA_DIR = args.output_dir.resolve()

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {DASHBOARD_DATA_DIR}\n")

    if args.parse_all_runs:
        parse_all(args.path)
    else:
        result = parse_run(args.path)
        if result is None:
            print("\nNo metrics parsed. Ensure the directory contains TensorBoard event files.")
            sys.exit(1)
        print("\nDone.")


if __name__ == "__main__":
    main()
