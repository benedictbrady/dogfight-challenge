"""Modal web endpoints for the training dashboard.

Serves training metrics, pool data, and run metadata from the Modal volume.
Parses TensorBoard events on-demand and caches the result as metrics.json.

Deploy:
    modal deploy modal_dashboard.py

Endpoints (after deploy):
    GET  /runs           — list all training runs (with status)
    GET  /metrics?run=X  — parsed TensorBoard metrics for a run
    GET  /pool?run=X     — pool.json for a self-play run
    GET  /config?run=X   — config.json for a run
    GET  /reparse?run=X  — force re-parse TB events
    GET  /hide?run=X     — hide a run from the dashboard
    GET  /unhide?run=X   — unhide a run
    GET  /delete?run=X   — permanently delete a run from the volume
"""

import modal
import os
import json
import shutil
import time
from datetime import datetime, timezone
from fastapi.responses import JSONResponse

app = modal.App("dogfight-dashboard")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tensorboard")
)

vol = modal.Volume.from_name("dogfight-training", create_if_missing=True)

RESULTS_DIR = "/results"
HIDDEN_FILE = "/results/.dashboard_hidden.json"

# If TB events were modified within this many seconds, run is "live"
LIVE_THRESHOLD_SECONDS = 600  # 10 minutes


def _load_hidden() -> set[str]:
    """Load the set of hidden run names."""
    try:
        with open(HIDDEN_FILE) as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def _save_hidden(hidden: set[str]):
    """Save the set of hidden run names."""
    with open(HIDDEN_FILE, "w") as f:
        json.dump(sorted(hidden), f)


def _get_run_status(path: str) -> str:
    """Determine run status: 'live', 'completed', or 'failed'.

    - live: TB events modified within LIVE_THRESHOLD_SECONDS
    - completed: has final.pt checkpoint
    - failed: everything else (no final.pt, not recently active)
    """
    now = time.time()

    # Check if TB events were recently written
    tb_dir = os.path.join(path, "runs")
    if os.path.isdir(tb_dir):
        latest_mtime = 0.0
        for root, _, files in os.walk(tb_dir):
            for f in files:
                if "events.out.tfevents" in f:
                    mtime = os.path.getmtime(os.path.join(root, f))
                    latest_mtime = max(latest_mtime, mtime)
        if latest_mtime > 0 and (now - latest_mtime) < LIVE_THRESHOLD_SECONDS:
            return "live"

    # Check for final checkpoint
    ckpt_dir = os.path.join(path, "checkpoints")
    if os.path.isdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, "final.pt")):
            return "completed"

    # Has eval.txt = completed even without final.pt
    if os.path.isfile(os.path.join(path, "eval.txt")):
        return "completed"

    return "failed"


def _discover_runs() -> list[dict]:
    """Scan the volume for all training run directories."""
    runs = []

    def _check_run(path: str, name: str, category: str):
        has_tb = os.path.isdir(os.path.join(path, "runs"))
        has_config = os.path.isfile(os.path.join(path, "config.json"))
        has_metrics = os.path.isfile(os.path.join(path, "metrics.json"))
        has_pool = (
            os.path.isfile(os.path.join(path, "pool", "pool.json"))
            or os.path.isfile(os.path.join(path, "pool.json"))
        )

        ckpt_dir = os.path.join(path, "checkpoints")
        checkpoints = []
        if os.path.isdir(ckpt_dir):
            checkpoints = sorted([
                f for f in os.listdir(ckpt_dir)
                if f.endswith(".pt")
            ])

        if has_tb or has_config:
            status = _get_run_status(path)
            runs.append({
                "name": name,
                "category": category,
                "path": path,
                "status": status,
                "has_metrics": has_metrics,
                "has_config": has_config,
                "has_pool": has_pool,
                "has_tb": has_tb,
                "checkpoints": checkpoints,
            })

    if not os.path.isdir(RESULTS_DIR):
        return runs

    for entry in sorted(os.listdir(RESULTS_DIR)):
        full = os.path.join(RESULTS_DIR, entry)
        if not os.path.isdir(full):
            continue

        if entry == "selfplay":
            for sub in sorted(os.listdir(full)):
                sub_path = os.path.join(full, sub)
                if os.path.isdir(sub_path):
                    _check_run(sub_path, f"selfplay/{sub}", "selfplay")
        elif entry == "unified":
            for sub in sorted(os.listdir(full)):
                sub_path = os.path.join(full, sub)
                if os.path.isdir(sub_path):
                    _check_run(sub_path, f"unified/{sub}", "unified")
        else:
            _check_run(full, entry, "curriculum")

    return runs


def _find_run_path(run_name: str) -> str | None:
    path = os.path.join(RESULTS_DIR, run_name)
    if os.path.isdir(path):
        return path
    return None


def _parse_tb(run_path: str) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dir = os.path.join(run_path, "runs")
    if not os.path.isdir(tb_dir):
        return {}

    # Find the deepest directory containing event files
    event_dirs = []
    for root, _, files in os.walk(tb_dir):
        if any("events.out.tfevents" in f for f in files):
            event_dirs.append(root)
    if not event_dirs:
        return {}

    categories: dict[str, dict[str, list[dict]]] = {}

    for event_dir in event_dirs:
        ea = EventAccumulator(event_dir)
        ea.Reload()

        for tag in sorted(ea.Tags().get("scalars", [])):
            parts = tag.split("/", 1)
            if len(parts) == 2:
                category, metric = parts
            else:
                category, metric = "misc", parts[0]
            category = category.strip()
            metric = metric.strip()

            if category not in categories:
                categories[category] = {}

            events = ea.Scalars(tag)
            points = [
                {"step": e.step, "wall_time": e.wall_time, "value": e.value}
                for e in events
            ]
            points.sort(key=lambda p: p["step"])
            categories[category][metric] = points

    run_name = os.path.basename(run_path)
    metrics = {
        "run_name": run_name,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "losses": categories.pop("losses", {}),
        "charts": categories.pop("charts", {}),
        "selfplay": categories.pop("selfplay", {}),
        "eval": categories.pop("eval", {}),
    }
    for cat_name, cat_data in categories.items():
        metrics[cat_name] = cat_data

    return metrics


# ---------------------------------------------------------------------------
# Web endpoints
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={"/results": vol}, min_containers=1)
@modal.fastapi_endpoint(method="GET")
def runs():
    """List all training runs on the volume, excluding hidden ones."""
    vol.reload()
    hidden = _load_hidden()
    discovered = _discover_runs()
    result = []
    for r in discovered:
        if r["name"] in hidden:
            continue
        result.append({
            "name": r["name"],
            "category": r["category"],
            "status": r["status"],
            "has_metrics": r["has_metrics"],
            "has_pool": r["has_pool"],
            "has_tb": r["has_tb"],
            "checkpoints": r["checkpoints"],
        })
    return result


@app.function(image=image, volumes={"/results": vol}, min_containers=1, timeout=300)
@modal.fastapi_endpoint(method="GET")
def metrics(run: str):
    """Return parsed TensorBoard metrics for a run."""
    import traceback

    try:
        vol.reload()
        run_path = _find_run_path(run)
        if not run_path:
            return JSONResponse({"error": f"Run not found: {run}"}, status_code=404)

        metrics_path = os.path.join(run_path, "metrics.json")
        is_live = _get_run_status(run_path) == "live"

        # For live runs, always re-parse to get fresh data.
        # For completed/failed runs, use cached metrics.json if available.
        if not is_live and os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                return json.load(f)

        parsed = _parse_tb(run_path)
        if not parsed:
            run_name = os.path.basename(run_path)
            return {
                "run_name": run_name,
                "parsed_at": datetime.now(timezone.utc).isoformat(),
                "losses": {}, "charts": {}, "selfplay": {}, "eval": {},
            }

        # Cache to volume (for live runs this gets overwritten on next request)
        with open(metrics_path, "w") as f:
            json.dump(parsed, f)
        vol.commit()

        return parsed
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR in metrics endpoint: {tb}")
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


@app.function(image=image, volumes={"/results": vol}, min_containers=1)
@modal.fastapi_endpoint(method="GET")
def pool(run: str):
    """Return pool.json for a self-play run."""
    vol.reload()
    run_path = _find_run_path(run)
    if not run_path:
        return JSONResponse({"error": f"Run not found: {run}"}, status_code=404)

    pool_path = os.path.join(run_path, "pool", "pool.json")
    if not os.path.isfile(pool_path):
        pool_path = os.path.join(run_path, "pool.json")
    if not os.path.isfile(pool_path):
        return JSONResponse({"error": f"No pool data for: {run}"}, status_code=404)

    with open(pool_path) as f:
        return json.load(f)


@app.function(image=image, volumes={"/results": vol}, min_containers=1)
@modal.fastapi_endpoint(method="GET")
def config(run: str):
    """Return config.json for a run."""
    vol.reload()
    run_path = _find_run_path(run)
    if not run_path:
        return JSONResponse({"error": f"Run not found: {run}"}, status_code=404)

    config_path = os.path.join(run_path, "config.json")
    if not os.path.isfile(config_path):
        return JSONResponse({"error": f"No config for: {run}"}, status_code=404)

    with open(config_path) as f:
        return json.load(f)


@app.function(image=image, volumes={"/results": vol})
@modal.fastapi_endpoint(method="GET")
def reparse(run: str):
    """Force re-parse TensorBoard events."""
    vol.reload()
    run_path = _find_run_path(run)
    if not run_path:
        return JSONResponse({"error": f"Run not found: {run}"}, status_code=404)

    parsed = _parse_tb(run_path)
    if not parsed:
        return JSONResponse({"error": f"No TensorBoard data found for: {run}"}, status_code=404)

    metrics_path = os.path.join(run_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(parsed, f)
    vol.commit()

    total_metrics = sum(
        1 for v in parsed.values() if isinstance(v, dict)
        for _ in v.values()
    )
    return {"status": "ok", "run": run, "metrics_count": total_metrics}


@app.function(image=image, volumes={"/results": vol})
@modal.fastapi_endpoint(method="GET")
def hide(run: str):
    """Hide a run from the dashboard (doesn't delete data)."""
    vol.reload()
    hidden = _load_hidden()
    hidden.add(run)
    _save_hidden(hidden)
    vol.commit()
    return {"status": "ok", "hidden": run}


@app.function(image=image, volumes={"/results": vol})
@modal.fastapi_endpoint(method="GET")
def unhide(run: str):
    """Unhide a previously hidden run."""
    vol.reload()
    hidden = _load_hidden()
    hidden.discard(run)
    _save_hidden(hidden)
    vol.commit()
    return {"status": "ok", "unhidden": run}


@app.function(image=image, volumes={"/results": vol}, timeout=120)
@modal.fastapi_endpoint(method="GET")
def delete(run: str):
    """Permanently delete a run from the volume."""
    vol.reload()
    run_path = _find_run_path(run)
    if not run_path:
        return JSONResponse({"error": f"Run not found: {run}"}, status_code=404)

    shutil.rmtree(run_path)

    # Also remove from hidden list if present
    hidden = _load_hidden()
    hidden.discard(run)
    _save_hidden(hidden)

    vol.commit()
    return {"status": "ok", "deleted": run}
