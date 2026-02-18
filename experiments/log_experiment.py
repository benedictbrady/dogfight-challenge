"""Log a new experiment to the registry.

Usage:
    python experiments/log_experiment.py --tag "sweep-ent" --config '{"n_envs": 128, "ent_coef": 0.01}' --notes "Testing higher entropy"
    python experiments/log_experiment.py --id 3 --status completed --results '{"win_rate": 0.45}'
"""

import argparse
import json
from datetime import date
from pathlib import Path

REGISTRY = Path(__file__).parent / "registry.json"


def load():
    with open(REGISTRY) as f:
        return json.load(f)


def save(data):
    with open(REGISTRY, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {REGISTRY}")


def new_experiment(tag: str, config: dict, notes: str = "") -> int:
    reg = load()
    exp_id = reg["next_id"]
    reg["next_id"] = exp_id + 1
    reg["experiments"].append({
        "id": exp_id,
        "date": str(date.today()),
        "tag": tag,
        "config": config,
        "status": "running",
        "notes": notes,
        "results": {},
    })
    save(reg)
    print(f"Created experiment #{exp_id}: {tag}")
    return exp_id


def update_experiment(exp_id: int, status: str = None, results: dict = None, notes: str = None):
    reg = load()
    for exp in reg["experiments"]:
        if exp["id"] == exp_id:
            if status:
                exp["status"] = status
            if results:
                exp["results"].update(results)
            if notes:
                exp["notes"] = notes
            save(reg)
            print(f"Updated experiment #{exp_id}")
            return
    print(f"Experiment #{exp_id} not found")


def list_experiments():
    reg = load()
    print(f"{'#':>3} {'Status':<10} {'Tag':<20} {'Date':<12} Notes")
    print("-" * 70)
    for exp in reg["experiments"]:
        print(f"{exp['id']:>3} {exp['status']:<10} {exp['tag']:<20} {exp['date']:<12} {exp.get('notes', '')[:40]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    new = sub.add_parser("new")
    new.add_argument("--tag", required=True)
    new.add_argument("--config", default="{}")
    new.add_argument("--notes", default="")

    upd = sub.add_parser("update")
    upd.add_argument("--id", type=int, required=True)
    upd.add_argument("--status")
    upd.add_argument("--results", default=None)
    upd.add_argument("--notes", default=None)

    sub.add_parser("list")

    args = parser.parse_args()
    if args.cmd == "new":
        new_experiment(args.tag, json.loads(args.config), args.notes)
    elif args.cmd == "update":
        results = json.loads(args.results) if args.results else None
        update_experiment(args.id, args.status, results, args.notes)
    elif args.cmd == "list":
        list_experiments()
    else:
        list_experiments()
