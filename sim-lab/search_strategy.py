"""Explore/exploit decision framework for physics simulator optimization.

Maintains a population of top candidate simulators, tracks which physics
ideas have been tried, and suggests whether to explore new approaches or
exploit (refine) the current best. Includes parameter sweep utilities.
"""

import random
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable

import numpy as np


@dataclass
class Candidate:
    """A candidate simulator with its evaluation score."""

    label: str
    code: str  # The equations_of_motion source code
    params: dict  # PendulumParams as dict
    mean_div_time: float
    approach: str  # e.g. "viscous", "drag", "coulomb", "combined"


class SearchStrategy:
    """Explore/exploit decision framework for iterative simulator improvement.

    Tracks a population of top-N candidates, monitors stagnation, and
    provides suggestions for what to try next. Also offers grid and random
    parameter sweep utilities.
    """

    def __init__(self, population_size: int = 10):
        self.population_size = population_size
        self.population: list[Candidate] = []  # Top-N candidates, sorted desc
        self.explored_regions: set[str] = set()  # Physics ideas tried
        self.consecutive_no_improvement: int = 0
        self.history: list[dict] = []  # All attempts

    def add_result(self, candidate: Candidate):
        """Record a result. Update population and stagnation counter."""
        # Record in history
        self.history.append({
            "label": candidate.label,
            "mean_div_time": candidate.mean_div_time,
            "approach": candidate.approach,
            "params": candidate.params,
        })

        # Check if this improves the best score
        best_before = self.population[0].mean_div_time if self.population else 0.0
        improved = candidate.mean_div_time > best_before

        # Insert into population
        self.population.append(candidate)
        self.population.sort(key=lambda c: c.mean_div_time, reverse=True)
        self.population = self.population[: self.population_size]

        # Update stagnation counter
        if improved:
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1

    def suggest_mode(self) -> str:
        """Return 'exploit' (70%) or 'explore' (30%).

        Forces 'explore' after 5 consecutive attempts with no improvement.
        """
        if self.consecutive_no_improvement >= 5:
            return "explore"
        return "exploit" if random.random() < 0.7 else "explore"

    def suggest_next(self) -> dict:
        """Return a suggestion dict with mode, best_candidate, and hint text."""
        mode = self.suggest_mode()
        best = self.population[0] if self.population else None

        if mode == "exploit" and best is not None:
            hint = self._exploit_hint(best)
        else:
            hint = self._explore_hint()

        return {
            "mode": mode,
            "best_candidate": best,
            "hint": hint,
        }

    def _exploit_hint(self, best: Candidate) -> str:
        """Generate a hint for refining the best candidate."""
        hints = [
            f"Try adjusting b1 coefficient around {best.params.get('b1', 0):.4f}",
            f"Try adjusting b2 coefficient around {best.params.get('b2', 0):.4f}",
            "Fine-tune the friction coefficients with a narrow parameter sweep",
            "Try small perturbations (+/-20%) on the best params",
            "Consider asymmetric damping: different coefficients for each joint",
            "Run a focused sweep around the best parameter region",
        ]
        return random.choice(hints)

    def _explore_hint(self) -> str:
        """Generate a hint for exploring a new physics idea."""
        all_regions = {
            "viscous_friction": "Add viscous friction (torque proportional to angular velocity)",
            "air_drag": "Add air drag (torque proportional to velocity squared)",
            "coulomb_friction": "Add Coulomb friction (constant torque opposing motion)",
            "arm_calibration": "Calibrate arm length ratios from pixel measurements",
            "combined_optimization": "Combine multiple dissipation mechanisms and optimize jointly",
            "bearing_friction": "Model bearing friction with Stribeck curve",
            "aerodynamic_drag": "Model distributed aerodynamic drag along arm length",
            "joint_stiffness": "Add torsional spring stiffness at joints",
            "mass_distribution": "Account for distributed mass (moment of inertia corrections)",
            "nonlinear_damping": "Try velocity-dependent nonlinear damping models",
        }

        # Prefer unexplored regions
        unexplored = {
            k: v for k, v in all_regions.items() if k not in self.explored_regions
        }

        if unexplored:
            region, description = random.choice(list(unexplored.items()))
            return f"[Unexplored] {region}: {description}"
        else:
            region, description = random.choice(list(all_regions.items()))
            return f"[Revisit] {region}: {description}"

    def mark_explored(self, region: str):
        """Record that a physics idea has been tried."""
        self.explored_regions.add(region)

    def parameter_sweep(
        self,
        eom_func: Callable,
        base_params: dict,
        sweep_config: dict,
        lab: Any,
    ) -> list[tuple[dict, float]]:
        """Grid search over continuous params.

        Args:
            eom_func: Equations of motion function to evaluate.
            base_params: Base parameter dict to modify.
            sweep_config: Dict of param_name -> (min, max, n_points).
                Example: {"b1": (0.01, 1.0, 10), "b2": (0.01, 0.5, 10)}
            lab: Object with lab.eval(eom_func, params, label) -> EvalScore.

        Returns:
            List of (params_dict, mean_div_time) sorted by score descending.
        """
        # Build grid axes
        param_names = list(sweep_config.keys())
        axes = []
        for name in param_names:
            lo, hi, n = sweep_config[name]
            axes.append(np.linspace(lo, hi, n))

        combos = list(product(*axes))
        total = len(combos)
        print(f"Parameter sweep: {total} combinations across {param_names}")

        results: list[tuple[dict, float]] = []
        best_score = 0.0
        t0 = time.time()

        for i, values in enumerate(combos):
            # Build params for this combo
            params = dict(base_params)
            for name, val in zip(param_names, values):
                params[name] = float(val)

            label = "sweep_" + "_".join(
                f"{n}={v:.4f}" for n, v in zip(param_names, values)
            )

            try:
                score = lab.eval(eom_func, params, label)
                mdt = score.mean_div_time
            except Exception as e:
                print(f"  [{i + 1}/{total}] FAILED: {e}")
                mdt = 0.0

            results.append((params, mdt))

            if mdt > best_score:
                best_score = mdt

            if (i + 1) % max(1, total // 10) == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                print(
                    f"  [{i + 1}/{total}] best={best_score:.3f}s "
                    f"elapsed={elapsed:.1f}s"
                )

        results.sort(key=lambda x: x[1], reverse=True)

        elapsed = time.time() - t0
        print(
            f"Sweep complete: {total} combos in {elapsed:.1f}s. "
            f"Best={results[0][1]:.3f}s"
        )

        return results

    def random_sweep(
        self,
        eom_func: Callable,
        base_params: dict,
        sweep_config: dict,
        lab: Any,
        n_samples: int = 50,
    ) -> list[tuple[dict, float]]:
        """Random search over param space. More efficient than grid for high dimensions.

        Args:
            eom_func: Equations of motion function to evaluate.
            base_params: Base parameter dict to modify.
            sweep_config: Dict of param_name -> (min, max, n_points).
                The n_points field is ignored; n_samples controls count.
            lab: Object with lab.eval(eom_func, params, label) -> EvalScore.
            n_samples: Number of random parameter combinations to try.

        Returns:
            List of (params_dict, mean_div_time) sorted by score descending.
        """
        param_names = list(sweep_config.keys())
        bounds = [(sweep_config[name][0], sweep_config[name][1]) for name in param_names]

        print(f"Random sweep: {n_samples} samples across {param_names}")

        results: list[tuple[dict, float]] = []
        best_score = 0.0
        t0 = time.time()

        for i in range(n_samples):
            # Sample uniformly within bounds
            values = [
                np.random.uniform(lo, hi) for lo, hi in bounds
            ]

            params = dict(base_params)
            for name, val in zip(param_names, values):
                params[name] = float(val)

            label = "rsweep_" + "_".join(
                f"{n}={v:.4f}" for n, v in zip(param_names, values)
            )

            try:
                score = lab.eval(eom_func, params, label)
                mdt = score.mean_div_time
            except Exception as e:
                print(f"  [{i + 1}/{n_samples}] FAILED: {e}")
                mdt = 0.0

            results.append((params, mdt))

            if mdt > best_score:
                best_score = mdt

            if (i + 1) % max(1, n_samples // 10) == 0 or (i + 1) == n_samples:
                elapsed = time.time() - t0
                print(
                    f"  [{i + 1}/{n_samples}] best={best_score:.3f}s "
                    f"elapsed={elapsed:.1f}s"
                )

        results.sort(key=lambda x: x[1], reverse=True)

        elapsed = time.time() - t0
        print(
            f"Random sweep complete: {n_samples} samples in {elapsed:.1f}s. "
            f"Best={results[0][1]:.3f}s"
        )

        return results

    def leaderboard(self) -> str:
        """Formatted table of population."""
        if not self.population:
            return "No candidates evaluated yet."

        lines = []
        lines.append(f"{'Rank':<6} {'Label':<30} {'Approach':<20} {'Mean Div Time':>14}")
        lines.append(f"{'-'*6} {'-'*30} {'-'*20} {'-'*14}")

        for i, c in enumerate(self.population):
            lines.append(
                f"{i + 1:<6} {c.label:<30} {c.approach:<20} {c.mean_div_time:>14.3f}s"
            )

        return "\n".join(lines)

    def status(self) -> str:
        """Summary of search state: best score, stagnation, explored regions."""
        lines = []
        lines.append("=== Search Strategy Status ===")

        if self.population:
            best = self.population[0]
            lines.append(f"Best score:        {best.mean_div_time:.3f}s ({best.label})")
        else:
            lines.append("Best score:        N/A (no candidates yet)")

        lines.append(f"Population size:   {len(self.population)}/{self.population_size}")
        lines.append(f"Total attempts:    {len(self.history)}")
        lines.append(f"Stagnation count:  {self.consecutive_no_improvement}")

        if self.explored_regions:
            lines.append(f"Explored regions:  {', '.join(sorted(self.explored_regions))}")
        else:
            lines.append("Explored regions:  (none)")

        if self.consecutive_no_improvement >= 5:
            lines.append("** Stagnation detected -- next suggestion will force EXPLORE mode **")

        return "\n".join(lines)
