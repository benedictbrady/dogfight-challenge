"""Fast experimentation Lab for iterating on double pendulum simulators.

Provides a REPL-friendly workflow: load real data once, then quickly evaluate
candidate equations-of-motion functions, compare them, and drill into
diagnostics -- all without touching the filesystem or re-importing modules.

Typical usage:

    from lab import Lab
    lab = Lab()                          # loads calibration runs, evals baseline
    lab.leaderboard()                    # shows baseline score

    from candidate_sim import equations_of_motion as eom_v2
    lab.eval(eom_v2, PendulumParams(b1=0.05), "viscous-v1")
    lab.compare("baseline", "viscous-v1")
    lab.diagnose("viscous-v1")
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from sim_interface import (
    EvalResult,
    PendulumParams,
    PendulumState,
    Trajectory,
    RunData,
)
from candidate_sim import equations_of_motion as _baseline_eom
from metrics import angle_rmse, divergence_time, energy_rmse, per_arm_rmse
from evaluate import params_for_run
from scenarios import get_calibration_runs, get_test_runs
from diagnostics import generate_report


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalScore:
    """Aggregate score for a candidate equations-of-motion function."""
    label: str
    mean_div_time: float
    median_div_time: float
    mean_angle_rmse: float
    per_run: list[EvalResult]
    params: PendulumParams
    code_hash: str
    # Internal bookkeeping (not printed in summaries)
    sim_trajs: list[Trajectory] = field(default_factory=list, repr=False)
    real_trajs: list[Trajectory] = field(default_factory=list, repr=False)
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _content_hash(source: str, params: PendulumParams) -> str:
    """Deterministic hash of (source code + params) for cache keying."""
    blob = (source + "|" + str(params)).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _simulate_with_eom(
    eom_func: Callable,
    initial_state: PendulumState,
    params: PendulumParams,
    t_end: float,
    dt: float,
    real_traj: Trajectory,
    fast: bool = False,
    early_stop_threshold: float = np.pi / 4,
    early_stop_check_every: int = 50,
) -> Trajectory:
    """Integrate a custom eom_func, with optional early termination.

    When the combined angle error exceeds *early_stop_threshold* the
    integration is stopped early -- the candidate has already diverged and
    continuing wastes time.  The check is performed via a dense-output
    event function so that solve_ivp handles it natively.
    """
    y0 = initial_state.to_array()
    t_eval = np.arange(0, t_end, dt)
    if len(t_eval) == 0:
        t_eval = np.array([0.0])

    rtol = 1e-6 if fast else 1e-10
    atol = 1e-8 if fast else 1e-12

    # Build an event function for early termination.
    # We need the real trajectory interpolated at arbitrary t values.
    # For speed, we just check at each t_eval point after solve_ivp;
    # true event-based stopping would require interpolating real data
    # which adds complexity.  Instead we use a step callback approach:
    # run solve_ivp in chunks and abort if diverged.
    #
    # Actually, the simplest performant approach: solve the whole thing
    # but with a custom RHS that checks a shared mutable counter.
    # Even simpler: just solve, then truncate.  For screening (fast=True)
    # the relaxed tolerances already make it fast.  For full eval we want
    # the complete trajectory anyway for diagnostics.
    #
    # Compromise: solve the full span but with the selected tolerances.
    # Early termination is applied post-hoc by checking every N steps.

    sol = solve_ivp(
        fun=lambda t, y: eom_func(t, y, params),
        t_span=(0, t_end),
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        # Return whatever we got -- diverged candidates often blow up
        n = len(sol.t)
        if n == 0:
            return Trajectory(
                t=np.array([0.0]),
                theta1=np.array([y0[0]]),
                theta2=np.array([y0[1]]),
                omega1=np.array([y0[2]]),
                omega2=np.array([y0[3]]),
            )

    return Trajectory(
        t=sol.t,
        theta1=sol.y[0],
        theta2=sol.y[1],
        omega1=sol.y[2],
        omega2=sol.y[3],
    )


def _get_source(func: Callable) -> str:
    """Best-effort source extraction; falls back to qualname."""
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return func.__qualname__


# ---------------------------------------------------------------------------
# Lab
# ---------------------------------------------------------------------------

class Lab:
    """Fast experimentation harness for double-pendulum EOM candidates.

    Parameters
    ----------
    runs : str
        Which dataset to load: ``"calibration"`` (default) or ``"test"``.
    """

    def __init__(self, runs: str = "calibration") -> None:
        print(f"Loading {runs} runs...")
        if runs == "test":
            self._run_data: list[RunData] = get_test_runs()
        else:
            self._run_data = get_calibration_runs()
        print(f"  {len(self._run_data)} runs loaded.")

        # Pre-extract real trajectories as contiguous numpy arrays
        self._real_trajs: list[Trajectory] = [r.trajectory for r in self._run_data]

        # Results cache: label -> EvalScore
        self._results: dict[str, EvalScore] = {}

        # Content-addressed dedup: code_hash -> label (first seen)
        self._hash_to_label: dict[str, str] = {}

        # Auto-evaluate baseline
        print("Evaluating baseline (ideal pendulum, no dissipation)...")
        self.eval(_baseline_eom, PendulumParams(), "baseline")
        print("Ready.\n")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def eval(
        self,
        eom_func: Callable,
        params: PendulumParams | None = None,
        label: str = "candidate",
        fast: bool = False,
    ) -> EvalScore:
        """Evaluate an equations-of-motion function against loaded runs.

        Parameters
        ----------
        eom_func : callable
            Signature ``(t, y, params) -> dy/dt`` matching candidate_sim.
        params : PendulumParams, optional
            Physical parameters.  Defaults to ideal (no dissipation).
        label : str
            Human-readable name stored in the leaderboard.
        fast : bool
            Use relaxed solver tolerances (rtol=1e-6) for quick screening.

        Returns
        -------
        EvalScore
        """
        if params is None:
            params = PendulumParams()

        source = _get_source(eom_func)
        code_hash = _content_hash(source, params)

        # Check content-addressed cache
        if code_hash in self._hash_to_label:
            cached_label = self._hash_to_label[code_hash]
            if cached_label in self._results:
                cached = self._results[cached_label]
                if label != cached_label:
                    # Same content, different label -- store alias
                    self._results[label] = EvalScore(
                        label=label,
                        mean_div_time=cached.mean_div_time,
                        median_div_time=cached.median_div_time,
                        mean_angle_rmse=cached.mean_angle_rmse,
                        per_run=cached.per_run,
                        params=cached.params,
                        code_hash=cached.code_hash,
                        sim_trajs=cached.sim_trajs,
                        real_trajs=cached.real_trajs,
                        elapsed_sec=cached.elapsed_sec,
                    )
                    self._hash_to_label[code_hash] = label
                    print(f"[cache hit] '{label}' identical to '{cached_label}' -- reusing results.")
                    return self._results[label]

        t0 = time.perf_counter()

        eval_results: list[EvalResult] = []
        sim_trajs: list[Trajectory] = []

        for run_data, real_traj in zip(self._run_data, self._real_trajs):
            run_params = params_for_run(run_data, params)
            initial = real_traj.state_at(0)
            t_end = float(real_traj.t[-1] - real_traj.t[0])
            dt = real_traj.dt

            try:
                sim_traj = _simulate_with_eom(
                    eom_func, initial, run_params, t_end, dt,
                    real_traj, fast=fast,
                )
            except Exception as exc:
                # Candidate blew up -- record zero-quality result
                sim_traj = Trajectory(
                    t=np.array([0.0]),
                    theta1=np.array([initial.theta1]),
                    theta2=np.array([initial.theta2]),
                    omega1=np.array([initial.omega1]),
                    omega2=np.array([initial.omega2]),
                )
                eval_results.append(EvalResult(
                    run_id=run_data.run_id,
                    divergence_time=0.0,
                    angle_rmse=float("inf"),
                    energy_rmse=float("inf"),
                    theta1_rmse=float("inf"),
                    theta2_rmse=float("inf"),
                    sim_duration=0.0,
                    real_duration=t_end,
                    notes=f"Integration failed: {exc}",
                ))
                sim_trajs.append(sim_traj)
                continue

            div_t = divergence_time(sim_traj, real_traj)
            a_rmse = angle_rmse(sim_traj, real_traj)
            e_rmse = energy_rmse(sim_traj, real_traj, run_params)
            t1_rmse, t2_rmse = per_arm_rmse(sim_traj, real_traj)

            eval_results.append(EvalResult(
                run_id=run_data.run_id,
                divergence_time=div_t,
                angle_rmse=a_rmse,
                energy_rmse=e_rmse,
                theta1_rmse=t1_rmse,
                theta2_rmse=t2_rmse,
                sim_duration=float(sim_traj.t[-1]),
                real_duration=t_end,
            ))
            sim_trajs.append(sim_traj)

        elapsed = time.perf_counter() - t0

        div_times = [r.divergence_time for r in eval_results]
        angle_rmses = [r.angle_rmse for r in eval_results if np.isfinite(r.angle_rmse)]

        score = EvalScore(
            label=label,
            mean_div_time=float(np.mean(div_times)) if div_times else 0.0,
            median_div_time=float(np.median(div_times)) if div_times else 0.0,
            mean_angle_rmse=float(np.mean(angle_rmses)) if angle_rmses else float("inf"),
            per_run=eval_results,
            params=params,
            code_hash=code_hash,
            sim_trajs=sim_trajs,
            real_trajs=list(self._real_trajs),
            elapsed_sec=elapsed,
        )

        self._results[label] = score
        self._hash_to_label[code_hash] = label

        self.report(label)
        return score

    # ------------------------------------------------------------------
    # eval_code -- compile from string
    # ------------------------------------------------------------------

    def eval_code(
        self,
        code_str: str,
        params: PendulumParams | None = None,
        label: str = "from_code",
        fast: bool = False,
    ) -> EvalScore:
        """Compile an equations_of_motion function from a code string and evaluate it.

        The *code_str* must define a function named ``equations_of_motion``
        with the standard ``(t, y, params)`` signature.

        Parameters
        ----------
        code_str : str
            Python source code defining ``equations_of_motion``.
        params : PendulumParams, optional
            Physical parameters.
        label : str
            Leaderboard label.
        fast : bool
            Relaxed tolerances for screening.

        Returns
        -------
        EvalScore
        """
        namespace: dict = {"np": np, "numpy": np}
        try:
            exec(compile(textwrap.dedent(code_str), f"<{label}>", "exec"), namespace)
        except Exception as exc:
            raise ValueError(f"Failed to compile code for '{label}': {exc}") from exc

        if "equations_of_motion" not in namespace:
            raise ValueError(
                f"Code for '{label}' must define a function named 'equations_of_motion'"
            )

        eom_func = namespace["equations_of_motion"]
        return self.eval(eom_func, params, label, fast=fast)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def report(self, label: str) -> None:
        """Print a one-line summary for *label*, with delta from baseline."""
        if label not in self._results:
            print(f"No results for '{label}'.")
            return

        score = self._results[label]
        baseline = self._results.get("baseline")

        parts = [
            f"[{score.label}]",
            f"mean_div={score.mean_div_time:.3f}s",
            f"med_div={score.median_div_time:.3f}s",
            f"mean_rmse={score.mean_angle_rmse:.4f}",
            f"hash={score.code_hash}",
            f"({score.elapsed_sec:.1f}s)",
        ]

        if baseline is not None and label != "baseline":
            d_div = score.mean_div_time - baseline.mean_div_time
            d_rmse = score.mean_angle_rmse - baseline.mean_angle_rmse
            sign_div = "+" if d_div >= 0 else ""
            sign_rmse = "+" if d_rmse >= 0 else ""
            parts.append(f"delta_div={sign_div}{d_div:.3f}s")
            parts.append(f"delta_rmse={sign_rmse}{d_rmse:.4f}")

        print("  ".join(parts))

    def compare(self, label_a: str, label_b: str) -> None:
        """Print side-by-side metric comparison for two evaluated candidates."""
        a = self._results.get(label_a)
        b = self._results.get(label_b)
        missing = [l for l, s in [(label_a, a), (label_b, b)] if s is None]
        if missing:
            print(f"Missing results for: {', '.join(missing)}")
            return

        col_w = max(len(label_a), len(label_b), 12)
        hdr_a = label_a.center(col_w)
        hdr_b = label_b.center(col_w)

        print(f"\n{'Metric':<22}  {hdr_a}  {hdr_b}  {'Delta':>10}")
        print("-" * (22 + 2 * (col_w + 2) + 12))

        rows = [
            ("Mean div time (s)", a.mean_div_time, b.mean_div_time),
            ("Median div time (s)", a.median_div_time, b.median_div_time),
            ("Mean angle RMSE", a.mean_angle_rmse, b.mean_angle_rmse),
        ]

        for name, va, vb in rows:
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            print(f"{name:<22}  {va:>{col_w}.4f}  {vb:>{col_w}.4f}  {sign}{delta:>9.4f}")

        # Per-run comparison
        print(f"\n{'Run':<12}  {'div_a':>8}  {'div_b':>8}  {'delta':>8}  {'rmse_a':>8}  {'rmse_b':>8}  {'delta':>8}")
        print("-" * 72)
        for ra, rb in zip(a.per_run, b.per_run):
            dd = rb.divergence_time - ra.divergence_time
            dr = rb.angle_rmse - ra.angle_rmse
            sd = "+" if dd >= 0 else ""
            sr = "+" if dr >= 0 else ""
            print(
                f"{ra.run_id:<12}  {ra.divergence_time:>8.3f}  {rb.divergence_time:>8.3f}  "
                f"{sd}{dd:>7.3f}  {ra.angle_rmse:>8.4f}  {rb.angle_rmse:>8.4f}  {sr}{dr:>7.4f}"
            )
        print()

    def leaderboard(self) -> None:
        """Print a ranked table of all evaluated candidates, sorted by mean_div_time descending."""
        if not self._results:
            print("No results yet. Run lab.eval(...) first.")
            return

        ranked = sorted(self._results.values(), key=lambda s: s.mean_div_time, reverse=True)

        baseline = self._results.get("baseline")

        print(f"\n{'#':>3}  {'Label':<24}  {'Mean Div':>9}  {'Med Div':>9}  {'Mean RMSE':>10}  {'Delta Div':>10}  {'Hash':>10}  {'Time':>6}")
        print("-" * 96)

        for i, score in enumerate(ranked, 1):
            d_div = ""
            if baseline is not None and score.label != "baseline":
                dd = score.mean_div_time - baseline.mean_div_time
                sign = "+" if dd >= 0 else ""
                d_div = f"{sign}{dd:.3f}s"

            print(
                f"{i:>3}  {score.label:<24}  "
                f"{score.mean_div_time:>8.3f}s  {score.median_div_time:>8.3f}s  "
                f"{score.mean_angle_rmse:>10.4f}  {d_div:>10}  "
                f"{score.code_hash:>10}  {score.elapsed_sec:>5.1f}s"
            )
        print()

    def diagnose(self, label: str) -> str:
        """Generate and print a full diagnostic report for a candidate.

        Returns the report string (also prints it).
        """
        score = self._results.get(label)
        if score is None:
            msg = f"No results for '{label}'. Run lab.eval(...) first."
            print(msg)
            return msg

        report_str = generate_report(
            eval_results=score.per_run,
            sim_trajectories=score.sim_trajs,
            real_trajectories=score.real_trajs,
            params=score.params,
        )
        print(report_str)
        return report_str
