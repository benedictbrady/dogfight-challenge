"""Diagnostic analysis of sim vs real discrepancies.

Produces a markdown report identifying missing physics (energy decay,
velocity-dependent drag, Coulomb friction) for the AI improvement loop.
"""

import numpy as np

from sim_interface import EvalResult, PendulumParams, Trajectory


def _energy_decay_analysis(
    sim_trajs: list[Trajectory],
    real_trajs: list[Trajectory],
    results: list[EvalResult],
    params: PendulumParams,
) -> str:
    """Analyze energy conservation (sim) vs energy decay (real)."""
    lines = ["## Energy Analysis\n"]

    for i, (sim, real, res) in enumerate(zip(sim_trajs, real_trajs, results)):
        e_sim = sim.energy(params)
        e_real = real.energy(params)
        n = min(len(e_sim), len(e_real))
        e_sim, e_real = e_sim[:n], e_real[:n]

        # Sim should conserve energy (ideal); real should lose energy
        sim_e0, sim_ef = e_sim[0], e_sim[-1]
        real_e0, real_ef = e_real[0], e_real[-1]
        sim_drift = (sim_ef - sim_e0) / (abs(sim_e0) + 1e-12)
        real_decay = (real_ef - real_e0) / (abs(real_e0) + 1e-12)

        # Fit linear decay rate to real energy
        t = np.arange(n) * sim.dt
        if len(t) > 1:
            coeffs = np.polyfit(t, e_real, 1)
            decay_rate = coeffs[0]  # energy units per second
        else:
            decay_rate = 0.0

        lines.append(f"### {res.run_id}")
        lines.append(f"- Sim energy drift: {sim_drift:+.4%} (should be ~0%)")
        lines.append(f"- Real energy change: {real_decay:+.4%}")
        lines.append(f"- Real energy linear decay rate: {decay_rate:.4f} units/s")
        if abs(real_decay) > 0.05:
            lines.append(
                f"- **Signal**: Real data loses {abs(real_decay)*100:.1f}% energy "
                f"-> dissipation is present but not modeled"
            )
        lines.append("")

    return "\n".join(lines)


def _velocity_drag_analysis(
    sim_trajs: list[Trajectory],
    real_trajs: list[Trajectory],
    results: list[EvalResult],
) -> str:
    """Check for velocity-dependent error patterns (air drag signature)."""
    lines = ["## Velocity-Dependent Error Analysis (Air Drag Signal)\n"]

    for sim, real, res in zip(sim_trajs, real_trajs, results):
        n = min(len(sim), len(real))
        err1 = sim.omega1[:n] - real.omega1[:n]
        err2 = sim.omega2[:n] - real.omega2[:n]
        speed1 = np.abs(real.omega1[:n])
        speed2 = np.abs(real.omega2[:n])

        # Correlation between speed and velocity error
        # Positive correlation = sim overshoots at high speed = missing drag
        if np.std(speed1) > 1e-8 and np.std(err1) > 1e-8:
            corr1 = float(np.corrcoef(speed1, np.abs(err1))[0, 1])
        else:
            corr1 = 0.0
        if np.std(speed2) > 1e-8 and np.std(err2) > 1e-8:
            corr2 = float(np.corrcoef(speed2, np.abs(err2))[0, 1])
        else:
            corr2 = 0.0

        lines.append(f"### {res.run_id}")
        lines.append(f"- Arm 1 speed-error correlation: {corr1:+.3f}")
        lines.append(f"- Arm 2 speed-error correlation: {corr2:+.3f}")
        if corr1 > 0.3 or corr2 > 0.3:
            lines.append(
                "- **Signal**: Error grows with velocity -> likely missing air drag"
            )
        lines.append("")

    return "\n".join(lines)


def _friction_analysis(
    sim_trajs: list[Trajectory],
    real_trajs: list[Trajectory],
    results: list[EvalResult],
) -> str:
    """Check for direction-change anomalies (Coulomb friction signature)."""
    lines = ["## Direction-Change Analysis (Coulomb Friction Signal)\n"]

    for sim, real, res in zip(sim_trajs, real_trajs, results):
        n = min(len(sim), len(real))
        # Find zero-crossings in real omega (direction reversals)
        zc1 = np.where(np.diff(np.sign(real.omega1[:n])))[0]
        zc2 = np.where(np.diff(np.sign(real.omega2[:n])))[0]

        # Measure error spikes around direction changes
        window = 5  # frames
        err1 = np.abs(sim.theta1[:n] - real.theta1[:n])
        err2 = np.abs(sim.theta2[:n] - real.theta2[:n])

        if len(zc1) > 0:
            zc_err1 = np.mean([
                np.mean(err1[max(0, z - window):min(n, z + window)])
                for z in zc1
            ])
            bg_err1 = np.mean(err1)
            ratio1 = zc_err1 / (bg_err1 + 1e-12)
        else:
            ratio1 = 1.0

        if len(zc2) > 0:
            zc_err2 = np.mean([
                np.mean(err2[max(0, z - window):min(n, z + window)])
                for z in zc2
            ])
            bg_err2 = np.mean(err2)
            ratio2 = zc_err2 / (bg_err2 + 1e-12)
        else:
            ratio2 = 1.0

        lines.append(f"### {res.run_id}")
        lines.append(f"- Arm 1 direction changes: {len(zc1)}, error ratio at reversals: {ratio1:.2f}x")
        lines.append(f"- Arm 2 direction changes: {len(zc2)}, error ratio at reversals: {ratio2:.2f}x")
        if ratio1 > 1.5 or ratio2 > 1.5:
            lines.append(
                "- **Signal**: Elevated error at direction changes -> possible Coulomb friction"
            )
        lines.append("")

    return "\n".join(lines)


def _amplitude_decay_analysis(
    real_trajs: list[Trajectory],
    results: list[EvalResult],
) -> str:
    """Check amplitude decay over time in real data."""
    lines = ["## Amplitude Decay Analysis\n"]

    for real, res in zip(real_trajs, results):
        n = len(real)
        if n < 20:
            continue

        # Split into first and second half
        mid = n // 2
        amp1_first = np.std(real.theta1[:mid])
        amp1_second = np.std(real.theta1[mid:])
        amp2_first = np.std(real.theta2[:mid])
        amp2_second = np.std(real.theta2[mid:])

        decay1 = (amp1_second - amp1_first) / (amp1_first + 1e-12)
        decay2 = (amp2_second - amp2_first) / (amp2_first + 1e-12)

        lines.append(f"### {res.run_id}")
        lines.append(f"- Arm 1 amplitude change (2nd half vs 1st): {decay1:+.2%}")
        lines.append(f"- Arm 2 amplitude change (2nd half vs 1st): {decay2:+.2%}")
        if decay1 < -0.1 or decay2 < -0.1:
            lines.append("- **Signal**: Amplitude decay confirms dissipation present")
        lines.append("")

    return "\n".join(lines)


def generate_report(
    eval_results: list[EvalResult],
    sim_trajectories: list[Trajectory],
    real_trajectories: list[Trajectory],
    params: PendulumParams,
) -> str:
    """Generate a comprehensive diagnostic markdown report.

    This report is designed to be fed to Claude for the next iteration
    of simulator improvement.
    """
    div_times = [r.divergence_time for r in eval_results]
    angle_rmses = [r.angle_rmse for r in eval_results]

    sections = []

    # Summary
    sections.append("# Diagnostic Report: Candidate Simulator vs Real Data\n")
    sections.append("## Summary\n")
    sections.append(f"- Runs evaluated: {len(eval_results)}")
    sections.append(f"- Mean divergence time: {np.mean(div_times):.2f}s")
    sections.append(f"- Median divergence time: {np.median(div_times):.2f}s")
    sections.append(f"- Mean angle RMSE: {np.mean(angle_rmses):.4f} rad")
    sections.append(f"- Current model: **Ideal double pendulum (no dissipation)**")
    sections.append("")

    # Per-run summary table
    sections.append("## Per-Run Results\n")
    sections.append("| Run | Div Time (s) | Angle RMSE | Energy RMSE | theta1 RMSE | theta2 RMSE |")
    sections.append("|-----|-------------|------------|-------------|-------------|-------------|")
    for r in eval_results:
        sections.append(
            f"| {r.run_id} | {r.divergence_time:.2f} | {r.angle_rmse:.4f} | "
            f"{r.energy_rmse:.4f} | {r.theta1_rmse:.4f} | {r.theta2_rmse:.4f} |"
        )
    sections.append("")

    # Detailed analyses
    sections.append(
        _energy_decay_analysis(sim_trajectories, real_trajectories, eval_results, params)
    )
    sections.append(
        _velocity_drag_analysis(sim_trajectories, real_trajectories, eval_results)
    )
    sections.append(
        _friction_analysis(sim_trajectories, real_trajectories, eval_results)
    )
    sections.append(
        _amplitude_decay_analysis(real_trajectories, eval_results)
    )

    # Recommendations
    sections.append("## Recommended Next Steps\n")
    sections.append(
        "Based on the diagnostic signals above, consider adding the following "
        "physics to `candidate_sim.py`:\n"
    )
    sections.append("1. **Viscous friction** (b1, b2): Torque proportional to angular velocity")
    sections.append("2. **Air drag** (cd1, cd2): Torque proportional to velocity squared")
    sections.append("3. **Coulomb friction** (c1, c2): Constant torque opposing motion direction")
    sections.append("")
    sections.append(
        "The `PendulumParams` dataclass already has slots for these coefficients. "
        "Set them to nonzero values and add corresponding terms to `equations_of_motion()`."
    )
    sections.append("")

    return "\n".join(sections)
