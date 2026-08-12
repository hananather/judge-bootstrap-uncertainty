"""Journal-scale evidence design for the judge-bootstrap demonstration.

This remains a single readable research script.  It uses the elementary binary
model from ``judge_bootstrap_demo.py`` and adds denser, decision-relevant
scenarios without becoming a package or a generic experiment framework.

The script deliberately runs the smoke configuration when executed directly.
The report-scale configuration exists as ``run_final`` but is an explicit
second step after the smoke figures have been inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path
import csv
import hashlib
import json
import time

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np

from judge_bootstrap_demo import (
    NOMINAL_COVERAGE,
    Z_975,
    Interval,
    Scenario,
    delta_interval,
    draw_trial,
    full_fixed_stratum_bootstrap_interval,
    textbook_plus_target_resampling_interval,
    textbook_validation_only_interval,
)


ROOT = Path(__file__).resolve().parent
BASE_SEED = 20260810
RHO_TARGETS = (0.05, 0.10, 0.20, 0.40, 0.75, 1.00, 1.50, 2.00)
DRIFT_GRID = tuple(round(value, 4) for value in np.linspace(0.0, 0.10, 9))
PALETTE = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#4D4D4D")


@dataclass(frozen=True)
class JournalCell:
    scenario: Scenario
    key: str
    regime: str
    rho_target: float | None
    rho_actual: float | None
    sensitivity_decline: float | None = None
    specificity_decline: float | None = None
    run_ablation: bool = False


@dataclass
class RunningMoments:
    """Sufficient statistics for a scalar summary, avoiding millions of dicts."""

    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0

    def add(self, value: float) -> None:
        if np.isfinite(value):
            self.count += 1
            self.total += value
            self.total_squares += value * value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else np.nan

    @property
    def variance(self) -> float:
        if self.count < 2:
            return np.nan
        centered = self.total_squares - self.total**2 / self.count
        return max(centered / (self.count - 1), 0.0)

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def mcse(self) -> float:
        return self.sd / sqrt(self.count) if self.count else np.nan


@dataclass
class MethodSummary:
    count: int = 0
    coverage_count: int = 0
    theta_hat: RunningMoments | None = None
    width: RunningMoments | None = None
    bootstrap_sd_squared: RunningMoments | None = None
    analytic_se_squared: RunningMoments | None = None
    weak_j: RunningMoments | None = None
    discarded: RunningMoments | None = None
    theta_hat_values: list[float] | None = None
    reported_se_squared_values: list[float] | None = None

    def __post_init__(self) -> None:
        self.theta_hat = RunningMoments()
        self.width = RunningMoments()
        self.bootstrap_sd_squared = RunningMoments()
        self.analytic_se_squared = RunningMoments()
        self.weak_j = RunningMoments()
        self.discarded = RunningMoments()
        self.theta_hat_values = []
        self.reported_se_squared_values = []

    def add(self, theta: float, theta_hat: float, interval: Interval) -> None:
        self.count += 1
        self.theta_hat.add(theta_hat)
        self.theta_hat_values.append(theta_hat)
        self.width.add(interval.width)
        self.coverage_count += int(interval.lower <= theta <= interval.upper)
        if np.isfinite(interval.raw_bootstrap_sd):
            self.bootstrap_sd_squared.add(interval.raw_bootstrap_sd**2)
            self.reported_se_squared_values.append(interval.raw_bootstrap_sd**2)
        elif np.isfinite(interval.analytic_se):
            self.analytic_se_squared.add(interval.analytic_se**2)
            self.reported_se_squared_values.append(interval.analytic_se**2)
        else:
            self.reported_se_squared_values.append(np.nan)
        self.weak_j.add(interval.weak_j_fraction)
        self.discarded.add(interval.discarded_fraction)


def stable_seed(key: str) -> np.random.SeedSequence:
    """Make scenario randomization reproducible without relying on Python hashes."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    words = np.frombuffer(digest[:16], dtype=np.uint32).tolist()
    return np.random.SeedSequence([BASE_SEED, *words])


def normal_cdf(value: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(value)
    result = np.vectorize(lambda item: 0.5 * (1.0 + erf(float(item) / sqrt(2.0))))(array)
    return float(result) if result.ndim == 0 else result


def wilson_interval(successes: int, trials: int, z: float = Z_975) -> tuple[float, float]:
    """A binomial Monte Carlo interval for empirical coverage."""
    if trials == 0:
        return np.nan, np.nan
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    radius = z * sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2)) / denominator
    return center - radius, center + radius


def validation_variance_numerator(theta: float, tpr: float, tnr: float, n_pass: int, n_fail: int) -> float:
    """The numerator of the calibration component before division by J squared."""
    return (
        theta**2 * tpr * (1.0 - tpr) / n_pass
        + (1.0 - theta) ** 2 * tnr * (1.0 - tnr) / n_fail
    )


def target_judge_rate(theta: float, tpr: float, tnr: float) -> float:
    return theta * tpr + (1.0 - theta) * (1.0 - tnr)


def solve_target_size_for_rho(
    theta: float, tpr: float, tnr: float, n_pass: int, n_fail: int, rho_target: float
) -> tuple[int, float]:
    """Choose integer m so V_target/V_validation is as close as possible to rho."""
    q = target_judge_rate(theta, tpr, tnr)
    numerator = validation_variance_numerator(theta, tpr, tnr, n_pass, n_fail)
    proposed_m = q * (1.0 - q) / (rho_target * numerator)
    target_size = max(1, int(round(proposed_m)))
    actual_rho = q * (1.0 - q) / (target_size * numerator)
    return target_size, actual_rho


def study_a_cells() -> list[JournalCell]:
    """Stable-calibration cells, indexed by the scientifically useful variance ratio rho."""
    cells: list[JournalCell] = []
    symmetric_regimes = (
        ("low success, N=200", 0.20, 0.875, 0.875, 200),
        ("low success, N=800", 0.20, 0.875, 0.875, 800),
        ("balanced success, N=200", 0.50, 0.875, 0.875, 200),
        ("balanced success, N=800", 0.50, 0.875, 0.875, 800),
        ("high success, N=200", 0.80, 0.875, 0.875, 200),
        ("high success, N=800", 0.80, 0.875, 0.875, 800),
        ("high success, asymmetric, N=400", 0.80, 0.95, 0.80, 400),
    )
    for regime, theta, tpr, tnr, validation_size in symmetric_regimes:
        for rho_target in RHO_TARGETS:
            n_pass = validation_size // 2
            n_fail = validation_size // 2
            target_size, rho_actual = solve_target_size_for_rho(
                theta, tpr, tnr, n_pass, n_fail, rho_target
            )
            key = (
                f"A|{regime}|rho_target={rho_target:.2f}|m={target_size}|"
                f"N={validation_size}"
            )
            ablation_rhos = (0.10, 1.00, 2.00)
            run_ablation = regime == "high success, N=200" and rho_target in ablation_rhos
            cells.append(
                JournalCell(
                    scenario=Scenario(
                        study="A", name=key, theta=theta,
                        validation_tpr=tpr, validation_tnr=tnr,
                        target_tpr=tpr, target_tnr=tnr,
                        target_size=target_size, n_pass=n_pass, n_fail=n_fail,
                    ),
                    key=key, regime=regime, rho_target=rho_target,
                    rho_actual=rho_actual, run_ablation=run_ablation,
                )
            )
    return cells


def study_b_cells() -> list[JournalCell]:
    cells: list[JournalCell] = []
    for sensitivity_decline in DRIFT_GRID:
        for specificity_decline in DRIFT_GRID:
            key = f"B|da={sensitivity_decline:.4f}|db={specificity_decline:.4f}"
            cells.append(
                JournalCell(
                    scenario=Scenario(
                        study="B", name=key, theta=0.80,
                        validation_tpr=0.875, validation_tnr=0.875,
                        target_tpr=0.875 - sensitivity_decline,
                        target_tnr=0.875 - specificity_decline,
                        target_size=2000, n_pass=100, n_fail=100,
                    ),
                    key=key, regime="target calibration drift", rho_target=None,
                    rho_actual=None, sensitivity_decline=sensitivity_decline,
                    specificity_decline=specificity_decline,
                )
            )
    return cells


def finalise_summary(cell: JournalCell, method: str, statistics: MethodSummary) -> dict[str, float | int | str]:
    theta_moments = statistics.theta_hat
    width_moments = statistics.width
    bootstrap_sd_squared = statistics.bootstrap_sd_squared
    analytic_se_squared = statistics.analytic_se_squared
    empirical_sd = theta_moments.sd
    rms_bootstrap_sd = sqrt(bootstrap_sd_squared.mean) if bootstrap_sd_squared.count else np.nan
    rms_analytic_se = sqrt(analytic_se_squared.mean) if analytic_se_squared.count else np.nan
    rms_reported_se = rms_bootstrap_sd if np.isfinite(rms_bootstrap_sd) else rms_analytic_se
    ratio_low, ratio_high = outer_bootstrap_ratio_interval(
        statistics.theta_hat_values,
        statistics.reported_se_squared_values,
        f"{cell.key}|{method}|se_calibration_ratio",
    )
    coverage = statistics.coverage_count / statistics.count
    coverage_low, coverage_high = wilson_interval(statistics.coverage_count, statistics.count)
    bias = theta_moments.mean - cell.scenario.theta
    bias_mcse = theta_moments.mcse
    analytic_bias = np.nan
    if cell.scenario.study == "B":
        analytic_bias = first_order_drift_bias(cell.scenario, float(cell.sensitivity_decline), float(cell.specificity_decline))
    return {
        "study": cell.scenario.study,
        "scenario_key": cell.key,
        "regime": cell.regime,
        "method": method,
        "theta": cell.scenario.theta,
        "validation_tpr": cell.scenario.validation_tpr,
        "validation_tnr": cell.scenario.validation_tnr,
        "target_tpr": cell.scenario.target_tpr,
        "target_tnr": cell.scenario.target_tnr,
        "target_size_m": cell.scenario.target_size,
        "validation_total_N": cell.scenario.validation_size,
        "n_pass": cell.scenario.n_pass,
        "n_fail": cell.scenario.n_fail,
        "rho_target": cell.rho_target if cell.rho_target is not None else np.nan,
        "rho_actual": cell.rho_actual if cell.rho_actual is not None else np.nan,
        "sensitivity_decline": cell.sensitivity_decline if cell.sensitivity_decline is not None else np.nan,
        "specificity_decline": cell.specificity_decline if cell.specificity_decline is not None else np.nan,
        "replications": statistics.count,
        "coverage_count": statistics.coverage_count,
        "coverage": coverage,
        "coverage_wilson_low": coverage_low,
        "coverage_wilson_high": coverage_high,
        "coverage_mcse": sqrt(coverage * (1.0 - coverage) / statistics.count),
        "bias": bias,
        "bias_mcse": bias_mcse,
        "bias_mc95_low": bias - Z_975 * bias_mcse,
        "bias_mc95_high": bias + Z_975 * bias_mcse,
        "rmse": sqrt(theta_moments.variance * (statistics.count - 1) / statistics.count + bias**2),
        "empirical_sd": empirical_sd,
        "mean_width": width_moments.mean,
        "mean_width_mcse": width_moments.mcse,
        "rms_raw_bootstrap_sd": rms_bootstrap_sd,
        "rms_analytic_se": rms_analytic_se,
        "rms_reported_se": rms_reported_se,
        "se_calibration_ratio": rms_reported_se / empirical_sd if empirical_sd > 0.0 else np.nan,
        "se_calibration_ratio_low": ratio_low,
        "se_calibration_ratio_high": ratio_high,
        "weak_j_fraction": statistics.weak_j.mean,
        "discarded_fraction": statistics.discarded.mean,
        "analytic_bias": analytic_bias,
        "empirical_minus_analytic_bias": bias - analytic_bias if np.isfinite(analytic_bias) else np.nan,
    }


def outer_bootstrap_ratio_interval(
    theta_hat_values: list[float], reported_se_squared_values: list[float], key: str, draws: int = 449
) -> tuple[float, float]:
    """Outer-replication bootstrap interval for the reported-SE calibration ratio."""
    theta_hat = np.asarray(theta_hat_values, dtype=float)
    reported_se_squared = np.asarray(reported_se_squared_values, dtype=float)
    valid = np.isfinite(theta_hat) & np.isfinite(reported_se_squared)
    theta_hat = theta_hat[valid]
    reported_se_squared = reported_se_squared[valid]
    if len(theta_hat) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(stable_seed(key))
    ratios = np.empty(draws)
    for draw in range(draws):
        indices = rng.integers(0, len(theta_hat), size=len(theta_hat))
        empirical_sd = np.std(theta_hat[indices], ddof=1)
        ratios[draw] = sqrt(np.mean(reported_se_squared[indices])) / empirical_sd
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def run_study_a_cell(cell: JournalCell, replications: int, bootstrap_draws: int) -> list[dict[str, float | int | str]]:
    """Run paired outer data: all A methods see the same sample in each replication."""
    outer_seed, textbook_seed, full_seed, ablation_seed = stable_seed(cell.key).spawn(4)
    outer_rng = np.random.default_rng(outer_seed)
    textbook_rng = np.random.default_rng(textbook_seed)
    full_rng = np.random.default_rng(full_seed)
    ablation_rng = np.random.default_rng(ablation_seed)
    methods = {
        "textbook_validation_only": MethodSummary(),
        "full_fixed_stratum_bootstrap": MethodSummary(),
        "delta_method": MethodSummary(),
    }
    if cell.run_ablation:
        methods["textbook_plus_target_resampling"] = MethodSummary()

    for _ in range(replications):
        trial = draw_trial(cell.scenario, outer_rng)
        theta_hat = float(trial["theta_hat"])
        intervals: dict[str, Interval] = {
            "textbook_validation_only": textbook_validation_only_interval(
                trial, cell.scenario, bootstrap_draws, textbook_rng
            ),
            "full_fixed_stratum_bootstrap": full_fixed_stratum_bootstrap_interval(
                trial, cell.scenario, bootstrap_draws, full_rng
            ),
            "delta_method": delta_interval(trial, cell.scenario),
        }
        if cell.run_ablation:
            intervals["textbook_plus_target_resampling"] = textbook_plus_target_resampling_interval(
                trial, cell.scenario, bootstrap_draws, ablation_rng
            )
        for method, interval in intervals.items():
            methods[method].add(cell.scenario.theta, theta_hat, interval)
    return [finalise_summary(cell, method, summary) for method, summary in methods.items()]


def run_study_b_cell(cell: JournalCell, replications: int, bootstrap_draws: int) -> dict[str, float | int | str]:
    """Only full bootstrap is relevant in the drift sensitivity study."""
    outer_seed, full_seed = stable_seed(cell.key).spawn(2)
    outer_rng = np.random.default_rng(outer_seed)
    full_rng = np.random.default_rng(full_seed)
    summary = MethodSummary()
    for _ in range(replications):
        trial = draw_trial(cell.scenario, outer_rng)
        interval = full_fixed_stratum_bootstrap_interval(
            trial, cell.scenario, bootstrap_draws, full_rng
        )
        summary.add(cell.scenario.theta, float(trial["theta_hat"]), interval)
    return finalise_summary(cell, "full_fixed_stratum_bootstrap", summary)


def first_order_drift_bias(scenario: Scenario, sensitivity_decline: float, specificity_decline: float) -> float:
    """Probability-limit transport bias; finite-sample ratio bias may remain."""
    j_validation = scenario.validation_tpr + scenario.validation_tnr - 1.0
    return (
        -scenario.theta * sensitivity_decline
        + (1.0 - scenario.theta) * specificity_decline
    ) / j_validation


def first_order_drift_coverage(scenario: Scenario, sensitivity_decline: float, specificity_decline: float) -> float:
    """A displayed normal approximation, not a fitted curve or an exact guarantee."""
    target_tpr = scenario.validation_tpr - sensitivity_decline
    target_tnr = scenario.validation_tnr - specificity_decline
    q_target = target_judge_rate(scenario.theta, target_tpr, target_tnr)
    j_validation = scenario.validation_tpr + scenario.validation_tnr - 1.0
    limiting_estimate = (q_target + scenario.validation_tnr - 1.0) / j_validation
    variance = (
        q_target * (1.0 - q_target) / scenario.target_size
        + limiting_estimate**2 * scenario.validation_tpr * (1.0 - scenario.validation_tpr) / scenario.n_pass
        + (1.0 - limiting_estimate) ** 2 * scenario.validation_tnr * (1.0 - scenario.validation_tnr) / scenario.n_fail
    ) / j_validation**2
    standard_error = sqrt(variance)
    standardized_bias = first_order_drift_bias(scenario, sensitivity_decline, specificity_decline) / standard_error
    return float(normal_cdf(Z_975 - standardized_bias) - normal_cdf(-Z_975 - standardized_bias))


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def panel_letter(axis: plt.Axes, letter: str) -> None:
    """Place panel labels inside the plotting area so they never collide with axis labels."""
    axis.text(
        0.015, 0.985, letter, transform=axis.transAxes, fontsize=11,
        fontweight="bold", va="top", ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.35},
    )


def journal_figure_study_a(summary: list[dict[str, float | int | str]], output: Path) -> None:
    rows = [row for row in summary if row["study"] == "A"]
    with plt.rc_context({
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.75))
        axis_a, axis_b, axis_c, axis_d = axes.flat

        rho_law = np.geomspace(min(RHO_TARGETS), max(RHO_TARGETS), 240)
        coverage_law = 2.0 * normal_cdf(Z_975 / np.sqrt(1.0 + rho_law)) - 1.0
        rho_90 = (Z_975 / 1.6448536269514722) ** 2 - 1.0
        regimes = list(dict.fromkeys(str(row["regime"]) for row in rows))
        regime_offsets = dict(zip(regimes, np.geomspace(0.975, 1.025, len(regimes))))

        for regime in regimes:
            points = [
                row for row in rows
                if row["regime"] == regime and row["method"] == "textbook_validation_only"
            ]
            color = "#B45F8C" if "asymmetric" in regime else "#777777"
            marker = "D" if "asymmetric" in regime else "o"
            axis_a.errorbar(
                [float(row["rho_actual"]) * regime_offsets[regime] for row in points],
                [100.0 * float(row["coverage"]) for row in points],
                yerr=np.vstack([
                    [100.0 * (float(row["coverage"]) - float(row["coverage_wilson_low"])) for row in points],
                    [100.0 * (float(row["coverage_wilson_high"]) - float(row["coverage"])) for row in points],
                ]),
                color=color, marker=marker, linestyle="none", markersize=2.7,
                capsize=1.2, alpha=0.62,
            )
        axis_a.plot(
            rho_law, 100.0 * coverage_law, color="#D55E00", linewidth=1.8,
            label="First-order coverage law",
        )
        axis_a.axhline(95.0, color="#666666", linestyle="--", linewidth=0.9)
        axis_a.axvline(rho_90, color="#888888", linestyle=":", linewidth=0.9)
        rho_one_rows = [
            row for row in rows
            if row["method"] == "textbook_validation_only" and abs(float(row["rho_target"]) - 1.0) < 1e-9
        ]
        rho_one_coverage = 100.0 * float(np.mean([float(row["coverage"]) for row in rho_one_rows]))
        axis_a.annotate(
            f"ρ = 1: half the variance omitted\nmean coverage {rho_one_coverage:.1f}%",
            xy=(1.0, rho_one_coverage), xytext=(0.20, 73.2),
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8},
            fontsize=6.8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        rho_two_rows = [
            row for row in rows
            if row["method"] == "textbook_validation_only" and abs(float(row["rho_target"]) - 2.0) < 1e-9
        ]
        rho_two_coverage = 100.0 * float(np.mean([float(row["coverage"]) for row in rho_two_rows]))
        axis_a.text(2.03, rho_two_coverage, f"{rho_two_coverage:.1f}%", fontsize=6.8, va="center")
        axis_a.text(
            rho_90 * 1.04, 96.1, "First-order coverage\nfalls below 90%",
            fontsize=6.6, va="top", color="#555555",
        )
        axis_a.text(0.052, 94.1, "first-order law", color="#D55E00", fontsize=6.5)
        axis_a.set_xscale("log")
        axis_a.set_xlim(0.04, 2.35)
        axis_a.set_ylim(68, 97)
        axis_a.set_xlabel("Variance ratio ρ (log scale)")
        axis_a.set_ylabel("Coverage of nominal 95% interval (%)")
        axis_a.grid(alpha=0.16)
        axis_a.set_title("Coverage falls as omitted variance grows")
        panel_letter(axis_a, "A")

        for method, color, marker, label in (
            ("textbook_validation_only", "#D55E00", "o", "Validation-only bootstrap"),
            ("full_fixed_stratum_bootstrap", "#0072B2", "s", "Full bootstrap"),
        ):
            points = [row for row in rows if row["method"] == method]
            axis_b.errorbar(
                [float(row["rho_actual"]) for row in points],
                [float(row["se_calibration_ratio"]) for row in points],
                yerr=np.vstack([
                    [float(row["se_calibration_ratio"]) - float(row["se_calibration_ratio_low"]) for row in points],
                    [float(row["se_calibration_ratio_high"]) - float(row["se_calibration_ratio"]) for row in points],
                ]),
                color=color, marker=marker, linestyle="none", markersize=2.5,
                capsize=1.0, alpha=0.38, label=label,
            )
        axis_b.plot(
            rho_law, (1.0 + rho_law) ** -0.5, color="#D55E00", linewidth=1.7,
            linestyle="--", label=r"First-order law $(1+\rho)^{-1/2}$",
        )
        axis_b.axhline(1.0, color="#0072B2", linewidth=1.5, linestyle="--")
        rho_one_textbook = np.mean([
            float(row["se_calibration_ratio"]) for row in rho_one_rows
        ])
        rho_one_full = np.mean([
            float(row["se_calibration_ratio"]) for row in rows
            if row["method"] == "full_fixed_stratum_bootstrap"
            and abs(float(row["rho_target"]) - 1.0) < 1e-9
        ])
        axis_b.annotate(
            f"ρ = 1\nvalidation-only: {rho_one_textbook:.2f}\nfull: {rho_one_full:.2f}",
            xy=(1.0, rho_one_textbook), xytext=(0.17, 0.58),
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8},
            fontsize=6.8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        axis_b.set_xscale("log")
        axis_b.set_xlim(0.04, 2.35)
        axis_b.set_ylim(0.52, 1.08)
        axis_b.set_xlabel("Variance ratio ρ (log scale)")
        axis_b.text(1.42, 1.012, "full bootstrap", color="#0072B2", fontsize=6.5)
        axis_b.text(1.34, 0.62, "validation-only law", color="#D55E00", fontsize=6.5, rotation=-24)
        axis_b.text(0.052, 0.535, "points: simulation; dashed line: first-order", fontsize=6.0, color="#555555")
        axis_b.set_ylabel("Reported SE ÷ empirical SD")
        axis_b.grid(alpha=0.16)
        axis_b.set_title("Reported uncertainty becomes too small (1 = calibrated)")
        panel_letter(axis_b, "B")

        ablation_regime = "high success, N=200"
        ablation_rhos = (0.10, 1.00, 2.00)
        ablation_methods = (
            ("textbook_validation_only", "#D55E00", "o", "Validation only"),
            ("textbook_plus_target_resampling", "#0072B2", "^", "+ target resampling"),
            ("full_fixed_stratum_bootstrap", "#009E73", "s", "Full fixed-strata bootstrap"),
        )
        positions = np.arange(len(ablation_rhos), dtype=float)
        for offset, (method, color, marker, label) in zip((-0.14, 0.0, 0.14), ablation_methods):
            points = []
            for rho in ablation_rhos:
                points.append(next(
                    row for row in rows
                    if row["regime"] == ablation_regime and row["method"] == method
                    and abs(float(row["rho_target"]) - rho) < 1e-9
                ))
            axis_c.errorbar(
                positions + offset,
                [100.0 * float(row["coverage"]) for row in points],
                yerr=np.vstack([
                    [100.0 * (float(row["coverage"]) - float(row["coverage_wilson_low"])) for row in points],
                    [100.0 * (float(row["coverage_wilson_high"]) - float(row["coverage"])) for row in points],
                ]),
                color=color, marker=marker, linestyle="none", markersize=4.2,
                capsize=2.0, label=label,
            )
        axis_c.axhline(95.0, color="#666666", linestyle="--", linewidth=0.9)
        axis_c.annotate(
            "82.1% → 94.4%",
            xy=(1.0, 94.36), xytext=(0.53, 88.2),
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8},
            fontsize=6.9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
        axis_c.set_xticks(positions)
        axis_c.set_xticklabels(["0.1", "1", "2"])
        axis_c.set_ylim(68, 97)
        axis_c.set_xlabel("Variance ratio ρ")
        axis_c.set_ylabel("Coverage of nominal 95% interval (%)")
        axis_c.grid(axis="y", alpha=0.16)
        axis_c.legend(frameon=False, loc="lower left")
        axis_c.set_title("Resampling target cases restores coverage")
        panel_letter(axis_c, "C")

        allocation_regimes = (
            ("20%\nsuccess", 0.20, 0.875, 0.875),
            ("50%\nsuccess", 0.50, 0.875, 0.875),
            ("80%\nsuccess", 0.80, 0.875, 0.875),
            ("80%, asymmetric\njudge", 0.80, 0.95, 0.80),
        )
        pass_shares = []
        for _, theta, tpr, tnr in allocation_regimes:
            ratio = (
                theta * sqrt(tpr * (1.0 - tpr))
                / ((1.0 - theta) * sqrt(tnr * (1.0 - tnr)))
            )
            pass_shares.append(100.0 * ratio / (1.0 + ratio))
        fail_shares = [100.0 - share for share in pass_shares]
        x = np.arange(len(allocation_regimes))
        axis_d.bar(x, pass_shares, color="#E69F00", width=0.70, label="True Pass stratum")
        axis_d.bar(
            x, fail_shares, bottom=pass_shares, color="#CC79A7", width=0.70,
            label="True Fail stratum",
        )
        for position, (pass_share, fail_share) in enumerate(zip(pass_shares, fail_shares)):
            axis_d.text(position, pass_share / 2.0, f"{pass_share:.0f}%", ha="center", va="center", fontsize=7.1)
            axis_d.text(position, pass_share + fail_share / 2.0, f"{fail_share:.0f}%", ha="center", va="center", fontsize=7.1)
        axis_d.set_ylim(0, 100)
        axis_d.set_xticks(x)
        axis_d.set_xticklabels([item[0] for item in allocation_regimes])
        axis_d.set_ylabel("Precision-optimal label share (%)")
        axis_d.set_title("Label allocation should follow the estimand\nGold: true Pass; rose: true Fail")
        panel_letter(axis_d, "D")

        fig.suptitle(
            "Holding a random target rate fixed creates predictable overconfidence",
            y=0.985, fontsize=11.0,
        )
        fig.text(
            0.5, 0.012,
            "Seven designs are shown per variance ratio; coverage bars are Wilson intervals. "
            "Panel D minimizes estimator variance, not diagnostic error.",
            ha="center", fontsize=6.9,
        )
        fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.94), h_pad=2.1, w_pad=1.45)
        fig.savefig(output.with_suffix(".png"), dpi=300)
        fig.savefig(output.with_suffix(".svg"))
        plt.close(fig)


def journal_figure_study_b(summary: list[dict[str, float | int | str]], output: Path) -> None:
    rows = [row for row in summary if row["study"] == "B"]
    declines = np.array(DRIFT_GRID)
    bias = np.empty((len(declines), len(declines)))
    coverage = np.empty_like(bias)
    lower = np.empty_like(bias)
    upper = np.empty_like(bias)
    analytic = np.empty_like(bias)
    lookup = {(float(row["sensitivity_decline"]), float(row["specificity_decline"])): row for row in rows}
    for row_index, da in enumerate(declines):
        for column_index, db in enumerate(declines):
            row = lookup[(float(da), float(db))]
            bias[row_index, column_index] = 100.0 * float(row["bias"])
            coverage[row_index, column_index] = 100.0 * float(row["coverage"])
            lower[row_index, column_index] = 100.0 * float(row["coverage_wilson_low"])
            upper[row_index, column_index] = 100.0 * float(row["coverage_wilson_high"])
            analytic[row_index, column_index] = 100.0 * float(row["analytic_bias"])

    with plt.rc_context({
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
    }):
        figure = plt.figure(figsize=(7.2, 6.55))
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
        axis_a = figure.add_subplot(grid[0, 0])
        axis_b = figure.add_subplot(grid[0, 1])
        axis_c = figure.add_subplot(grid[1, :])

        step = 100.0 * (declines[1] - declines[0])
        cell_edges = np.linspace(-step / 2.0, 10.0 + step / 2.0, len(declines) + 1)
        maximum = max(abs(bias.min()), abs(bias.max()), 1.0)
        image_a = axis_a.pcolormesh(
            cell_edges, cell_edges, bias, shading="flat", cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        )
        contour_a = axis_a.contour(
            100.0 * declines, 100.0 * declines, analytic,
            levels=[0.0], colors="#222222", linewidths=1.2,
        )
        axis_a.annotate(
            "Bias cancels to first order",
            xy=(7.4, 1.85), xytext=(4.3, 3.2),
            arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 0.8},
            fontsize=6.4, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86},
        )
        axis_a.text(
            0.98, 0.98,
            "Per 1-point decline:\nsensitivity: −1.07 points\nspecificity: +0.27 points",
            transform=axis_a.transAxes, ha="right", va="top", fontsize=6.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86},
        )
        colorbar_a = figure.colorbar(
            image_a, ax=axis_a, orientation="horizontal", fraction=0.07, pad=0.18, aspect=24,
        )
        colorbar_a.set_label("Bias in estimated success rate (percentage points)", fontsize=6.8)
        colorbar_a.ax.tick_params(labelsize=6.5)
        axis_a.set_xlim(cell_edges[0], cell_edges[-1])
        axis_a.set_ylim(cell_edges[0], cell_edges[-1])
        axis_a.set_xticks((0, 2.5, 5, 7.5, 10))
        axis_a.set_yticks((0, 2.5, 5, 7.5, 10))
        axis_a.set_xlabel("Specificity decline (percentage points)")
        axis_a.set_ylabel("Sensitivity decline (percentage points)")
        axis_a.set_title("How the corrected estimate moves")
        panel_letter(axis_a, "A")

        reference_scenario = study_b_cells()[0].scenario
        coverage_bounds = (0, 50, 80, 90, 95, 100)
        coverage_colors = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.08, 0.95, 5)))
        coverage_norm = BoundaryNorm(coverage_bounds, coverage_colors.N)
        image_b = axis_b.pcolormesh(
            cell_edges, cell_edges, coverage, shading="flat",
            cmap=coverage_colors, norm=coverage_norm,
        )
        fine_declines = np.linspace(0.0, 0.10, 201)
        fine_sensitivity, fine_specificity = np.meshgrid(fine_declines, fine_declines, indexing="ij")
        fine_coverage = np.empty_like(fine_sensitivity)
        for row_index in range(fine_declines.size):
            for column_index in range(fine_declines.size):
                fine_coverage[row_index, column_index] = 100.0 * first_order_drift_coverage(
                    reference_scenario, float(fine_sensitivity[row_index, column_index]),
                    float(fine_specificity[row_index, column_index]),
                )
        for level, color in ((50, "white"), (80, "white"), (90, "#222222")):
            contour = axis_b.contour(
                100.0 * fine_declines, 100.0 * fine_declines, fine_coverage,
                levels=[level], colors=color, linewidths=0.9,
            )
            axis_b.clabel(
                contour, fmt={level: f"{level}% first-order"}, fontsize=6.2,
                inline=True,
            )
        colorbar_b = figure.colorbar(
            image_b, ax=axis_b, orientation="horizontal", fraction=0.07, pad=0.18, aspect=24,
            boundaries=coverage_bounds, ticks=(25, 65, 85, 92.5, 97.5),
        )
        colorbar_b.ax.set_xticklabels(("<50", "50–80", "80–90", "90–95", "≥95"))
        colorbar_b.set_label("Empirical coverage (%)", fontsize=7.0)
        colorbar_b.ax.tick_params(labelsize=6.3)
        axis_b.set_xlim(cell_edges[0], cell_edges[-1])
        axis_b.set_ylim(cell_edges[0], cell_edges[-1])
        axis_b.set_xticks((0, 2.5, 5, 7.5, 10))
        axis_b.set_yticks((0, 2.5, 5, 7.5, 10))
        axis_b.set_xlabel("Specificity decline (percentage points)")
        axis_b.set_ylabel("Sensitivity decline (percentage points)")
        axis_b.set_title("When the nominal 95% interval stops covering")
        panel_letter(axis_b, "B")

        slice_specs = (
            ("Sensitivity only", "#0072B2", "o", lambda d: (d, 0.0)),
            ("Specificity only", "#D55E00", "s", lambda d: (0.0, d)),
            ("Both decline", "#009E73", "^", lambda d: (d, d)),
        )
        continuous = np.linspace(0.0, 0.10, 201)
        five_point_values = {}
        for label, color, marker, selector in slice_specs:
            selected = []
            for decline in declines:
                da, db = selector(float(decline))
                selected.append(lookup[(da, db)])
            x = 100.0 * declines
            y = [100.0 * float(row["coverage"]) for row in selected]
            yerr = np.vstack([
                [100.0 * (float(row["coverage"]) - float(row["coverage_wilson_low"])) for row in selected],
                [100.0 * (float(row["coverage_wilson_high"]) - float(row["coverage"])) for row in selected],
            ])
            axis_c.errorbar(
                x, y, yerr=yerr, marker=marker, color=color,
                linewidth=1.4, markersize=3.8, capsize=1.8, label=label,
            )
            curve = [
                100.0 * first_order_drift_coverage(reference_scenario, *selector(float(value)))
                for value in continuous
            ]
            axis_c.plot(100.0 * continuous, curve, color=color, linestyle="--", linewidth=1.0)
            da_five, db_five = selector(0.05)
            five_point_values[label] = 100.0 * float(lookup[(da_five, db_five)]["coverage"])
        axis_c.axhline(95.0, color="#555555", linestyle=":", linewidth=1.0)
        axis_c.plot([], [], color="#555555", linestyle="--", linewidth=1.0, label="First-order reference")
        annotations = (
            ("Sensitivity only", "#0072B2", (5.0, five_point_values["Sensitivity only"]), (5.55, 65.0)),
            ("Both decline", "#009E73", (5.0, five_point_values["Both decline"]), (5.55, 83.5)),
            ("Specificity only", "#D55E00", (5.0, five_point_values["Specificity only"]), (5.55, 91.0)),
        )
        for label, color, point, text_position in annotations:
            axis_c.annotate(
                f"{label}: {point[1]:.1f}%",
                xy=point, xytext=text_position, color=color, fontsize=7.0,
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
            )
        axis_c.text(9.85, 96.2, "95% nominal", color="#555555", fontsize=6.8, ha="right")
        axis_c.set_xlim(-0.15, 10.15)
        axis_c.set_ylim(0, 100)
        axis_c.set_xlabel("Decline from validation to production (percentage points)")
        axis_c.set_ylabel("Coverage of nominal 95% interval (%)")
        axis_c.set_title("The same amount of drift can imply very different inferential risk")
        axis_c.legend(frameon=False, loc="lower left", ncol=2)
        axis_c.grid(alpha=0.16)
        panel_letter(axis_c, "C")

        maximum_wilson_half_width = 0.5 * float(np.max(upper - lower))
        figure.suptitle(
            "A bootstrap cannot repair sensitivity and specificity that change after validation",
            y=0.985, fontsize=11.0,
        )
        figure.text(
            0.5, 0.010,
            r"$\theta=0.80$; target $n=2{,}000$; validation $n_{Pass}=n_{Fail}=100$." "\n"
            f"Empirical: heatmap tiles, points, and Panel C solid lines." "\n"
            f"First-order: Panels A and B contours and Panel C dashed curves. "
            f"Maximum Wilson half-width: {maximum_wilson_half_width:.1f} points.",
            ha="center", va="bottom", fontsize=6.3, linespacing=1.15,
        )
        figure.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.14, hspace=0.58, wspace=0.34)
        figure.savefig(output.with_suffix(".png"), dpi=300)
        figure.savefig(output.with_suffix(".svg"))
        plt.close(figure)


def reader_figure_target_resampling(
    summary: list[dict[str, float | int | str]], output: Path
) -> None:
    """Show what changes when production sampling is added to the baseline."""
    rows = [row for row in summary if row["study"] == "A"]
    regime = "high success, N=200"
    scenarios = (
        (0.10, "Production uncertainty is small\n(10% of validation uncertainty)"),
        (1.00, "Production and validation\nuncertainty are equal"),
        (2.00, "Production uncertainty is twice\nvalidation uncertainty"),
    )
    y_positions = np.arange(len(scenarios))[::-1]
    methods = (
        ("textbook_validation_only", "#D55E00", "o", "Production rate held fixed"),
        ("textbook_plus_target_resampling", "#0072B2", "s", "Production cases resampled"),
    )

    with plt.rc_context({
        "font.size": 8.6,
        "axes.titlesize": 9.6,
        "axes.labelsize": 8.8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 8.0,
    }):
        figure, (coverage_axis, width_axis) = plt.subplots(
            1, 2, figsize=(8.5, 4.45), sharey=True,
            gridspec_kw={"width_ratios": (1.15, 0.85)},
        )
        for y, (rho, _) in zip(y_positions, scenarios):
            selected = {
                method: next(
                    row for row in rows
                    if row["regime"] == regime
                    and row["method"] == method
                    and abs(float(row["rho_target"]) - rho) < 1e-9
                )
                for method, *_ in methods
            }
            coverage_values = [
                100.0 * float(selected[method]["coverage"])
                for method, *_ in methods
            ]
            coverage_axis.plot(
                [min(coverage_values), max(coverage_values)], [y, y],
                color="#B8B8B8", linewidth=1.8, zorder=1,
            )
            for method, color, marker, label in methods:
                row = selected[method]
                value = 100.0 * float(row["coverage"])
                error = np.array([[
                    100.0 * (float(row["coverage"]) - float(row["coverage_wilson_low"]))
                ], [
                    100.0 * (float(row["coverage_wilson_high"]) - float(row["coverage"]))
                ]])
                coverage_axis.errorbar(
                    value, y, xerr=error, color=color, marker=marker,
                    markersize=6.0, capsize=3.2, capthick=1.4,
                    elinewidth=1.7, linewidth=1.3,
                    label=label if y == y_positions[0] else None, zorder=3,
                )
            coverage_axis.text(
                coverage_values[0] - 0.25, y + 0.20, f"{coverage_values[0]:.1f}",
                color=methods[0][1], ha="right", fontsize=7.3,
            )
            coverage_axis.text(
                coverage_values[1] + 0.25, y + 0.20, f"{coverage_values[1]:.1f}",
                color=methods[1][1], ha="left", fontsize=7.3,
            )
            coverage_axis.text(
                0.5 * sum(coverage_values), y - 0.23,
                f"+{coverage_values[1] - coverage_values[0]:.1f} points",
                color="#444444", ha="center", fontsize=7.1,
            )

            width_values = [
                100.0 * float(selected[method]["mean_width"])
                for method, *_ in methods
            ]
            width_axis.plot(
                [min(width_values), max(width_values)], [y, y],
                color="#B8B8B8", linewidth=1.8, zorder=1,
            )
            for method, color, marker, label in methods:
                value = 100.0 * float(selected[method]["mean_width"])
                width_axis.plot(
                    value, y, color=color, marker=marker, markersize=6.0,
                    label=label if y == y_positions[0] else None, zorder=3,
                )
            width_axis.text(
                width_values[0] - 0.20, y + 0.20, f"{width_values[0]:.1f}",
                color=methods[0][1], ha="right", fontsize=7.3,
            )
            width_axis.text(
                width_values[1] + 0.20, y + 0.20, f"{width_values[1]:.1f}",
                color=methods[1][1], ha="left", fontsize=7.3,
            )
            width_axis.text(
                0.5 * sum(width_values), y - 0.23,
                f"+{width_values[1] - width_values[0]:.1f} points",
                color="#444444", ha="center", fontsize=7.1,
            )

        coverage_axis.axvline(95.0, color="#555555", linestyle="--", linewidth=1.1)
        coverage_axis.text(
            95.0, y_positions[0] + 0.43, "95", ha="center",
            fontsize=7.4, color="#444444",
        )
        coverage_axis.set_xlim(68, 98)
        coverage_axis.set_ylim(-0.45, 2.45)
        coverage_axis.set_yticks(y_positions)
        coverage_axis.set_yticklabels([label for _, label in scenarios])
        coverage_axis.set_xlabel("Coverage of the nominal 95% interval (%)")
        coverage_axis.set_title("Did the interval contain the truth?")
        coverage_axis.grid(axis="x", alpha=0.15)

        width_axis.set_xlim(12, 27)
        width_axis.set_xlabel("Mean interval width (percentage points)")
        width_axis.set_title("How much wider was the interval?")
        width_axis.grid(axis="x", alpha=0.15)

        handles, labels = coverage_axis.get_legend_handles_labels()
        figure.legend(
            handles, labels, frameon=False, loc="lower center",
            bbox_to_anchor=(0.5, 0.055), ncol=2,
            handletextpad=0.5, columnspacing=1.8,
        )
        figure.suptitle(
            "One missing source of uncertainty explains the undercoverage",
            y=0.975, fontsize=11.2,
        )
        figure.text(
            0.5, 0.885,
            "The only change between the two methods is whether production cases are resampled.",
            ha="center", fontsize=8.2,
        )
        figure.text(
            0.5, 0.018,
            "Controlled example: true success rate 80%; 200 fixed, outcome-balanced validation labels. "
            "Coverage error bars are 95% Wilson intervals across 5,000 repeated studies.",
            ha="center", fontsize=7.0,
        )
        figure.subplots_adjust(
            left=0.33, right=0.98, top=0.74, bottom=0.24, wspace=0.22,
        )
        figure.savefig(output.with_suffix(".png"), dpi=300)
        figure.savefig(output.with_suffix(".svg"))
        plt.close(figure)


def reader_figure_calibration_drift(
    summary: list[dict[str, float | int | str]], output: Path
) -> None:
    """Explain why resampling cannot repair validation-to-production drift."""
    rows = [row for row in summary if row["study"] == "B"]
    scenarios = (
        (0.00, 0.00, "No change after validation", "#666666", "o"),
        (0.05, 0.00, "Recognizes 5 fewer of every 100 true passes", "#0072B2", "o"),
        (0.00, 0.05, "Recognizes 5 fewer of every 100 true failures", "#D55E00", "s"),
        (0.05, 0.05, "Both recognition rates decline by 5 points", "#009E73", "^"),
    )
    selected_rows = []
    for sensitivity_decline, specificity_decline, label, color, marker in scenarios:
        row = next(
            item for item in rows
            if abs(float(item["sensitivity_decline"]) - sensitivity_decline) < 1e-9
            and abs(float(item["specificity_decline"]) - specificity_decline) < 1e-9
        )
        selected_rows.append((row, label, color, marker))
    y_positions = np.arange(len(selected_rows))[::-1]

    with plt.rc_context({
        "font.size": 8.4,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.8,
    }):
        figure, (axis_bias, axis_coverage) = plt.subplots(
            1, 2, figsize=(7.2, 4.45), sharey=True,
            gridspec_kw={"width_ratios": (1.0, 1.15)},
        )
        for y, (row, _, color, marker) in zip(y_positions, selected_rows):
            bias = 100.0 * float(row["bias"])
            bias_error = np.array([[
                100.0 * (float(row["bias"]) - float(row["bias_mc95_low"]))
            ], [
                100.0 * (float(row["bias_mc95_high"]) - float(row["bias"]))
            ]])
            axis_bias.plot([0.0, bias], [y, y], color=color, linewidth=1.7, alpha=0.75)
            axis_bias.errorbar(
                bias, y, xerr=bias_error, color=color, marker=marker,
                markersize=5.8, capsize=3.2, capthick=1.4,
                elinewidth=1.7, linewidth=1.2, zorder=3,
            )
            bias_label = "about 0" if abs(bias) < 0.1 else f"{bias:+.1f} points"
            axis_bias.text(
                bias + (0.18 if bias >= 0 else -0.18), y + 0.17, bias_label,
                color=color, ha="left" if bias >= 0 else "right", fontsize=7.2,
            )

            coverage = 100.0 * float(row["coverage"])
            coverage_error = np.array([[
                100.0 * (float(row["coverage"]) - float(row["coverage_wilson_low"]))
            ], [
                100.0 * (float(row["coverage_wilson_high"]) - float(row["coverage"]))
            ]])
            axis_coverage.errorbar(
                coverage, y, xerr=coverage_error, color=color, marker=marker,
                markersize=5.8, capsize=3.2, capthick=1.4,
                elinewidth=1.7, linewidth=1.2, zorder=3,
            )
            axis_coverage.text(
                coverage - 0.55, y + 0.17, f"{coverage:.1f} out of 100",
                color=color, ha="right", fontsize=7.2,
            )

        axis_bias.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axis_bias.set_xlim(-6.4, 2.5)
        axis_bias.set_xlabel("Error in the estimated success rate\n(percentage points)")
        axis_bias.set_title("Bias in the corrected rate\n(0 = no systematic error)")
        axis_bias.grid(axis="x", alpha=0.15)
        axis_bias.set_yticks(y_positions)
        axis_bias.set_yticklabels([label for _, label, _, _ in selected_rows])

        axis_coverage.axvline(95.0, color="#555555", linestyle="--", linewidth=1.0)
        axis_coverage.set_xlim(65, 98)
        axis_coverage.set_xlabel("Intervals containing the true rate\n(out of 100 repeated studies)")
        axis_coverage.set_title("Coverage of the 95% interval\n(95 of 100 is the target)")
        axis_coverage.grid(axis="x", alpha=0.15)

        axis_bias.set_ylim(-0.45, 3.45)
        figure.suptitle(
            "Resampling cannot repair an evaluator that changed after validation",
            y=0.975, fontsize=11.0,
        )
        figure.text(
            0.5, 0.885,
            "The same five-point change has different consequences in an 80%-success population.",
            ha="center", fontsize=8.1,
        )
        figure.text(
            0.5, 0.055,
            "Catching true passes is sensitivity; catching true failures is specificity. "
            "A bootstrap reflects sampling variation, not stale evaluator accuracy.",
            ha="center", fontsize=7.0,
        )
        figure.subplots_adjust(left=0.46, right=0.98, top=0.74, bottom=0.22, wspace=0.28)
        figure.savefig(output.with_suffix(".png"), dpi=300)
        figure.savefig(output.with_suffix(".svg"))
        plt.close(figure)


def run_configuration(name: str, a_replications: int, b_replications: int, bootstrap_draws: int) -> Path:
    """Run a bounded configuration and write only full-precision summaries."""
    output = ROOT / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    summaries: list[dict[str, float | int | str]] = []
    for cell in study_a_cells():
        summaries.extend(run_study_a_cell(cell, a_replications, bootstrap_draws))
    for cell in study_b_cells():
        summaries.append(run_study_b_cell(cell, b_replications, bootstrap_draws))
    write_csv(output / "journal_summary.csv", summaries)
    journal_figure_study_a(summaries, output / "study_a_journal")
    journal_figure_study_b(summaries, output / "study_b_journal")
    reader_figure_target_resampling(summaries, output / "reader_target_resampling")
    reader_figure_calibration_drift(summaries, output / "reader_calibration_drift")
    metadata = {
        "configuration": name,
        "base_seed": BASE_SEED,
        "study_a_replications": a_replications,
        "study_b_replications": b_replications,
        "bootstrap_draws": bootstrap_draws,
        "study_a_cells": len(study_a_cells()),
        "study_b_cells": len(study_b_cells()),
        "elapsed_seconds": time.perf_counter() - start,
        "study_a_keys": [cell.key for cell in study_a_cells()],
        "study_b_keys": [cell.key for cell in study_b_cells()],
    }
    with (output / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return output


def run_smoke() -> Path:
    return run_configuration("journal_smoke", a_replications=300, b_replications=300, bootstrap_draws=250)


def run_final() -> Path:
    return run_configuration("journal_final", a_replications=5000, b_replications=3000, bootstrap_draws=1000)


if __name__ == "__main__":
    print(run_smoke())
