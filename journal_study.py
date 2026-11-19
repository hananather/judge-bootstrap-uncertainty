"""Journal-scale evidence design for the judge-bootstrap demonstration.

This remains a single readable research script.  It uses the elementary binary
model from ``judge_bootstrap_demo.py`` and adds denser, decision-relevant
scenarios without becoming a package or a generic experiment framework.

The script deliberately runs the smoke configuration when executed directly.
The report-scale configuration exists as ``run_final`` but is an explicit
second step after the smoke results have been checked.

The script writes summaries only.  ``make_figures.py`` draws the figures from
the saved CSV file.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np

from judge_bootstrap_demo import (
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


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
