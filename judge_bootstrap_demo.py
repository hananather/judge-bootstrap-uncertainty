"""A compact coverage audit for judge-calibrated success-rate intervals.

This is deliberately a small research script, not a package.  It reproduces
the textbook interval's central omission (holding the target judge rate fixed),
then compares it with a bootstrap that resamples every random component and a
transparent delta-method interval.  See METHOD.md before interpreting output.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
Z_975 = 1.959963984540054
NOMINAL_COVERAGE = 0.95


@dataclass(frozen=True)
class Scenario:
    """One independent-target, fixed-stratum-validation data-generating process."""

    study: str
    name: str
    theta: float
    validation_tpr: float
    validation_tnr: float
    target_tpr: float
    target_tnr: float
    target_size: int
    n_pass: int
    n_fail: int

    @property
    def validation_size(self) -> int:
        return self.n_pass + self.n_fail


@dataclass
class Interval:
    lower: float
    upper: float
    status: str
    weak_j_fraction: float = 0.0
    discarded_fraction: float = 0.0
    raw_bootstrap_sd: float = np.nan
    analytic_se: float = np.nan

    @property
    def width(self) -> float:
        return self.upper - self.lower


def rg_point(q_hat: float, tpr_hat: float, tnr_hat: float) -> tuple[float, float]:
    """Return the raw Rogan--Gladen estimate and its denominator J."""
    j_hat = tpr_hat + tnr_hat - 1.0
    if j_hat == 0.0:
        return np.nan, j_hat
    return (q_hat + tnr_hat - 1.0) / j_hat, j_hat


def draw_trial(scenario: Scenario, rng: np.random.Generator) -> dict[str, float | int]:
    """Draw one production sample and one outcome-balanced validation sample."""
    target_judge_positive_probability = (
        scenario.theta * scenario.target_tpr
        + (1.0 - scenario.theta) * (1.0 - scenario.target_tnr)
    )
    target_judge_passes = int(
        rng.binomial(scenario.target_size, target_judge_positive_probability)
    )
    validation_true_pass_judge_passes = int(
        rng.binomial(scenario.n_pass, scenario.validation_tpr)
    )
    validation_true_fail_judge_fails = int(
        rng.binomial(scenario.n_fail, scenario.validation_tnr)
    )

    q_hat = target_judge_passes / scenario.target_size
    tpr_hat = validation_true_pass_judge_passes / scenario.n_pass
    tnr_hat = validation_true_fail_judge_fails / scenario.n_fail
    theta_hat, j_hat = rg_point(q_hat, tpr_hat, tnr_hat)

    return {
        "target_judge_passes": target_judge_passes,
        "validation_true_pass_judge_passes": validation_true_pass_judge_passes,
        "validation_true_fail_judge_fails": validation_true_fail_judge_fails,
        "q_hat": q_hat,
        "tpr_hat": tpr_hat,
        "tnr_hat": tnr_hat,
        "theta_hat": theta_hat,
        "j_hat": j_hat,
    }


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """Percentile endpoints constrained only for final prevalence reporting."""
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(np.clip(lower, 0.0, 1.0)), float(np.clip(upper, 0.0, 1.0))


def textbook_validation_only_interval(
    trial: dict[str, float | int], scenario: Scenario, draws: int, rng: np.random.Generator
) -> Interval:
    """Reproduce the textbook logic: pool validation pairs and hold q-hat fixed.

    The validation sample deliberately has fixed human outcome strata, yet the
    original procedure resamples the pooled pairs.  It also drops degenerate
    resamples and clips every replicate.  Those behaviours are kept here only
    as an explanatory failure baseline.
    """
    n11 = int(trial["validation_true_pass_judge_passes"])
    n10 = scenario.n_pass - n11
    n00 = int(trial["validation_true_fail_judge_fails"])
    n01 = scenario.n_fail - n00
    probabilities = np.array([n11, n10, n01, n00], dtype=float) / scenario.validation_size
    boot = rng.multinomial(scenario.validation_size, probabilities, size=draws)

    human_passes = boot[:, 0] + boot[:, 1]
    human_fails = boot[:, 2] + boot[:, 3]
    valid_classes = (human_passes > 0) & (human_fails > 0)
    tpr_star = np.divide(
        boot[:, 0], human_passes, out=np.zeros(draws), where=valid_classes
    )
    tnr_star = np.divide(
        boot[:, 3], human_fails, out=np.zeros(draws), where=valid_classes
    )
    j_star = tpr_star + tnr_star - 1.0
    valid = valid_classes & (j_star > 0.0)
    discarded_fraction = 1.0 - float(np.mean(valid))
    if not np.any(valid):
        return Interval(np.nan, np.nan, "no_textbook_draws", discarded_fraction=1.0)

    # This is intentionally the textbook's clipped replicate distribution.
    raw_theta_star = (float(trial["q_hat"]) + tnr_star[valid] - 1.0) / j_star[valid]
    theta_star = np.clip(raw_theta_star, 0.0, 1.0)
    lower, upper = percentile_interval(theta_star)
    return Interval(
        lower, upper, "ok", discarded_fraction=discarded_fraction,
        raw_bootstrap_sd=float(np.std(raw_theta_star, ddof=1)),
    )


def textbook_plus_target_resampling_interval(
    trial: dict[str, float | int], scenario: Scenario, draws: int, rng: np.random.Generator
) -> Interval:
    """Ablate only the missing q-hat component; retain all other textbook choices."""
    n11 = int(trial["validation_true_pass_judge_passes"])
    n10 = scenario.n_pass - n11
    n00 = int(trial["validation_true_fail_judge_fails"])
    n01 = scenario.n_fail - n00
    probabilities = np.array([n11, n10, n01, n00], dtype=float) / scenario.validation_size
    boot = rng.multinomial(scenario.validation_size, probabilities, size=draws)
    q_star = rng.binomial(scenario.target_size, float(trial["q_hat"]), size=draws) / scenario.target_size

    human_passes = boot[:, 0] + boot[:, 1]
    human_fails = boot[:, 2] + boot[:, 3]
    valid_classes = (human_passes > 0) & (human_fails > 0)
    tpr_star = np.divide(
        boot[:, 0], human_passes, out=np.zeros(draws), where=valid_classes
    )
    tnr_star = np.divide(
        boot[:, 3], human_fails, out=np.zeros(draws), where=valid_classes
    )
    j_star = tpr_star + tnr_star - 1.0
    valid = valid_classes & (j_star > 0.0)
    discarded_fraction = 1.0 - float(np.mean(valid))
    if not np.any(valid):
        return Interval(np.nan, np.nan, "no_textbook_plus_q_draws", discarded_fraction=1.0)

    raw_theta_star = (q_star[valid] + tnr_star[valid] - 1.0) / j_star[valid]
    theta_star = np.clip(raw_theta_star, 0.0, 1.0)
    lower, upper = percentile_interval(theta_star)
    return Interval(
        lower, upper, "ok", discarded_fraction=discarded_fraction,
        raw_bootstrap_sd=float(np.std(raw_theta_star, ddof=1)),
    )


def full_fixed_stratum_bootstrap_interval(
    trial: dict[str, float | int], scenario: Scenario, draws: int, rng: np.random.Generator
) -> Interval:
    """Resample the production judge rate and each fixed validation outcome stratum.

    Weak denominators are recorded and surfaced.  They are never discarded to
    create an apparently narrower interval; if present, this simple regular-case
    routine returns the non-informative interval [0, 1] and a status flag.
    """
    q_star = rng.binomial(scenario.target_size, float(trial["q_hat"]), size=draws) / scenario.target_size
    tpr_star = (
        rng.binomial(scenario.n_pass, float(trial["tpr_hat"]), size=draws) / scenario.n_pass
    )
    tnr_star = (
        rng.binomial(scenario.n_fail, float(trial["tnr_hat"]), size=draws) / scenario.n_fail
    )
    j_star = tpr_star + tnr_star - 1.0
    weak = j_star <= 0.0
    weak_fraction = float(np.mean(weak))
    if np.any(weak):
        return Interval(0.0, 1.0, "weak_j", weak_j_fraction=weak_fraction)

    raw_theta_star = (q_star + tnr_star - 1.0) / j_star
    lower, upper = percentile_interval(raw_theta_star)
    return Interval(
        lower, upper, "ok", weak_j_fraction=weak_fraction,
        raw_bootstrap_sd=float(np.std(raw_theta_star, ddof=1)),
    )


def delta_interval(trial: dict[str, float | int], scenario: Scenario) -> Interval:
    """First-order interval with all three random components made explicit."""
    theta_hat = float(trial["theta_hat"])
    j_hat = float(trial["j_hat"])
    if not np.isfinite(theta_hat) or j_hat <= 0.0:
        return Interval(0.0, 1.0, "weak_j")
    q_hat = float(trial["q_hat"])
    tpr_hat = float(trial["tpr_hat"])
    tnr_hat = float(trial["tnr_hat"])
    variance = (
        q_hat * (1.0 - q_hat) / scenario.target_size
        + theta_hat**2 * tpr_hat * (1.0 - tpr_hat) / scenario.n_pass
        + (1.0 - theta_hat) ** 2 * tnr_hat * (1.0 - tnr_hat) / scenario.n_fail
    ) / j_hat**2
    standard_error = np.sqrt(max(variance, 0.0))
    lower = float(np.clip(theta_hat - Z_975 * standard_error, 0.0, 1.0))
    upper = float(np.clip(theta_hat + Z_975 * standard_error, 0.0, 1.0))
    return Interval(lower, upper, "ok", analytic_se=float(standard_error))


def one_row(
    scenario: Scenario,
    replication: int,
    method: str,
    trial: dict[str, float | int],
    interval: Interval,
) -> dict[str, float | int | str]:
    theta_hat = float(trial["theta_hat"])
    return {
        "study": scenario.study,
        "scenario": scenario.name,
        "replication": replication,
        "method": method,
        "theta": scenario.theta,
        "validation_tpr": scenario.validation_tpr,
        "validation_tnr": scenario.validation_tnr,
        "target_tpr": scenario.target_tpr,
        "target_tnr": scenario.target_tnr,
        "target_size": scenario.target_size,
        "validation_size": scenario.validation_size,
        "n_pass": scenario.n_pass,
        "n_fail": scenario.n_fail,
        "theta_hat": theta_hat,
        "j_hat": float(trial["j_hat"]),
        "lower": interval.lower,
        "upper": interval.upper,
        "covered": int(interval.lower <= scenario.theta <= interval.upper),
        "width": interval.width,
        "status": interval.status,
        "weak_j_fraction": interval.weak_j_fraction,
        "discarded_fraction": interval.discarded_fraction,
    }


def run_scenario(
    scenario: Scenario,
    replications: int,
    bootstrap_draws: int,
    include_textbook_plus_q: bool,
    seed_sequence: np.random.SeedSequence,
) -> list[dict[str, float | int | str]]:
    """Run paired outer trials: each method sees the same simulated data."""
    outer_seed, textbook_seed, ablation_seed, full_seed = seed_sequence.spawn(4)
    outer_rng = np.random.default_rng(outer_seed)
    textbook_rng = np.random.default_rng(textbook_seed)
    ablation_rng = np.random.default_rng(ablation_seed)
    full_rng = np.random.default_rng(full_seed)
    rows: list[dict[str, float | int | str]] = []

    for replication in range(replications):
        trial = draw_trial(scenario, outer_rng)
        methods = {
            "textbook_validation_only": textbook_validation_only_interval(
                trial, scenario, bootstrap_draws, textbook_rng
            ),
            "full_fixed_stratum_bootstrap": full_fixed_stratum_bootstrap_interval(
                trial, scenario, bootstrap_draws, full_rng
            ),
            "delta_method": delta_interval(trial, scenario),
        }
        if include_textbook_plus_q:
            methods["textbook_plus_target_resampling"] = (
                textbook_plus_target_resampling_interval(
                    trial, scenario, bootstrap_draws, ablation_rng
                )
            )
        rows.extend(
            one_row(scenario, replication, method, trial, interval)
            for method, interval in methods.items()
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    """Compute tidy method-by-scenario operating characteristics."""
    grouped: dict[tuple[str, str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["study"]), str(row["scenario"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, float | str]] = []
    for _, group in grouped.items():
        first = group[0]
        coverage = np.asarray([float(row["covered"]) for row in group])
        theta_hat = np.asarray([float(row["theta_hat"]) for row in group])
        width = np.asarray([float(row["width"]) for row in group])
        summaries.append(
            {
                "study": str(first["study"]),
                "scenario": str(first["scenario"]),
                "method": str(first["method"]),
                "target_size": float(first["target_size"]),
                "validation_size": float(first["validation_size"]),
                "judge_rate_decline": float(first["validation_tpr"])
                - float(first["target_tpr"]),
                "replications": float(len(group)),
                "coverage": float(coverage.mean()),
                "coverage_mcse": float(np.sqrt(coverage.mean() * (1.0 - coverage.mean()) / len(group))),
                "mean_width": float(width.mean()),
                "bias": float(theta_hat.mean() - float(first["theta"])),
                "rmse": float(np.sqrt(np.mean((theta_hat - float(first["theta"])) ** 2))),
                "weak_j_rate": float(np.mean([float(row["weak_j_fraction"]) for row in group])),
                "discarded_draw_rate": float(
                    np.mean([float(row["discarded_fraction"]) for row in group])
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_study_a(summary: list[dict[str, float | str]], output: Path) -> None:
    """Coverage and width by validation budget, faceted by target-sample size."""
    subset = [row for row in summary if row["study"] == "A"]
    methods = [
        "textbook_validation_only",
        "full_fixed_stratum_bootstrap",
        "delta_method",
    ]
    labels = {
        "textbook_validation_only": "Textbook: target rate held fixed",
        "full_fixed_stratum_bootstrap": "Full bootstrap",
        "delta_method": "Delta method",
    }
    colors = {
        "textbook_validation_only": "#c23b22",
        "full_fixed_stratum_bootstrap": "#0072b2",
        "delta_method": "#009e73",
    }
    markers = {method: marker for method, marker in zip(methods, ["o", "s", "^"])}
    target_sizes = sorted({int(row["target_size"]) for row in subset})

    fig, axes = plt.subplots(2, len(target_sizes), figsize=(11, 7), sharex="col")
    for column, target_size in enumerate(target_sizes):
        current = [row for row in subset if int(row["target_size"]) == target_size]
        for method in methods:
            points = sorted(
                [row for row in current if row["method"] == method],
                key=lambda row: float(row["validation_size"]),
            )
            if not points:
                continue
            x = [float(row["validation_size"]) for row in points]
            coverage = [float(row["coverage"]) for row in points]
            mcse = [float(row["coverage_mcse"]) for row in points]
            widths = [float(row["mean_width"]) for row in points]
            axes[0, column].errorbar(
                x, coverage, yerr=mcse, label=labels[method], color=colors[method],
                marker=markers[method], linewidth=2, capsize=3,
            )
            axes[1, column].plot(
                x, widths, label=labels[method], color=colors[method],
                marker=markers[method], linewidth=2,
            )
        axes[0, column].axhline(NOMINAL_COVERAGE, color="#333333", linestyle="--", linewidth=1)
        axes[0, column].set_title(f"Target sample m = {target_size}")
        axes[0, column].set_ylim(0.55, 1.01)
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[1, column].set_xlabel("Outcome-balanced validation labels")
    axes[0, 0].set_ylabel("Empirical 95% coverage")
    axes[1, 0].set_ylabel("Mean interval width")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        "Production-population success probability; target cases are sampled at random, not treated as one fixed finite batch.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.suptitle(
        "Study A: holding the target judge rate fixed produces intervals that miss too often",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def plot_study_b(summary: list[dict[str, float | str]], output: Path) -> None:
    """Bias and coverage under stylized target judge drift."""
    points = sorted(
        [row for row in summary if row["study"] == "B"],
        key=lambda row: float(row["judge_rate_decline"]),
    )
    drift = np.asarray([float(row["judge_rate_decline"]) for row in points])
    bias_pp = 100.0 * np.asarray([float(row["bias"]) for row in points])
    coverage = np.asarray([float(row["coverage"]) for row in points])
    mcse = np.asarray([float(row["coverage_mcse"]) for row in points])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(100.0 * drift, bias_pp, color="#8e44ad", marker="o", linewidth=2)
    axes[0].axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_xlabel(
        "Validation-to-production decline in judge sensitivity\nand specificity (percentage points)"
    )
    axes[0].set_ylabel("Bias in estimated production success rate\n(percentage points)")
    axes[0].grid(alpha=0.2)

    axes[1].errorbar(100.0 * drift, coverage, yerr=mcse, color="#8e44ad", marker="o", linewidth=2, capsize=3)
    axes[1].axhline(NOMINAL_COVERAGE, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set_xlabel(
        "Validation-to-production decline in judge sensitivity\nand specificity (percentage points)"
    )
    axes[1].set_ylabel("Empirical 95% coverage")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Study B: even a full bootstrap cannot repair calibration drift", y=1.03)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def assert_numerical_identities() -> None:
    """Small algebra gates; these are not a general test suite."""
    theta, tpr, tnr = 0.80, 0.875, 0.875
    j = tpr + tnr - 1.0
    q = (1.0 - tnr) + j * theta
    recovered, recovered_j = rg_point(q, tpr, tnr)
    assert np.isclose(recovered, theta)
    assert np.isclose(recovered_j, j)

    target_component = q * (1.0 - q) / (300 * j**2)
    validation_component = (
        theta**2 * tpr * (1.0 - tpr) / 250
        + (1.0 - theta) ** 2 * tnr * (1.0 - tnr) / 250
    ) / j**2
    assert target_component > 0.0
    assert np.isclose(target_component + validation_component - validation_component, target_component)

    # With a perfect judge, only target-rate sampling uncertainty remains.
    perfect = Scenario("check", "perfect", theta, 1.0, 1.0, 1.0, 1.0, 300, 100, 100)
    q_perfect = theta
    variance = q_perfect * (1.0 - q_perfect) / perfect.target_size
    assert variance > 0.0
    assert np.isclose(rg_point(q_perfect, 1.0, 1.0)[0], theta)


def study_a_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for target_size in (300, 2000):
        for validation_size in (100, 200, 500):
            scenarios.append(
                Scenario(
                    study="A",
                    name=f"m{target_size}_n{validation_size}",
                    theta=0.80,
                    validation_tpr=0.875,
                    validation_tnr=0.875,
                    target_tpr=0.875,
                    target_tnr=0.875,
                    target_size=target_size,
                    n_pass=validation_size // 2,
                    n_fail=validation_size // 2,
                )
            )
    return scenarios


def study_b_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for drift in (0.0, 0.025, 0.05, 0.10):
        scenarios.append(
            Scenario(
                study="B",
                name=f"judge_rates_down_{drift:.3f}",
                theta=0.80,
                validation_tpr=0.875,
                validation_tnr=0.875,
                target_tpr=0.875 - drift,
                target_tnr=0.875 - drift,
                target_size=2000,
                n_pass=100,
                n_fail=100,
            )
        )
    return scenarios


def run_configuration(name: str, replications: int, bootstrap_draws: int, seed: int = 20260810) -> Path:
    """Run both studies and write one self-contained result bundle."""
    assert_numerical_identities()
    output = ROOT / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    root_seed = np.random.SeedSequence(seed)
    scenario_seeds = root_seed.spawn(len(study_a_scenarios()) + len(study_b_scenarios()))
    rows: list[dict[str, float | int | str]] = []

    for index, scenario in enumerate(study_a_scenarios()):
        rows.extend(
            run_scenario(
                scenario,
                replications,
                bootstrap_draws,
                include_textbook_plus_q=(scenario.target_size == 300 and scenario.validation_size == 500),
                seed_sequence=scenario_seeds[index],
            )
        )
    for index, scenario in enumerate(study_b_scenarios(), start=len(study_a_scenarios())):
        # Study B isolates transport bias; only the complete bootstrap is meaningful here.
        scenario_rows = run_scenario(
            scenario,
            replications,
            bootstrap_draws,
            include_textbook_plus_q=False,
            seed_sequence=scenario_seeds[index],
        )
        rows.extend(row for row in scenario_rows if row["method"] == "full_fixed_stratum_bootstrap")

    summary = summarize(rows)
    write_csv(output / "raw_intervals.csv", rows)
    write_csv(output / "summary.csv", summary)
    plot_study_a(summary, output / "study_a_coverage_width")
    plot_study_b(summary, output / "study_b_drift")
    metadata = {
        "configuration": name,
        "replications": replications,
        "bootstrap_draws": bootstrap_draws,
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "scenarios": [asdict(scenario) for scenario in study_a_scenarios() + study_b_scenarios()],
    }
    with (output / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return output


def run_smoke() -> Path:
    """Fast check of all pathways before a report-scale run."""
    return run_configuration("smoke", replications=120, bootstrap_draws=250)


def run_final() -> Path:
    """Readable report-scale setting: about 0.4 percentage-point coverage MCSE."""
    return run_configuration("final", replications=3000, bootstrap_draws=1000)


if __name__ == "__main__":
    print(run_smoke())
