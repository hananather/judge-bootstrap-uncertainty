"""Component-by-component audit of the Chapter 5 judge bootstrap.

The script changes one implementation choice at a time:

0. reproduce the textbook procedure;
1. resample the target judge-pass rate;
2. match a fixed outcome-stratified validation design;
3. replace discarded weak-denominator draws by Mohammed's proposed boundary
   rule; and
4. take quantiles before constraining the reported interval endpoints.

This is a focused simulation script, not a package.  It writes aggregate
operating characteristics only.  The report-scale run uses 20,000 outer Monte
Carlo studies per cell; the inner bootstrap has 1,000 draws.

``make_figures.py`` draws the figure from the saved CSV file.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import csv
import json
import os
import time
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTER_REPLICATIONS = 20_000
BOOTSTRAP_DRAWS = 1_000
BATCH_SIZE = 200
BASE_SEED = 20260812
Z_975 = 1.959963984540054

METHODS = (
    "textbook",
    "plus_target",
    "plus_fixed_strata",
    "plus_boundary_rule",
    "plus_endpoint_quantiles",
)

@dataclass(frozen=True)
class Cell:
    validation_total: int
    informedness: float
    theta: float = 0.80
    target_size: int = 2_000

    @property
    def n_pass(self) -> int:
        return self.validation_total // 2

    @property
    def n_fail(self) -> int:
        return self.validation_total // 2

    @property
    def tpr(self) -> float:
        return 0.5 * (1.0 + self.informedness)

    @property
    def tnr(self) -> float:
        return self.tpr

    @property
    def judge_pass_rate(self) -> float:
        return self.theta * self.tpr + (1.0 - self.theta) * (1.0 - self.tnr)

    @property
    def key(self) -> str:
        return f"N={self.validation_total}|J={self.informedness:.2f}"


@dataclass
class Accumulator:
    trials: int = 0
    covered: int = 0
    finite_widths: int = 0
    width_sum: float = 0.0
    width_sq_sum: float = 0.0
    nonfinite_intervals: int = 0

    def add(self, lower: np.ndarray, upper: np.ndarray, theta: float) -> None:
        finite = np.isfinite(lower) & np.isfinite(upper)
        width = upper[finite] - lower[finite]
        self.trials += lower.size
        self.covered += int(np.sum(finite & (lower <= theta) & (theta <= upper)))
        self.finite_widths += int(width.size)
        self.width_sum += float(np.sum(width))
        self.width_sq_sum += float(np.sum(width**2))
        self.nonfinite_intervals += int(np.sum(~finite))


def stable_seed(key: str) -> int:
    digest = sha256(f"{BASE_SEED}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def wilson(successes: int, trials: int) -> tuple[float, float]:
    p = successes / trials
    denominator = 1.0 + Z_975**2 / trials
    center = (p + Z_975**2 / (2.0 * trials)) / denominator
    radius = Z_975 * np.sqrt(
        p * (1.0 - p) / trials + Z_975**2 / (4.0 * trials**2)
    ) / denominator
    return float(center - radius), float(center + radius)


def row_quantiles(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(valid, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        quantiles = np.nanquantile(masked, (0.025, 0.975), axis=1)
    return quantiles[0], quantiles[1]


def clipped_draw_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=valid,
    )
    return row_quantiles(np.clip(raw, 0.0, 1.0), valid)


def boundary_completed_draws(
    numerator: np.ndarray,
    denominator: np.ndarray,
    clip_each_draw: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain weak draws using one explicit, testable completion rule.

    A negative denominator still defines a raw ratio, so it is retained.
    When the denominator is exactly zero, a positive numerator is assigned to
    the upper boundary, a negative numerator to the lower boundary, and 0/0 to
    the midpoint.  This is an operational comparator, not calibrated weak-J
    inference or a uniquely implied mathematical rule.
    """
    nonzero = denominator != 0.0
    completed = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=nonzero,
    )
    zero_completion = np.where(
        numerator > 0.0, 1.0, np.where(numerator < 0.0, 0.0, 0.5)
    )
    completed = np.where(nonzero, completed, zero_completion)
    if clip_each_draw:
        completed = np.clip(completed, 0.0, 1.0)
    lower, upper = np.quantile(completed, (0.025, 0.975), axis=1)
    if not clip_each_draw:
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
    return lower, upper


def simulate_cell(cell: Cell) -> dict[str, object]:
    rng = np.random.default_rng(stable_seed(cell.key))
    accumulators = {method: Accumulator() for method in METHODS}
    pooled_missing_total = 0
    pooled_weak_total = 0
    fixed_weak_total = 0
    inner_total = 0
    pooled_missing_attempted_total = 0
    pooled_weak_attempted_total = 0
    fixed_weak_attempted_total = 0
    attempted_inner_total = 0
    observed_weak_total = 0

    for start in range(0, OUTER_REPLICATIONS, BATCH_SIZE):
        size = min(BATCH_SIZE, OUTER_REPLICATIONS - start)
        q_hat = rng.binomial(cell.target_size, cell.judge_pass_rate, size=size) / cell.target_size
        tpr_hat = rng.binomial(cell.n_pass, cell.tpr, size=size) / cell.n_pass
        tnr_hat = rng.binomial(cell.n_fail, cell.tnr, size=size) / cell.n_fail
        observed_valid = tpr_hat + tnr_hat - 1.0 > 0.0
        observed_weak_total += int(np.sum(~observed_valid))

        q_star = rng.binomial(
            cell.target_size,
            q_hat[:, None],
            size=(size, BOOTSTRAP_DRAWS),
        ) / cell.target_size

        # Exact pooled-pair nonparametric bootstrap for an observed 50/50
        # validation set, represented hierarchically for vectorized sampling.
        human_passes = rng.binomial(
            cell.validation_total, 0.5, size=(size, BOOTSTRAP_DRAWS)
        )
        human_fails = cell.validation_total - human_passes
        pooled_tpr_count = rng.binomial(human_passes, tpr_hat[:, None])
        pooled_tnr_count = rng.binomial(human_fails, tnr_hat[:, None])
        pooled_classes = (human_passes > 0) & (human_fails > 0)
        pooled_tpr = np.divide(
            pooled_tpr_count,
            human_passes,
            out=np.zeros_like(pooled_tpr_count, dtype=float),
            where=pooled_classes,
        )
        pooled_tnr = np.divide(
            pooled_tnr_count,
            human_fails,
            out=np.zeros_like(pooled_tnr_count, dtype=float),
            where=pooled_classes,
        )
        pooled_j = pooled_tpr + pooled_tnr - 1.0
        pooled_valid = pooled_classes & (pooled_j > 0.0)

        fixed_tpr = rng.binomial(
            cell.n_pass, tpr_hat[:, None], size=(size, BOOTSTRAP_DRAWS)
        ) / cell.n_pass
        fixed_tnr = rng.binomial(
            cell.n_fail, tnr_hat[:, None], size=(size, BOOTSTRAP_DRAWS)
        ) / cell.n_fail
        fixed_j = fixed_tpr + fixed_tnr - 1.0
        fixed_valid = fixed_j > 0.0

        pooled_missing_total += int(np.sum(~pooled_classes))
        pooled_weak_total += int(np.sum(pooled_classes & (pooled_j <= 0.0)))
        fixed_weak_total += int(np.sum(fixed_j <= 0.0))
        inner_total += size * BOOTSTRAP_DRAWS
        attempted_rows = observed_valid[:, None]
        pooled_missing_attempted_total += int(
            np.sum(attempted_rows & ~pooled_classes)
        )
        pooled_weak_attempted_total += int(
            np.sum(attempted_rows & pooled_classes & (pooled_j <= 0.0))
        )
        fixed_weak_attempted_total += int(
            np.sum(attempted_rows & (fixed_j <= 0.0))
        )
        attempted_inner_total += int(np.sum(observed_valid)) * BOOTSTRAP_DRAWS

        textbook_lower, textbook_upper = clipped_draw_interval(
            q_hat[:, None] + pooled_tnr - 1.0, pooled_j, pooled_valid
        )
        textbook_lower[~observed_valid] = np.nan
        textbook_upper[~observed_valid] = np.nan
        accumulators["textbook"].add(textbook_lower, textbook_upper, cell.theta)

        target_lower, target_upper = clipped_draw_interval(
            q_star + pooled_tnr - 1.0, pooled_j, pooled_valid
        )
        target_lower[~observed_valid] = np.nan
        target_upper[~observed_valid] = np.nan
        accumulators["plus_target"].add(target_lower, target_upper, cell.theta)

        stratum_numerator = q_star + fixed_tnr - 1.0
        strata_lower, strata_upper = clipped_draw_interval(
            stratum_numerator, fixed_j, fixed_valid
        )
        strata_lower[~observed_valid] = np.nan
        strata_upper[~observed_valid] = np.nan
        accumulators["plus_fixed_strata"].add(strata_lower, strata_upper, cell.theta)

        boundary_lower, boundary_upper = boundary_completed_draws(
            stratum_numerator, fixed_j, clip_each_draw=True
        )
        boundary_lower[~observed_valid] = np.nan
        boundary_upper[~observed_valid] = np.nan
        accumulators["plus_boundary_rule"].add(
            boundary_lower, boundary_upper, cell.theta
        )

        endpoint_lower, endpoint_upper = boundary_completed_draws(
            stratum_numerator, fixed_j, clip_each_draw=False
        )
        endpoint_lower[~observed_valid] = np.nan
        endpoint_upper[~observed_valid] = np.nan
        accumulators["plus_endpoint_quantiles"].add(
            endpoint_lower, endpoint_upper, cell.theta
        )

    rows = []
    for method, acc in accumulators.items():
        reportable = acc.finite_widths
        conditional_coverage = acc.covered / reportable if reportable else np.nan
        if reportable:
            coverage_low, coverage_high = wilson(acc.covered, reportable)
        else:
            coverage_low, coverage_high = np.nan, np.nan
        operational_low, operational_high = wilson(acc.covered, acc.trials)
        mean_width = acc.width_sum / acc.finite_widths if acc.finite_widths else np.nan
        rows.append({
            "validation_total_N": cell.validation_total,
            "n_pass": cell.n_pass,
            "n_fail": cell.n_fail,
            "theta": cell.theta,
            "target_size_m": cell.target_size,
            "tpr": cell.tpr,
            "tnr": cell.tnr,
            "J": cell.informedness,
            "method": method,
            "outer_replications": acc.trials,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "report_rate": reportable / acc.trials,
            "conditional_coverage": conditional_coverage,
            "coverage_wilson_low": coverage_low,
            "coverage_wilson_high": coverage_high,
            "operational_coverage": acc.covered / acc.trials,
            "operational_coverage_wilson_low": operational_low,
            "operational_coverage_wilson_high": operational_high,
            "mean_width": mean_width,
            "nonfinite_interval_rate": acc.nonfinite_intervals / acc.trials,
            "observed_nonpositive_J_rate": observed_weak_total / OUTER_REPLICATIONS,
            "pooled_missing_class_draw_rate": pooled_missing_total / inner_total,
            "pooled_nonpositive_J_draw_rate": pooled_weak_total / inner_total,
            "fixed_nonpositive_J_draw_rate": fixed_weak_total / inner_total,
            "pooled_missing_class_draw_rate_when_attempted": (
                pooled_missing_attempted_total / attempted_inner_total
                if attempted_inner_total else np.nan
            ),
            "pooled_nonpositive_J_draw_rate_when_attempted": (
                pooled_weak_attempted_total / attempted_inner_total
                if attempted_inner_total else np.nan
            ),
            "fixed_nonpositive_J_draw_rate_when_attempted": (
                fixed_weak_attempted_total / attempted_inner_total
                if attempted_inner_total else np.nan
            ),
        })
    return {"key": cell.key, "rows": rows}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_final() -> Path:
    output = ROOT / "results" / "component_ablation"
    output.mkdir(parents=True, exist_ok=True)
    cells = [Cell(validation_total=n, informedness=j) for n in (40, 100) for j in (0.20, 0.30, 0.40, 0.75)]
    started = time.perf_counter()
    results = []
    workers = min(4, len(cells), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(simulate_cell, cell): cell for cell in cells}
        for future in as_completed(futures):
            results.append(future.result())
    rows = [row for result in sorted(results, key=lambda item: item["key"]) for row in result["rows"]]
    write_csv(output / "component_summary.csv", rows)
    metadata = {
        "outer_replications_per_cell": OUTER_REPLICATIONS,
        "bootstrap_draws_per_outer_study": BOOTSTRAP_DRAWS,
        "cells": [cell.key for cell in cells],
        "workers": workers,
        "seed": BASE_SEED,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


if __name__ == "__main__":
    print(run_final())
