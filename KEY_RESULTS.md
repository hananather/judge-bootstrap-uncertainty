# Key results

## Bottom line

A bootstrap is only as complete as its resampling design. If production cases
are a random sample from an ongoing population, holding their observed judge
pass rate fixed omits real sampling uncertainty. Separately, even a complete
bootstrap cannot correct evaluator error rates that no longer transport from
validation to production.

The original procedure is still meaningful for a narrower fixed-batch
question. The problem is interpreting a calibration-conditional interval as a
population interval.

## Result 1: resample random production cases

The first experiment uses an 80% true success rate and 200 human-labelled
validation cases. A valid 95% interval should contain the true rate in about 95
of 100 repeated studies.

| Production uncertainty relative to validation uncertainty | Production rate held fixed | Production rate also resampled |
|---|---:|---:|
| 10% as large | 93.2% | 94.4% |
| Equal | 82.1% | 94.4% |
| Twice as large | 72.8% | 93.9% |

When the omitted source of uncertainty became material, the incomplete
bootstrap produced intervals that were much too confident. Adding production
resampling alone brought coverage close to its intended 95% level.

![Target resampling restores coverage](results/journal_final/reader_target_resampling.png)

## Result 2: weak denominators must be reported

| Validation labels | True $J$ | Weak draws discarded | Studies returning no interval | Coverage among reports | Report-and-cover probability |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.20 | 12.0% | 13.4% | 93.7% | 81.1% |
| 100 | 0.20 | 6.3% | 2.9% | 94.8% | 92.1% |

Conditional coverage alone hides how often the estimator cannot return a
regular ratio interval. A tested weak-draw completion widened the intervals,
but it is an operational comparison rather than a validated solution to weak
identification.

![Weak denominators expose discarded draws and nonreporting](results/component_ablation/discarding_and_boundary_rules.png)

## Result 3: a bootstrap cannot repair calibration drift

The second experiment keeps the true success rate at 80% and changes the
evaluator's production sensitivity, specificity, or both by five percentage
points after validation.

| Change after validation | Bias in corrected rate | Coverage of nominal 95% interval |
|---|---:|---:|
| No change | approximately 0.0 points | 93.8% |
| Sensitivity declines by 5 points | -5.2 points | 70.5% |
| Specificity declines by 5 points | +1.5 points | 94.0% |
| Both decline by 5 points | -3.9 points | 80.0% |

Sensitivity drift mattered more here because 80% of cases truly passed. In a
low-success population, specificity drift could be more consequential.

![Calibration drift creates bias that resampling cannot remove](results/journal_final/reader_calibration_drift.png)

## Practical decisions

- Resample production cases when the estimand concerns an ongoing or future
  production population.
- Holding a fully observed batch fixed can be appropriate when that batch is
  itself the target.
- Use fresh human labels or a defensible bridge study when evaluator accuracy
  may have changed.
- Do not interpret additional bootstrap draws as protection against systematic
  evaluator error, sample-selection bias, or calibration drift.
- Report discarded-draw and nonreporting rates alongside conditional coverage.
- Treat a weak evaluator as an identification problem, not merely a numerical
  cleanup problem.

## Evidence record

- `journal_summary.csv` contains 252 full-precision summaries.
- Study A used 5,000 outer replications per cell.
- Study B used 3,000 outer replications per cell.
- Every interval used 1,000 bootstrap draws.
- A six-cell stability check with 5,000 bootstrap draws changed coverage by at
  most 0.32 percentage points.
- No weak-denominator events occurred in the regular Study A grid. The separate
  20,000-replication stress study deliberately activated them.

See [`TECHNICAL_RESULTS.md`](TECHNICAL_RESULTS.md) for assumptions, formulas,
the complete parameter grid, Monte Carlo checks, and technical figures. See
[`BOOTSTRAP_COMPONENT_STUDY.md`](BOOTSTRAP_COMPONENT_STUDY.md) for the
component-by-component interpretation.
