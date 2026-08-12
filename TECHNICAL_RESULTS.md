# Technical evidence record for the judge-bootstrap studies

For a plain-language explanation of the setup, terminology, figures, and
practical conclusions, start with [`README.md`](README.md). This file preserves
the full design and verification record for technical review.

## Purpose

This study turns the earlier single-cell diagnostic into a compact evidence
design.  It still concerns a binary automated judge and an authoritative binary
outcome.  It does not claim to model all LLM evaluation, human disagreement, or
production drift.

The target is the production-population success probability: target cases are
sampled at random, not treated as one fixed finite batch.  Validation has fixed,
outcome-balanced authoritative strata.  The
stable-calibration study assumes the judge's sensitivity and specificity carry
from validation to target; the drift study deliberately violates that premise.

The figures are organized around three decisions:

1. If the target cases represent a random production population, resample the
   target cases as well as the validation strata. Holding the target rate fixed
   is appropriate only when the estimand is the realized, fully scored batch.
2. If sensitivity or specificity may have changed since validation, obtain
   current authoritative labels. More bootstrap draws cannot repair stale
   evaluator performance.
3. If the validation budget is chosen for estimator precision, allocate labels
   according to the target success rate and evaluator error rates rather than
   defaulting automatically to a 50/50 split.

## Study A: the variance-ratio design

For each cell, define

\[
\rho=\frac{V_{\mathrm{target}}}{V_{\mathrm{validation}}}
=
\frac{q(1-q)/m}
{\theta^2a(1-a)/n_1+(1-\theta)^2b(1-b)/n_0}.
\]

The script solves for the integer target-sample size $m$ for requested
values of $\rho$.  This has a practical advantage: the experiment directly
varies the amount of uncertainty that the textbook interval leaves out.

The design crosses:

- true success rate $\theta\in\{0.2,0.5,0.8\}$;
- balanced validation total $N\in\{200,800\}$;
- $a=b=0.875$; and
- $\rho\in\{0.05,0.10,0.20,0.40,0.75,1,1.5,2\}$.

It adds an asymmetric high-success regime,
$\theta=0.8,a=0.95,b=0.8,N=400$.  There are 56 cells.  The target-rate
resampling ablation is restricted to $\rho$ near 0.1, 1, and 2 in the
high-success, $N=200$ symmetric regime.

The first-order benchmark for the validation-only standard error is

\[
\frac{\operatorname{SE}_{\mathrm{textbook}}}
{\operatorname{SD}(\widehat\theta)}
\approx(1+\rho)^{-1/2}.
\]

It is a derived variance law, not a fitted curve.  It predicts that an interval
may become precise-looking while omitting an increasing share of uncertainty.

## Study B: directional calibration drift

Study B fixes $\theta=0.8$, $a_v=b_v=0.875$, $m=2000$, and
$n_1=n_0=100$, then independently lowers target sensitivity and specificity
by $d_a,d_b\in\{0,0.0125,\ldots,0.10\}$.  The 81 cells distinguish
sensitivity-only, specificity-only, and common-mode drift.

At first order, the calibration-transport bias is

\[
\operatorname{Bias}(\widehat\theta)
=
\frac{-\theta d_a+(1-\theta)d_b}{a_v+b_v-1}.
\]

For a high-success target, sensitivity loss produces larger negative bias than
an equally sized specificity loss produces positive bias.  A complete bootstrap
still describes the stipulated sampling mechanism; it cannot make an outdated
calibration relationship true.

## Outputs and interpretation

`journal_summary.csv` reports full-precision summaries for every cell and
method: coverage counts, coverage with Wilson Monte Carlo bounds, bias and its
Monte Carlo interval, RMSE, empirical standard deviation, interval width,
reported-standard-error calibration, and weak-denominator/discard diagnostics.
No millions-row intermediate CSV is written.

The full results support four clear conclusions.

- **Omitted target-sample variation has a predictable cost.** First-order
  coverage falls below 90% at $\rho\approx0.42$. At $\rho=1$, the target and
  validation components contribute equal first-order variance, but the
  validation-only interval omits the target half: mean coverage was 82.6% and
  its reported standard error was only 0.72 times the empirical sampling
  standard deviation. Mean coverage fell from 93.6% at $\rho=0.05$ to 73.5% at
  $\rho=2$.  Across the 56 cells, the first-order coverage law was within 0.74
  percentage points on average of the empirical result.  The full bootstrap
  covered between 93.6% and 95.2%; its reported standard error was between 0.97
  and 1.05 times the empirical sampling standard deviation.
- **Target resampling isolates the main correction.** In the high-success,
  200-label design, coverage at $\rho=1$ increased from 82.1% to 94.4% when
  target cases were resampled. The complete fixed-strata bootstrap covered
  94.2%. At $\rho=2$, the corresponding values were 72.8%, 93.9%, and 94.0%.
- **Label allocation depends on the estimand.** Under the displayed binary
  model and a fixed validation budget, the first-order precision-optimal
  Pass:Fail allocation is 1:4 at a 20% success rate, 1:1 at 50%, and 4:1 at
  80% when sensitivity and specificity are equal. This is a precision rule,
  not a universal diagnostic-labeling rule.
- **The direction of evaluator drift matters.** Starting from sensitivity and
  specificity of 0.875, a five-percentage-point sensitivity decline produced
  about 5.2 points of downward bias and 70.5% coverage.  The same decline in
  specificity produced about 1.5 points of upward bias and 94.0% coverage.
  When both declined by five points, bias was about -3.9 points and coverage
  was 80.0%.  This asymmetry follows from the high target success rate
  ($\theta=0.8$): sensitivity drift receives weight $\theta$, while specificity
  drift receives weight $1-\theta$.

These are controlled binary results, not universal guarantees.  In particular,
the no-drift cell covered 93.8%, so the study does not claim exact nominal
coverage in finite samples.

Every empirical point, error bar, and heatmap tile is an unsmoothed simulation
estimate.  The only continuous curves and contours are derived mathematical
references: the variance-omission law, the first-order drift approximation, and
the zero first-order transport-bias line for the specified binary drift model.
Small finite-sample ratio bias can remain on this line.  No moving
average, LOESS fit, or post-hoc smoother is used.

## Verification

The full run completed on 2026-08-10. Study A used 5,000 outer replications in
each of 56 cells; Study B used 3,000 outer replications in each of 81 cells.
Every bootstrap interval used 1,000 bootstrap draws. The run produced all 252 expected
summary rows and no weak-denominator or discarded-draw events. A separate
six-cell check with 5,000 bootstrap draws changed coverage by at most 0.32
percentage points and mean interval width by less than 0.001.

### Reusable caption: Study A

**Figure.** What changes when a bootstrap holds a random target judge rate
fixed. Panels A and B show the coverage and standard-error laws over the ratio
of omitted target variance to included validation variance. Seven designs are
shown at each ratio; coverage bars are 95% Wilson intervals and standard-error
bars are outer-bootstrap intervals. Panel C isolates the target-resampling
correction in the high-success, 200-label design. Panel D gives the
first-order precision-optimal allocation of Pass and Fail validation labels;
balanced labels may still be preferred for evaluator diagnosis. Target cases
are sampled from a production population and sensitivity and specificity are
stable between validation and production. Curves and allocation shares are
derived references, not fitted smoothers.

### Reusable caption: Study B

**Figure.** Bias and coverage of a fixed-stratum bootstrap when target
sensitivity and specificity depart from their validation values. The change is
a stylized sensitivity analysis. It shows that resampling can quantify the
sampling uncertainty represented in its design but cannot remove transport bias
from changed evaluator behaviour. Panel A shows the direction and magnitude of
bias. Panel B uses decision bands for empirical coverage and overlays
first-order 50%, 80%, and 90% contours. Panel C compares sensitivity-only,
specificity-only, and joint declines; solid points and lines are simulations,
while dashed curves are first-order references. Wilson intervals are shown and
are mostly smaller than the markers.

## Caveats

These are controlled binary calculations.  They do not establish transport for
a specific LLM judge, cover multi-class labels, account for clusters or
reviewer disagreement, or compare outcome-balanced calibration with a current
probability-labelled sample.  Weak denominators are surfaced rather than
silently discarded; profile likelihood remains the appropriate next comparison
for boundary cases.
