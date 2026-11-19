# Method and result tables

Read [`README.md`](README.md) first. It explains the setup in plain words.

This file uses three words with one meaning each:

- **Run**: one simulated repetition of an evaluation.
- **Setting**: one combination of parameters.
- **Design**: in Study 1, a setting without its variance ratio. Seven designs
  at eight ratios give 56 settings.

## Contents

1. [Model and estimator](#1-model-and-estimator)
2. [Variance of the corrected pass rate](#2-variance-of-the-corrected-pass-rate)
3. [The intervals compared](#3-the-intervals-compared)
4. [Study 1: production-sample variance](#4-study-1-production-sample-variance)
5. [Study 2: weak judge](#5-study-2-weak-judge)
6. [Study 3: drift](#6-study-3-drift)
7. [Checks](#7-checks)
8. [Smaller results](#8-smaller-results)
9. [Files, columns and names](#9-files-columns-and-names)

## 1. Model and estimator

- **Human label ($Y$) and judge label ($Z$)**: 1 for Pass and 0 for Fail, for
  each case.
- **True pass rate ($\theta$)**: $\Pr(Y=1)$ in production.
- **Sensitivity ($a$)**: $\Pr(Z=1 \mid Y=1)$, the share of human-Pass cases
  that the judge passes.
- **Specificity ($b$)**: $\Pr(Z=0 \mid Y=0)$, the share of human-Fail cases
  that the judge fails.
- **Separation ($J$)**: $a + b - 1$, which is Youden's J statistic.
- **Judge pass rate ($q$)**: $\Pr(Z=1)$ in production.
- **Production cases ($m$)**: the size of the production sample.
- **Validation cases ($n_1$, $n_0$)**: the numbers of human-Pass and
  human-Fail cases. The number of validation labels is $N = n_1 + n_0$.

The judge passes a true pass with probability $a$ and a true fail with
probability $1-b$. So the judge pass rate is

$$
q = \theta a + (1-\theta)(1-b) = (1-b) + J\theta . \qquad (1)
$$

Solve equation (1) for $\theta$ and replace each quantity with its estimate.
The result is the Rogan-Gladen estimator, the corrected pass rate:

$$
\widehat\theta = \frac{\widehat q + \widehat b - 1}{\widehat a + \widehat b - 1}. \qquad (2)
$$

Each run draws three independent counts:

$$
k \sim \mathrm{Binomial}(m, q),\qquad
n_{11} \sim \mathrm{Binomial}(n_1, a),\qquad
n_{00} \sim \mathrm{Binomial}(n_0, b),
$$

with $\widehat q = k/m$, $\widehat a = n_{11}/n_1$ and
$\widehat b = n_{00}/n_0$. The design sets the class sizes $n_1$ and $n_0$ in
advance. The target $\theta$ is the pass rate of the stream that the
production sample comes from.

## 2. Variance of the corrected pass rate

A first-order (delta-method) expansion gives three terms:

$$
\mathrm{Var}(\widehat\theta) \approx \frac{1}{J^2}\left[
\underbrace{\frac{q(1-q)}{m}}_{\text{production sample}} +
\underbrace{\theta^2\frac{a(1-a)}{n_1} + (1-\theta)^2\frac{b(1-b)}{n_0}}_{\text{validation sample}}
\right]. \qquad (3)
$$

**The variance ratio** is the first term divided by the sum of the other two:

$$
\rho = \frac{q(1-q)/m}{\theta^2 a(1-a)/n_1 + (1-\theta)^2 b(1-b)/n_0}. \qquad (4)
$$

**Predicted coverage.** A bootstrap that holds $\widehat q$ fixed reports a
standard error that is too small by the factor $(1+\rho)^{-1/2}$. For a 95%
interval, the predicted coverage is

$$
2 \Phi\left(\frac{1.96}{\sqrt{1+\rho}}\right) - 1 , \qquad (5)
$$

where $\Phi$ is the standard normal distribution function. The prediction is
94.4% at $\rho = 0.05$, 83.4% at $\rho = 1$ and 74.2% at $\rho = 2$. It falls
below 90% at $\rho = 0.42$.

**The closed-form interval** is $\widehat\theta \pm 1.96 \times \widehat{\mathrm{SE}}$.
The standard error uses all three terms of equation (3), with each quantity
replaced by its estimate. The method limits the endpoints to $[0, 1]$.

**The production sample as the target.** The production-sample variance has
two parts:

$$
q(1-q) = \underbrace{\theta a(1-a) + (1-\theta) b(1-b)}_{\text{judge errors on the cases}}
+ \underbrace{J^2\theta(1-\theta)}_{\text{which cases were drawn}} . \qquad (6)
$$

If the target is the pass rate of the scored cases themselves, the second part
drops out. The first part stays, because the judge's labels on those cases are
still random. At $\theta = 0.8$ and $a = b = 0.875$ the first part is 0.109 of
0.199, or 55%. With $\rho = 1$, the ratio for this target is 0.55, and the
predicted coverage of the validation-only bootstrap is about 88%. No study
here simulates this target.

## 3. The intervals compared

Every bootstrap uses 1,000 draws and the percentile method: the interval runs
from the 2.5% point to the 97.5% point of the bootstrap values. $J^\ast$ is the
value of $\widehat J$ in one bootstrap draw.

| Interval | Production sample | Validation sample | A draw with $J^\ast \le 0$ | Limit to $[0,1]$ | Used in |
|---|---|---|---|---|---|
| **Validation-only bootstrap** | Held fixed | Resampled as one pool of pairs | Discarded | Every draw | Studies 1 and 2 |
| Step 1: add production resampling | Resampled | One pool of pairs | Discarded | Every draw | Studies 1 and 2 |
| Step 2: resample within classes | Resampled | Resampled within the human-Pass and human-Fail classes | Discarded | Every draw | Study 2 |
| Step 3: keep the discarded draws | Resampled | Within classes | Kept | Every draw | Study 2 |
| Step 4: limit the endpoints only | Resampled | Within classes | Kept | The two endpoints | Study 2 |
| **Two-sample bootstrap** | Resampled | Within classes | The interval becomes $[0, 1]$, with a flag | The two endpoints | Studies 1 and 3 |
| Closed-form interval | In the formula | In the formula | Does not apply | The two endpoints | Study 1 |

- **The validation-only bootstrap follows the chapter and `judgy`.** The
  chapter is Chapter 5 of *Evals for AI Engineers* by Shankar and Husain. This
  bootstrap also discards a draw with no human-Pass or no human-Fail case.
- **Steps 1 to 4 change one choice at a time**, so Study 2 measures each
  change alone.
- **Step 3 uses one rule to keep a draw.** It keeps the ratio when $J^\ast$ is
  negative. When $J^\ast$ is exactly zero, it assigns 1 for a positive
  numerator, 0 for a negative numerator and 0.5 for 0/0. No study validates
  this rule as an interval for a weak judge.
- **No draw of the two-sample bootstrap had $J^\ast \le 0$** in Study 1 or
  Study 3, so its $[0, 1]$ rule did not apply.

Study 2 withholds every interval when the estimated $J$ is zero or negative.
The `judgy` library raises an error in the same case.

## 4. Study 1: production-sample variance

**Design.** Seven designs at eight variance ratios give 56 settings. For each
setting, the script solves for the production sample size $m$ that gives the
target $\rho$. Each setting has 5,000 runs.

| Design | $\theta$ | $a$ | $b$ | $N$, with $n_1 = n_0 = N/2$ |
|---|---:|---:|---:|---:|
| Low pass rate | 0.2 | 0.875 | 0.875 | 200 and 800 |
| Middle pass rate | 0.5 | 0.875 | 0.875 | 200 and 800 |
| High pass rate | 0.8 | 0.875 | 0.875 | 200 and 800 |
| High pass rate, unequal judge | 0.8 | 0.95 | 0.80 | 400 |

**Results by variance ratio.** Each coverage is the mean over the seven
designs. The last column is for the validation-only bootstrap: its reported
standard error divided by the standard deviation of the corrected pass rate
across runs. A value of 1 means that the reported standard error is the right
size.

| $\rho$ | Predicted coverage | Validation-only bootstrap | Two-sample bootstrap | Closed-form interval | Standard error ÷ standard deviation |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 94.4% | 93.6% | 94.2% | 94.6% | 1.00 |
| 0.10 | 93.8% | 93.4% | 94.6% | 94.9% | 0.97 |
| 0.20 | 92.6% | 91.9% | 94.3% | 94.6% | 0.92 |
| 0.40 | 90.2% | 89.6% | 94.5% | 95.0% | 0.86 |
| 0.75 | 86.2% | 85.3% | 94.5% | 95.1% | 0.77 |
| 1.00 | 83.4% | 82.6% | 94.4% | 95.0% | 0.72 |
| 1.50 | 78.5% | 77.9% | 94.7% | 95.2% | 0.65 |
| 2.00 | 74.2% | 73.5% | 94.6% | 95.0% | 0.59 |

- **Coverage of the validation-only bootstrap is just below the prediction
  of equation (5).** It is below in 51 of 56 settings: 0.7 points below on
  average and 2.4 points below at most (unequal judge, $\rho = 1$).
- **Coverage of the two-sample bootstrap is 93.6% to 95.2%**, 94.5% on
  average. Its
  reported standard error is 0.97 to 1.05 times the standard deviation.
- **Coverage of the closed-form interval is 93.9% to 96.4%**, 94.9% on
  average.

**One change at a time.** The README table uses the high pass rate design with
$N = 200$. In that design, Step 1 alone gives 94.4%, 94.4% and 93.9% at
$\rho$ = 0.1, 1 and 2. The two-sample bootstrap gives 94.4%, 94.2% and 94.0%.
In these three settings, production resampling alone gives the gain.

## 5. Study 2: weak judge

**Design.** $\theta = 0.8$ and $m = 2{,}000$. The validation sample has
$N = 40$ or $N = 100$ labels, half human-Pass and half human-Fail. The judge
has $a = b = (1+J)/2$ with $J$ = 0.20, 0.30, 0.40 or 0.75. Each of the eight
settings has 20,000 runs.

**Two measures of coverage.** Coverage when reported counts only the runs that
returned an interval. The last column counts all runs: a run with no interval
cannot contain the truth.

**Validation-only bootstrap, all eight settings.** Draws discarded is the
share of bootstrap draws with $J^\ast \le 0$, among runs that returned an
interval.

| $N$ | $J$ | No interval | Draws discarded | Coverage when reported | Mean width | Reported and contains the truth |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.20 | 13.4% | 12.0% | 93.7% | 79.6 points | 81.1% |
| 40 | 0.30 | 3.7% | 6.8% | 94.0% | 68.9 points | 90.5% |
| 40 | 0.40 | 0.6% | 2.9% | 93.5% | 57.2 points | 93.0% |
| 40 | 0.75 | 0.0% | 0.0% | 91.6% | 27.4 points | 91.6% |
| 100 | 0.20 | 2.9% | 6.3% | 94.8% | 68.4 points | 92.1% |
| 100 | 0.30 | 0.1% | 1.5% | 94.7% | 51.7 points | 94.5% |
| 100 | 0.40 | 0.0% | 0.2% | 93.8% | 40.4 points | 93.8% |
| 100 | 0.75 | 0.0% | 0.0% | 92.7% | 20.1 points | 92.7% |

**Each step, at $J = 0.20$.** Coverage when reported, then mean width.

| Interval | $N = 40$ | $N = 100$ |
|---|---:|---:|
| Validation-only bootstrap | 93.7%, 79.6 points | 94.8%, 68.4 points |
| Step 1: add production resampling | 93.8%, 80.0 points | 95.2%, 69.5 points |
| Step 2: resample within classes | 93.5%, 79.1 points | 94.9%, 69.0 points |
| Step 3: keep the discarded draws | 96.5%, 90.7 points | 97.2%, 81.3 points |
| Step 4: limit the endpoints only | 96.5%, 90.7 points | 97.2%, 81.3 points |

- **Step 3 widens the interval.** Coverage when reported rises above 95%, and
  the mean width grows by about 12 points. At $N = 40$, the share of all runs
  that report and contain the truth rises from 81.0% to 83.6%. The share of
  runs with no interval does not change.
- **Steps 3 and 4 give the same coverage in all eight settings.** Their mean
  widths differ by less than 0.002 points, so the place of the limit to
  $[0,1]$ has no effect here.
- **A small validation sample gives coverage below 95% with a good judge.**
  At $J = 0.75$, coverage of the validation-only bootstrap is 91.6% at
  $N = 40$ and 92.7% at $N = 100$. Step 1 gives 91.7% and 93.5%.

## 6. Study 3: drift

**Design.** $\theta = 0.8$, $m = 2{,}000$ and $n_1 = n_0 = 100$. The validation
sample has $a_v = b_v = 0.875$. In production, sensitivity is $a_v - d_a$ and
specificity is $b_v - d_b$. Each of $d_a$ and $d_b$ takes nine values from 0
to 10 points, in steps of 1.25 points, for 81 settings. Each setting has 3,000
runs and uses the two-sample bootstrap.

**Bias.** In large samples the corrected pass rate tends to

$$
\theta + \frac{-\theta d_a + (1-\theta) d_b}{a_v + b_v - 1}. \qquad (7)
$$

At $\theta = 0.8$ and $J = 0.75$, each point of lost sensitivity adds −1.07
points of bias. Each point of lost specificity adds +0.27 points. The two
cancel when $d_b = 4 d_a$.

| Fall in sensitivity | Fall in specificity | Bias from equation (7) | Measured bias | Coverage |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0 points | 0.0 points | 93.8% |
| 5 points | 0 | −5.3 points | −5.2 points | 70.5% |
| 0 | 5 points | +1.3 points | +1.5 points | 94.0% |
| 5 points | 5 points | −4.0 points | −3.9 points | 80.0% |
| 10 points | 0 | −10.7 points | −10.5 points | 27.5% |
| 0 | 10 points | +2.7 points | +2.9 points | 89.0% |
| 10 points | 10 points | −8.0 points | −7.9 points | 49.2% |

Measured bias is within 0.3 points of equation (7) in all 81 settings. The mean
interval width stays between 14.6 and 15.6 points.

## 7. Checks

- **The Monte Carlo error of a coverage value is at most ±1.8 points.** The
  95% Wilson interval is at most ±1.3 points in Study 1, ±0.6 points in
  Study 2 and ±1.8 points in Study 3.
- **An exact value agrees with Study 2.** A run has no interval when the
  judge is correct on at most $N/2$ of the $N$ validation cases. That count
  is Binomial($N$, $(1+J)/2$). At $N = 40$ and $J = 0.20$ the exact
  probability is 13.0%. The simulated 13.4% is 1.7 standard errors above it.
- **1,000 bootstrap draws are enough.** With 5,000 draws in place of 1,000,
  coverage in six settings changed by at most 0.32 points. Mean width changed
  by less than 0.1 points
  (`results/journal_final/bootstrap_draw_stability.csv`).
- **The results are reproducible.** Every setting has its own seed. If you
  run the scripts again, you get the same two summary CSV files.
- **No figure has a fitted curve.** Every plotted point is a simulation
  result. The one curve in Figure 2 is the predicted coverage of
  equation (5).

## 8. Smaller results

**A pooled resample can lose a class.** The validation-only bootstrap
resamples the validation pairs as one pool. A draw can then contain no
human-Pass case or no human-Fail case. With a 50/50 sample of $N = 100$ the
chance is $2^{-99}$, about $1.6 \times 10^{-30}$. With 2 human-Pass and 18
human-Fail cases it is $0.9^{20} + 0.1^{20}$, or 12.2%. Resampling within
classes makes the event impossible.

**The split of the validation labels changes the variance.** For a set $N$,
the first-order variance is smallest when

$$
\frac{n_1}{n_0} = \frac{\theta\sqrt{a(1-a)}}{(1-\theta)\sqrt{b(1-b)}} . \qquad (8)
$$

With $a = b$, the best share of human-Pass labels equals $\theta$: 20%, 50% or
80% at the three pass rates of Study 1. With $a = 0.95$, $b = 0.80$ and
$\theta = 0.8$ it is 69%. This result comes from equation (3). No study
simulates it.

## 9. Files, columns and names

| Study | Script | Results file | Rows |
|---|---|---|---|
| Study 1 | `journal_study.py` | `results/journal_final/journal_summary.csv` | `study` = `A` |
| Study 2 | `component_ablation_study.py` | `results/component_ablation/component_summary.csv` | All |
| Study 3 | `journal_study.py` | `results/journal_final/journal_summary.csv` | `study` = `B` |

| Interval | `method` in the CSV files |
|---|---|
| Validation-only bootstrap | `textbook_validation_only`, `textbook` |
| Step 1: add production resampling | `textbook_plus_target_resampling`, `plus_target` |
| Step 2: resample within classes | `plus_fixed_strata` |
| Step 3: keep the discarded draws | `plus_boundary_rule` |
| Step 4: limit the endpoints only | `plus_endpoint_quantiles` |
| Two-sample bootstrap | `full_fixed_stratum_bootstrap` |
| Closed-form interval | `delta_method` |

| Column in the Study 2 tables | Column in `component_summary.csv` |
|---|---|
| No interval | 1 − `report_rate` |
| Draws discarded | `pooled_nonpositive_J_draw_rate_when_attempted` |
| Coverage when reported | `conditional_coverage` |
| Mean width | `mean_width` |
| Reported and contains the truth | `operational_coverage` |

The scripts say "target" for the production sample, "TPR" and "TNR" for
sensitivity and specificity, "informedness" for $J$, and "replication" for a
run.
