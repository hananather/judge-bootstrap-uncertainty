# Method note: a binary judge, a production rate, and bootstrap coverage

## Question

Suppose an automated evaluator produces $Z\in\{0,1\}$, while an
authoritative review would produce $Y\in\{0,1\}$.  We want the current
production success probability

\[
\theta=\Pr(Y=1),
\]

not merely the fraction of cases passed by the automated evaluator.

Let

\[
a=\Pr(Z=1\mid Y=1),\qquad b=\Pr(Z=0\mid Y=0),\qquad J=a+b-1.
\]

If $q=\Pr(Z=1)$, then

\[
q=(1-b)+J\theta,\qquad
\theta=\frac{q+b-1}{J}.
\]

The simulation uses the plug-in Rogan--Gladen estimator

\[
\widehat\theta=\frac{\widehat q+\widehat b-1}{\widehat a+\widehat b-1}.
\]

## Study A: how much variance is omitted?

Study A varies the share of first-order variance contributed by the random
production sample. Define

\[
\rho=
\frac{q(1-q)/m}
{\theta^2a(1-a)/n_1+(1-\theta)^2b(1-b)/n_0}.
\]

The script solves for the integer production-sample size $m$ at
$\rho\in\{0.05,0.10,0.20,0.40,0.75,1,1.5,2\}$. The main design crosses
$\theta\in\{0.2,0.5,0.8\}$ with balanced validation totals
$N\in\{200,800\}$ and $a=b=0.875$. An additional high-success design uses
$\theta=0.8$, $a=0.95$, $b=0.8$, and $N=400$. This produces 56 cells.

Within an outer replication,

\[
k\sim\operatorname{Binomial}(m,q),\quad
n_{11}\sim\operatorname{Binomial}(n_1,a),\quad
n_{00}\sim\operatorname{Binomial}(n_0,b).
\]

This treats $\theta$ as a fixed production-population success probability and
the target cases as an independent random sample.  The validation class totals
are fixed.  It is therefore not a finite-population inference exercise.

The complete delta variance is

\[
\operatorname{Var}(\widehat\theta)\approx
\frac{1}{J^2}\left[
\frac{q(1-q)}{m}+
\theta^2\frac{a(1-a)}{n_1}+
(1-\theta)^2\frac{b(1-b)}{n_0}
\right].
\]

The first term is target-sampling uncertainty.  The textbook bootstrap holds
$\widehat q$ fixed and cannot represent that term.  The complete bootstrap
draws $q^*$, $a^*$, and $b^*$ from their fitted binomial models; it keeps
the two validation strata separate.

Weak bootstrap denominators $J^*\leq0$ are recorded. They are not deleted.
For the deliberately regular Study A grid they should be negligible. If they
are not, the simple percentile interval is replaced here by the visible status
`weak_j` and $[0,1]$, rather than by a selectively conditioned interval.
This is a conservative status rule, not a validated weak-identification
interval; a separate weak-denominator comparison is still needed.

## Study B: calibration drift

Study B holds the validation rates at $a_v=b_v=0.875$, uses $m=2000$ and
200 balanced validation labels, and lowers target sensitivity and specificity
independently by

\[
d_a,d_b\in\{0,0.0125,\ldots,0.10\}.
\]

The bootstrap is complete for the *validation* sampling model, but it still
uses $(a_v,b_v)$ to correct a target governed by
$(a_v-d_a,b_v-d_b)$. The 81-cell grid separates sensitivity-only,
specificity-only, and joint drift. It is therefore expected to become biased
and to undercover as the validation calibration loses relevance. The drift is
stylized: it is a transparent sensitivity analysis, not a claim about how any
specific LLM judge drifts in production.

## Study C: component and weak-denominator stress test

Study C keeps $\theta=0.8$, $m=2000$, and fixed, balanced validation
strata. It crosses validation totals $N\in\{40,100\}$ with
$J\in\{0.20,0.30,0.40,0.75\}$, setting $a=b=(1+J)/2$. Each cell uses
20,000 outer replications and 1,000 bootstrap draws.

The method ladder changes target resampling, validation resampling, weak-draw
handling, and clipping placement sequentially. The weak-draw comparator keeps
raw ratios for negative $J^*$. When $J^*=0$, it assigns a positive numerator
to 1, a negative numerator to 0, and $0/0$ to 0.5. This convention is explicit
so it can be tested; it is not a uniquely implied or validated
weak-identification interval.

The study reports two quantities that should not be conflated:

- **conditional coverage**, among outer studies that return an interval; and
- **report-and-cover probability**, the probability that a study both returns
  an interval and contains the true value.

Weak inner-draw rates are reported conditional on the observed
$\widehat J>0$, because the textbook bootstrap is not attempted otherwise. The
fixed 50/50 design makes missing-class draws effectively absent, so this study
evaluates nonpositive denominators rather than missing-class handling.

## Figure captions for reuse

**Study A.** Empirical coverage and standard-error calibration for intervals
estimating a production-population success probability. Target cases are
sampled at random, while validation contains fixed, outcome-balanced
authoritative strata. The full bootstrap and delta method assume that the
judge's class-conditional error rates are stable between validation and target.
The textbook baseline holds the observed target judge-pass rate fixed. Coverage
error bars are 95% Wilson intervals for Monte Carlo coverage; the continuous
reference curve is derived from the variance decomposition, not fitted to the
simulation points.

**Study B.** Bias and coverage of the full fixed-stratum bootstrap under
stylized, independently varied declines in target sensitivity and specificity.
Negative bias means that the production success rate is underestimated. The
bootstrap correctly represents the stated sampling design, but no resampling
procedure can remove bias when validation calibration does not transport to
production. Displayed lines are either direct simulation connections or
prespecified first-order references; no moving average or fitted smoother is
used.

## Scope and limits

The demonstration does not model clustered traces, multiple classes, human
reviewer disagreement, prompt adaptation, model-version changes, or repeated
judge calls.  It does not compare the cost of outcome-balanced calibration with
current probability-sampled labels, and it does not claim that an old
calibration transports to a new population.  Those are distinct design
questions.

For weak denominators or boundary cases, profile likelihood is the appropriate
next comparison.  It is intentionally outside this first demonstration, whose
purpose is to isolate the missing target-rate variance term and the separate
problem of calibration drift.
