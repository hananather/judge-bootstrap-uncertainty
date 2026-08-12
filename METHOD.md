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

## Study A: coverage when calibration is stable

The target sample has $m\in\{300,2000\}$ independently sampled production
traces.  The validation data contain fixed numbers of authoritative Pass and
Fail examples, with total size $100$, $200$, or $500$.  The operating
values are

\[
\theta=0.80,\qquad a=b=0.875,\qquad J=0.75.
\]

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

Weak bootstrap denominators $J^*\leq0$ are recorded.  They are not deleted.
For the deliberately regular Study A grid they should be negligible.  If they
are not, the simple percentile interval is replaced here by the visible status
`weak_j` and $[0,1]$, rather than by a selectively conditioned interval.

## Study B: calibration drift

Study B holds the validation rates at $a_v=b_v=0.875$, uses $m=2000$ and
200 balanced validation labels, and lowers both target rates by

\[
d\in\{0,0.025,0.05,0.10\}.
\]

The bootstrap is complete for the *validation* sampling model, but it still
uses $(a_v,b_v)$ to correct a target governed by $(a_v-d,b_v-d)$.  It is
therefore expected to become biased and to undercover as $d$ increases.  The
drift is stylized: it is a transparent sensitivity analysis, not a claim about
how any specific LLM judge drifts in production.

## Figure captions for reuse

**Study A.** Empirical coverage and mean width for intervals estimating a
production-population success probability.  Target cases are sampled at random,
while validation contains fixed, outcome-balanced authoritative strata.  The
full bootstrap and delta method assume that the judge's class-conditional error
rates are stable between validation and target.  The textbook baseline holds
the observed target judge-pass rate fixed.  Coverage error bars show plus or
minus one Monte Carlo standard error.

**Study B.** Bias and coverage of the full fixed-stratum bootstrap under a
stylized symmetric decline in target sensitivity and specificity.  Negative
bias means that the production success rate is underestimated.  The bootstrap
correctly represents the stated sampling design, but no resampling procedure
can remove bias when validation calibration does not transport to production.

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
