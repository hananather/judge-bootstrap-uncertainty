# Bootstrap uncertainty for automated evaluators

This repository reconstructs the binary evaluator bootstrap proposed in
Shankar and Husain's *Evals for AI Engineers*, then adds one statistical
component at a time. It asks three practical questions:

1. What happens when a bootstrap ignores variation in randomly sampled
   production cases?
2. What happens when weak-denominator bootstrap draws are discarded?
3. What happens when the evaluator's error rates change after validation?

Start with this README for the intuition and main findings. See
[`KEY_RESULTS.md`](KEY_RESULTS.md) for a compact numerical summary,
[`BOOTSTRAP_COMPONENT_STUDY.md`](BOOTSTRAP_COMPONENT_STUDY.md) for the
component-by-component argument,
[`METHOD.md`](METHOD.md) for the statistical setup, and
[`TECHNICAL_RESULTS.md`](TECHNICAL_RESULTS.md) for the full evidence record.

## The practical question

An automated evaluator can score many production cases quickly, but it makes
mistakes. A smaller set of human-labelled cases tells us how often it makes
those mistakes. We want to use both sources to answer two questions:

1. What proportion of production cases would pass an authoritative human
   review?
2. How uncertain is that corrected success rate?

The setup has three pieces:

- **Production cases:** the automated evaluator labels a comparatively large
  sample as Pass or Fail.
- **Validation cases:** humans provide authoritative labels for a smaller
  sample, allowing us to measure the evaluator's error rates.
- **Corrected success rate:** the automated pass rate is adjusted using the
  measured error rates.

## Six ideas needed to read the results

**Judge pass rate.** The percentage of production cases that the automated
evaluator labels Pass. This is not necessarily the human-defined success rate.

**Sensitivity.** Among cases that truly pass, the percentage the evaluator
also labels Pass. It measures how well the evaluator recognizes true passes.

**Specificity.** Among cases that truly fail, the percentage the evaluator
also labels Fail. It measures how well the evaluator recognizes true failures.

**Evaluator informedness.** The correction denominator is
$J=\text{sensitivity}+\text{specificity}-1$. A value near zero means that the
evaluator is close to chance and the correction becomes unstable.

**Corrected success rate.** Let $\widehat q$ be the judge pass rate,
$\widehat a$ sensitivity, and $\widehat b$ specificity. The correction is

$$
\widehat\theta
=
\frac{\widehat q+\widehat b-1}
     {\widehat a+\widehat b-1}.
$$

For example, if sensitivity and specificity are both 87.5% and the evaluator
passes 72.5% of production cases, the corrected human-defined success rate is
80%.

**Coverage.** Imagine repeating the entire study many times. A valid 95%
confidence interval should contain the true success rate in about 95 out of
100 repetitions. If it succeeds only 82 times, it is too confident: its
intervals are too narrow or incorrectly centred.

## Result 1: production-sample uncertainty must be included

The production cases are a random sample from an ongoing population. Their
observed judge pass rate therefore changes from sample to sample. A bootstrap
that holds this rate fixed pretends that one source of uncertainty does not
exist.

![Including production-sample uncertainty restores interval coverage](results/journal_final/reader_target_resampling.png)

The figure uses one controlled setting: an 80% true success rate and 200
human-labelled validation cases.

- When production-sample uncertainty was small, both procedures worked about
  equally well.
- When production and validation contributed equal uncertainty, holding the
  production rate fixed gave only **82.1% coverage**. Adding production
  resampling alone gave **94.4% coverage**.
- When production uncertainty was twice as large as validation uncertainty,
  the incomplete procedure fell to **72.8% coverage**, while the
  production-resampled bootstrap remained near **93.9%**.

The reason is simple: the incomplete procedure acts as if the observed
production sample were the whole population. That assumption makes its
intervals look more precise than they are.

**Practical implication.** If the goal is inference about an ongoing or future
production population, resample the production cases as well as the validation
data. If the target is one fixed, fully scored batch, holding that batch fixed
may instead be appropriate. The estimand determines the resampling design.

## Result 2: weak evaluators create nonreporting, not just wide intervals

The original procedure removes bootstrap draws whose estimated denominator is
nonpositive. A separate 20,000-replication stress study makes those events
visible.

![Weak evaluators expose discarded draws and nonreporting](results/component_ablation/discarding_and_boundary_rules.png)

Two different events matter. With 40 validation labels and evaluator
informedness $J=0.20$, 12.0% of *inner bootstrap draws* were discarded among
studies where an interval was attempted. Separately, 13.4% of *repeated outer
studies* had a nonpositive observed denominator and returned no ratio interval.
The interval covered 93.7% of the time conditional on reporting, but the
probability of both reporting and covering was only 81.1%.

One prespecified rule retained negative-denominator draws and assigned an
explicit value when the denominator was exactly zero. It widened the interval
and raised conditional coverage, but it did not solve the observed
nonpositive-denominator problem. The lesson is to disclose weak-draw and
nonreporting rates and avoid treating a boundary convention as a validated
weak-identification method. The exact rule is documented in
[`BOOTSTRAP_COMPONENT_STUDY.md`](BOOTSTRAP_COMPONENT_STUDY.md).

## Result 3: resampling cannot repair stale evaluator accuracy

The correction also assumes that sensitivity and specificity measured during
validation still apply in production. The second experiment deliberately
breaks that assumption.

![Bias and coverage after evaluator accuracy changes](results/journal_final/reader_calibration_drift.png)

The setting has an 80% true success rate. The figure asks what happens when the
evaluator's ability to recognize true passes, true failures, or both declines
by five percentage points after validation.

- With no change after validation, the corrected estimate was essentially
  unbiased and the interval contained the truth **93.8 times out of 100**.
- When the evaluator recognized five fewer true passes per 100, the corrected
  success rate was **5.2 points too low** and coverage fell to **70.5%**.
- When it recognized five fewer true failures per 100, the corrected rate was
  **1.5 points too high**. Coverage remained near 94%, but the estimate was
  still systematically shifted.
- When both recognition rates declined, the corrected rate was **3.9 points
  too low** and coverage fell to **80.0%**.

Sensitivity drift mattered more in this example because 80% of cases truly
passed. Most cases therefore relied on the evaluator's ability to recognize
true passes. In a low-success population, specificity drift could matter more.

**Practical implication.** More bootstrap draws do not solve calibration
drift. If evaluator behaviour may have changed, collect fresh human labels from
the current population or use a defensible bridge study. A bootstrap measures
the sampling variation represented in its design; it does not make stale error
rates valid again.

## What the technical figures add

The three figures above are the report-facing explanation. The denser technical
figures remain useful as supporting evidence:

- [`study_a_journal.png`](results/journal_final/study_a_journal.png) shows the
  general variance-ratio law, standard-error mechanism, target-resampling
  ablation, and a separate label-allocation calculation.
- [`study_b_journal.png`](results/journal_final/study_b_journal.png) shows the
  complete bias and coverage surfaces across all 81 drift combinations.

The label-allocation calculation answers a different question from the
bootstrap experiment. It describes how to allocate a fixed validation budget
to improve estimator precision under a specified model. It is not a universal
recommendation for diagnosing evaluator failures.

## What these simulations do not establish

These are controlled binary experiments. They do not establish that a specific
LLM evaluator is accurate or stable. They do not cover multiple classes,
clustered traces, reviewer disagreement, repeated judge calls, prompt changes,
or evaluator-version changes. They isolate three statistical lessons:

1. repeat every random component relevant to the target of inference; and
2. do not confuse resampling uncertainty with protection against systematic
   evaluator error or drift; and
3. report nonreporting and weak-draw rates rather than conditioning the result
   silently on successful ratio estimation.

## Technical record and reproduction

- [`METHOD.md`](METHOD.md) gives the model, estimand, and interval definitions.
- [`TECHNICAL_RESULTS.md`](TECHNICAL_RESULTS.md) records the full design, numerical
  findings, verification checks, and technical captions.
- [`journal_study.py`](journal_study.py) contains the expanded simulations and
  plotting code.
- [`component_ablation_study.py`](component_ablation_study.py) contains the
  20,000-replication weak-denominator study.
- [`journal_summary.csv`](results/journal_final/journal_summary.csv) contains
  the full-precision summaries used in every displayed figure.

Install the two runtime dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The scripts are deliberately small and have no command-line interface. Running
`journal_study.py` directly invokes only the smoke study. The report-scale study
is explicit:

```bash
python3 -c "import journal_study as js; print(js.run_final())"
```

The component stress study is run separately:

```bash
python3 component_ablation_study.py
```

The current reader figures were regenerated from the existing final summary;
the report-scale simulations were not rerun during the explanation redesign.
