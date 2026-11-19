"""Draw the README figures from the saved simulation summaries.

Run ``python3 make_figures.py``. It reads the two CSV files in ``results/`` and
writes Figures 1 to 4 to ``figures/`` as SVG, PNG and PDF. It runs no
simulation. The PDF files keep the text as text.
Figure 1 is a TikZ diagram, ``figures/fig1-setup.tex``. Its numbers come from
``figures/fig1-numbers.tex``, which this script writes. To compile it, the
script needs ``pdflatex`` with TikZ and ``pdftocairo``; without them it skips
Figure 1 and says so.

Colour roles are the same in every figure: orange marks the validation-only
bootstrap and the runs where it fails, blue marks the two-sample bootstrap.
"""

from __future__ import annotations

from math import ceil, erf, sqrt
from pathlib import Path
import csv
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures"
Z_975 = 1.959963984540054
REFERENCE_DESIGN = "high success, N=200"

INK = "#172B3A"
SECONDARY = "#45556C"
MUTED = "#62748E"
AXIS = "#CAD5E2"
GRID = "#E2E8F0"
NEUTRAL_LIGHT = "#E2E8F0"
NEUTRAL_DARK = "#62748E"
ORANGE = "#F54900"
BLUE = "#0084D1"
VIOLET = "#7F22FE"
EMERALD = "#009966"
SLATE = "#314158"

STYLE = {
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "text.color": INK,
    "axes.labelcolor": SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "legend.frameon": False,
    "legend.fontsize": 9.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "svg.hashsalt": "judge-bootstrap-uncertainty",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def heading(figure: plt.Figure, title: str, subtitle: str) -> None:
    """The title states the claim. The subtitle names the measurement."""
    height = figure.get_figheight()
    figure.text(0.025, 1.0 - 0.20 / height, title, fontsize=13.5, fontweight=500,
                ha="left", va="top")
    figure.text(0.025, 1.0 - 0.50 / height, subtitle, fontsize=10.0,
                color=SECONDARY, ha="left", va="top")


def footnote(figure: plt.Figure, text: str) -> None:
    """The conditions a reader needs when the figure leaves the README."""
    figure.text(0.025, 0.14 / figure.get_figheight(), text, fontsize=8.2,
                color=MUTED, ha="left", va="bottom", linespacing=1.4)


def save(figure: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(exist_ok=True)
    figure.savefig(OUTPUT / f"{name}.svg", metadata={"Date": None})
    figure.savefig(OUTPUT / f"{name}.png", dpi=200)
    figure.savefig(OUTPUT / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(figure)


def at_most(value: float) -> str:
    """Round up to one decimal, so that an "at most" statement stays true."""
    return f"{ceil(value * 10.0 - 1e-9) / 10.0:.1f}"


def wilson_half_widths(row: dict[str, str]) -> tuple[float, float]:
    """The lower and upper half-widths of the 95% interval for a coverage, in points."""
    return (
        100.0 * (float(row["coverage"]) - float(row["coverage_wilson_low"])),
        100.0 * (float(row["coverage_wilson_high"]) - float(row["coverage"])),
    )


def figure_production_variance(rows: list[dict[str, str]]) -> None:
    """Figure 2. Coverage against the variance ratio, for all 56 settings of Study 1."""
    study = [row for row in rows if row["study"] == "A"]
    designs = list(dict.fromkeys(row["regime"] for row in study))
    jitter = dict(zip(designs, np.geomspace(0.93, 1.07, len(designs))))
    reference = {
        float(row["rho_target"]): int(row["target_size_m"])
        for row in study if row["regime"] == REFERENCE_DESIGN
    }
    methods = (
        ("full_fixed_stratum_bootstrap", BLUE, "s",
         "Two-sample bootstrap: resamples both samples"),
        ("textbook_validation_only", ORANGE, "o",
         "Validation-only bootstrap: holds the production sample fixed"),
    )

    figure, axis = plt.subplots(figsize=(8.0, 5.7))
    figure.subplots_adjust(left=0.085, right=0.925, top=0.845, bottom=0.315)

    rho_line = np.geomspace(0.043, 2.3, 300)
    predicted = [100.0 * (2.0 * normal_cdf(Z_975 / sqrt(1.0 + rho)) - 1.0) for rho in rho_line]
    axis.plot(rho_line, predicted, color=MUTED, linewidth=1.3, zorder=2)
    axis.axhline(95.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    axis.text(1.012, 95.0, "95%\nintended", transform=blended_transform_factory(
        axis.transAxes, axis.transData), fontsize=8.6, color=SECONDARY, va="center",
        linespacing=1.2)

    means: dict[str, dict[float, float]] = {}
    for method, color, marker, _ in methods:
        selected = [row for row in study if row["method"] == method]
        x = [float(row["rho_actual"]) * jitter[row["regime"]] for row in selected]
        y = [100.0 * float(row["coverage"]) for row in selected]
        errors = np.array([wilson_half_widths(row) for row in selected]).T
        axis.errorbar(x, y, yerr=errors, fmt="none", ecolor=color, elinewidth=0.8,
                      alpha=0.45, zorder=3)
        axis.scatter(x, y, s=26, color=color, marker=marker, edgecolor="white",
                     linewidth=0.7, zorder=4)
        means[method] = {
            rho: float(np.mean([
                100.0 * float(row["coverage"]) for row in selected
                if float(row["rho_target"]) == rho
            ]))
            for rho in reference
        }

    only = means["textbook_validation_only"]
    two_sample = [100.0 * float(row["coverage"]) for row in study
                  if row["method"] == "full_fixed_stratum_bootstrap"]
    notes = (
        (f"ρ = 1: {only[1.0]:.1f}% on average\nover the {len(designs)} designs",
         (0.93, only[1.0] - 0.4), (0.33, 80.6)),
        (f"ρ = 2: {only[2.0]:.1f}% on average",
         (1.86, only[2.0] - 0.3), (0.80, 70.6)),
        (f"{min(two_sample):.1f}% to {max(two_sample):.1f}% in all {len(two_sample)} settings",
         (0.42, 95.4), (0.16, 97.4)),
    )
    for text, point, position in notes:
        axis.annotate(
            text, xy=point, xytext=position, fontsize=9.0, color=INK, va="center",
            linespacing=1.3,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8,
                        "shrinkA": 3, "shrinkB": 3},
        )

    axis.set_xscale("log")
    axis.set_xlim(0.041, 2.42)
    axis.set_ylim(68.0, 98.6)
    axis.set_yticks(range(70, 96, 5))
    axis.set_yticklabels([f"{tick}%" for tick in range(70, 96, 5)])
    axis.set_xticks(list(reference))
    axis.set_xticklabels([f"{rho:g}\n{cases:,}" for rho, cases in reference.items()],
                         linespacing=1.45)
    axis.minorticks_off()
    axis.grid(axis="x", visible=False)
    axis.set_xlabel(
        "Top row: variance ratio ρ = production-sample variance ÷ validation-sample "
        "variance (log scale).\n"
        "Bottom row: production cases at that ρ in one design: 200 validation labels, "
        "80% true pass rate, 87.5% sensitivity and specificity.",
        fontsize=8.8, loc="left", labelpad=9, linespacing=1.4,
    )

    handles = [
        Line2D([], [], color=color, marker=marker, linestyle="none", markersize=6,
               markeredgecolor="white", markeredgewidth=0.7, label=label)
        for _, color, marker, label in methods
    ]
    handles.append(Line2D([], [], color=MUTED, linewidth=1.3,
                          label="Predicted coverage when the production-sample variance is omitted"))
    axis.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0),
                handletextpad=0.6, labelspacing=0.5, borderaxespad=0.6,
                frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)

    heading(
        figure,
        "A fixed production sample makes the interval too narrow",
        "Coverage: the share of simulated runs whose 95% interval contains the true pass rate",
    )
    largest = max(max(wilson_half_widths(row)) for row in study)
    footnote(
        figure,
        f"Each point is one setting: {len(designs)} designs at each of {len(reference)} ratios, "
        "5,000 simulated runs per setting, 1,000 bootstrap draws per run.\n"
        "Points are spread sideways so that they do not overlap. Vertical bars are 95% Monte "
        f"Carlo intervals, at most ±{at_most(largest)} points.",
    )
    save(figure, "fig2-production-variance")


def figure_weak_judge(rows: list[dict[str, str]]) -> None:
    """Figure 3. What the simulated runs of Study 2 return when the judge is weak."""
    study = [row for row in rows if row["method"] == "textbook"]
    groups = (("40", (7.75, 6.75, 5.75, 4.75)), ("100", (3.0, 2.0, 1.0, 0.0)))
    colors = {"covers": NEUTRAL_LIGHT, "misses": NEUTRAL_DARK, "none": ORANGE}
    columns = (1.17, 1.50, 1.76)

    figure, axis = plt.subplots(figsize=(8.0, 5.1))
    figure.subplots_adjust(left=0.125, right=0.60, top=0.725, bottom=0.215)
    beside = blended_transform_factory(axis.transAxes, axis.transData)

    for labels, positions in groups:
        settings = sorted((row for row in study if row["validation_total_N"] == labels),
                          key=lambda row: float(row["J"]))
        axis.text(-0.20, positions[0] + 0.72, f"{labels} validation labels",
                  transform=beside, fontsize=9.5, fontweight=500, va="center")
        for position, row in zip(positions, settings):
            covers = 100.0 * float(row["operational_coverage"])
            reported = 100.0 * float(row["report_rate"])
            segments = (
                ("covers", 0.0, covers),
                ("misses", covers, reported - covers),
                ("none", reported, 100.0 - reported),
            )
            for name, start, width in segments:
                axis.barh(position, width, left=start, height=0.62, color=colors[name],
                          edgecolor="white", linewidth=1.6, zorder=3)
            axis.text(1.6, position, f"{covers:.1f}%", fontsize=9.0, va="center", zorder=4)
            axis.text(-0.025, position, f"J = {float(row['J']):.2f}",
                      transform=beside, fontsize=9.5, ha="right", va="center")
            values = (
                f"{100.0 - reported:.1f}%",
                f"{100.0 * float(row['conditional_coverage']):.1f}%",
                f"{100.0 * float(row['mean_width']):.1f}",
            )
            for column, value in zip(columns, values):
                axis.text(column, position, value, transform=beside, fontsize=9.5,
                          ha="right", va="center")

    top = groups[0][1][0] + 0.95
    headers = ("No\ninterval", "Contains truth,\nwhen reported", "Mean width,\npoints")
    for column, header in zip(columns, headers):
        axis.text(column, top, header, transform=beside, fontsize=8.8, color=SECONDARY,
                  ha="right", va="center", linespacing=1.25)

    axis.set_xlim(0.0, 100.0)
    axis.set_ylim(-0.6, 8.55)
    axis.set_yticks([])
    axis.set_xticks(range(0, 101, 20))
    axis.set_xticklabels([f"{tick}%" for tick in range(0, 101, 20)])
    axis.grid(visible=False)
    axis.spines["left"].set_visible(False)
    axis.set_xlabel("Share of 20,000 simulated runs", fontsize=9.0, labelpad=7)

    handles = [
        Patch(facecolor=colors["covers"], label="Interval contains the true pass rate"),
        Patch(facecolor=colors["misses"], label="Interval misses it"),
        Patch(facecolor=colors["none"], label="No interval: estimated J is zero or negative"),
    ]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.018, 0.85),
                  ncol=3, handlelength=1.0, handleheight=1.0, columnspacing=1.6,
                  handletextpad=0.5)

    heading(
        figure,
        "With a weak judge, coverage among reported intervals hides the missing ones",
        "What a run returns, by judge separation J and number of validation labels",
    )
    footnote(
        figure,
        "Validation-only bootstrap. True pass rate 80%; 2,000 production cases; 1,000 bootstrap "
        "draws per run. J = sensitivity + specificity − 1.\n"
        "Sensitivity equals specificity: 60% at J = 0.20, 65% at 0.30, 70% at 0.40 and 87.5% "
        "at 0.75. Monte Carlo 95% error is at most ±0.6 points.",
    )
    save(figure, "fig3-weak-judge")


def figure_drift(rows: list[dict[str, str]]) -> None:
    """Figure 4. Bias and coverage in Study 3, when the judge changes after validation."""
    study = {
        (round(float(row["sensitivity_decline"]), 4), round(float(row["specificity_decline"]), 4)): row
        for row in rows if row["study"] == "B"
    }
    falls = sorted({key[0] for key in study})
    series = (
        ("Sensitivity falls", VIOLET, "o", lambda fall: (fall, 0.0)),
        ("Both fall", SLATE, "^", lambda fall: (fall, fall)),
        ("Specificity falls", EMERALD, "s", lambda fall: (0.0, fall)),
    )

    figure, (bias_axis, coverage_axis) = plt.subplots(1, 2, figsize=(8.0, 5.0))
    figure.subplots_adjust(left=0.075, right=0.925, top=0.715, bottom=0.255, wspace=0.42)
    x = [100.0 * fall for fall in falls]

    for name, color, marker, select in series:
        selected = [study[select(fall)] for fall in falls]
        bias = [100.0 * float(row["bias"]) for row in selected]
        coverage = [100.0 * float(row["coverage"]) for row in selected]
        style = {"color": color, "linewidth": 2.0, "marker": marker, "markersize": 6.5,
                 "markeredgecolor": "white", "markeredgewidth": 1.0, "zorder": 4,
                 "clip_on": False}
        bias_axis.plot(x, bias, label=name, **style)
        coverage_axis.plot(x, coverage, **style)
        bias_axis.text(10.45, bias[-1], f"{bias[-1]:+.1f}".replace("-", "−"),
                       fontsize=9.0, va="center")
        coverage_axis.text(10.45, coverage[-1], f"{coverage[-1]:.1f}%", fontsize=9.0,
                           va="center")

    bias_axis.axhline(0.0, color=MUTED, linewidth=0.9, zorder=2)
    bias_axis.set_ylim(-11.6, 4.0)
    bias_axis.set_yticks((-10, -5, 0))
    bias_axis.set_yticklabels(("−10", "−5", "0"))
    bias_axis.set_title("Bias of the corrected pass rate (percentage points)",
                        fontsize=9.5, color=INK, loc="left", pad=10)

    coverage_axis.axhline(95.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    coverage_axis.text(10.3, 97.2, "95% intended", fontsize=8.6, color=SECONDARY,
                       ha="right", va="bottom")
    coverage_axis.set_ylim(0.0, 106.0)
    coverage_axis.set_yticks(range(0, 101, 25))
    coverage_axis.set_yticklabels([f"{tick}%" for tick in range(0, 101, 25)])
    coverage_axis.set_title("Coverage of the 95% interval", fontsize=9.5, color=INK,
                            loc="left", pad=10)

    for axis in (bias_axis, coverage_axis):
        axis.set_xlim(-0.3, 10.3)
        axis.set_xticks((0, 2.5, 5, 7.5, 10))
        axis.set_xticklabels(("0", "2.5", "5", "7.5", "10"))
        axis.grid(axis="x", visible=False)
        axis.set_xlabel("Fall after validation (percentage points)", fontsize=9.0, labelpad=7)

    handles, labels = bias_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.018, 0.85),
                  ncol=3, columnspacing=1.8, handletextpad=0.5, handlelength=1.6)

    heading(
        figure,
        "A judge that changes after validation biases the corrected pass rate",
        "Bias and coverage when sensitivity, specificity or both fall between validation and production",
    )
    bias_error = max(
        100.0 * (float(row["bias_mc95_high"]) - float(row["bias"])) for row in study.values()
    )
    coverage_error = max(max(wilson_half_widths(row)) for row in study.values())
    footnote(
        figure,
        "Two-sample bootstrap. True pass rate 80%; 2,000 production cases; 200 validation labels; "
        "validation sensitivity and specificity 87.5%.\n"
        f"3,000 simulated runs per point. Monte Carlo 95% error is at most ±{at_most(bias_error)} "
        f"points of bias and ±{at_most(coverage_error)} points of coverage.",
    )
    save(figure, "fig4-drift")


def figure_setup(main_rows: list[dict[str, str]], weak_rows: list[dict[str, str]]) -> bool:
    """Figure 1. Write the diagram's numbers, then compile the TikZ source."""
    def find(rows: list[dict[str, str]], **conditions: str | float) -> dict[str, str]:
        return next(
            row for row in rows
            if all(
                row[key] == value if isinstance(value, str)
                else abs(float(row[key]) - value) < 1e-9
                for key, value in conditions.items()
            )
        )

    design = {"regime": REFERENCE_DESIGN, "rho_target": 1.0}
    only = find(main_rows, method="textbook_validation_only", **design)
    two_sample = find(main_rows, method="full_fixed_stratum_bootstrap", **design)
    weak = find(weak_rows, method="textbook", validation_total_N="40", J=0.2)
    drift = find(main_rows, study="B", sensitivity_decline=0.05, specificity_decline=0.0)
    numbers = {
        "OnlyCoverage": f"{100.0 * float(only['coverage']):.1f}",
        "TwoSampleCoverage": f"{100.0 * float(two_sample['coverage']):.1f}",
        "ProductionCases": only["target_size_m"],
        "ValidationLabels": only["validation_total_N"],
        "NoInterval": f"{100.0 * (1.0 - float(weak['report_rate'])):.1f}",
        "Discarded": f"{100.0 * float(weak['pooled_nonpositive_J_draw_rate_when_attempted']):.1f}",
        "WeakJ": f"{float(weak['J']):.2f}",
        "WeakLabels": weak["validation_total_N"],
        "DriftBias": f"{-100.0 * float(drift['bias']):.1f}",
        "DriftCoverage": f"{100.0 * float(drift['coverage']):.1f}",
        "DriftFall": f"{100.0 * float(drift['sensitivity_decline']):g}",
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "fig1-numbers.tex").write_text(
        "% Written by make_figures.py from the CSV files in results/. Do not edit.\n"
        + "".join(f"\\def\\{name}{{{value}}}\n" for name, value in numbers.items())
    )

    # On macOS the MacTeX binary has TikZ; an older pdflatex on PATH may not.
    mactex = Path("/Library/TeX/texbin/pdflatex")
    pdflatex = str(mactex) if mactex.exists() else shutil.which("pdflatex")
    pdftocairo = shutil.which("pdftocairo")
    if not pdflatex or not pdftocairo:
        return False
    with tempfile.TemporaryDirectory() as build:
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={build}", "fig1-setup.tex"],
            cwd=OUTPUT, check=True, capture_output=True,
        )
        pdf = Path(build) / "fig1-setup.pdf"
        shutil.copyfile(pdf, OUTPUT / "fig1-setup.pdf")
        subprocess.run([pdftocairo, "-svg", str(pdf), str(OUTPUT / "fig1-setup.svg")], check=True)
        subprocess.run([pdftocairo, "-png", "-r", "200", "-singlefile", str(pdf),
                        str(OUTPUT / "fig1-setup")], check=True)
    return True


def main() -> None:
    main_rows = read_rows(ROOT / "results" / "journal_final" / "journal_summary.csv")
    weak_rows = read_rows(ROOT / "results" / "component_ablation" / "component_summary.csv")
    with plt.rc_context(STYLE):
        figure_production_variance(main_rows)
        figure_weak_judge(weak_rows)
        figure_drift(main_rows)
    print(f"Success: wrote Figures 2 to 4 to {OUTPUT}")
    if figure_setup(main_rows, weak_rows):
        print(f"Success: wrote Figure 1 to {OUTPUT}")
    else:
        print("Warning: pdflatex or pdftocairo not found. Figure 1 was not rebuilt.")


if __name__ == "__main__":
    main()
