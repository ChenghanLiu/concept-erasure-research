"""Render publication figures from frozen reporting tables, without model imports.

Run with ``python -B results/reporting/generate_figures.py`` after generating
the reporting CSV tables. This script uses only the standard library and CPU
Matplotlib, reads no model weights, and never runs an experiment.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORTING_ROOT = REPOSITORY_ROOT / "results" / "reporting"
CONCEPTS = (
    "Taylor Swift", "Vincent van Gogh", "Claude Monet", "car", "airplane", "dog", "cat"
)
STYLE_CONCEPTS = ("Vincent van Gogh", "Claude Monet")
RANKS = (1, 2, 4, 8)
BUDGETS = (10, 15, 20)
BLUE = "#2873A6"
TEAL = "#268477"
ORANGE = "#C96E32"
RED = "#B34F58"
GRAY = "#6B7280"


def _inside_reporting(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(REPORTING_ROOT.resolve()):
        raise ValueError(f"Path must remain inside {REPORTING_ROOT}: {resolved}")
    return resolved


def _read_csv(directory: Path, name: str, required: tuple[str, ...]) -> list[dict]:
    path = _inside_reporting(directory / name)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} has no data rows")
    return rows


def _number(row: dict, column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {column} for {row.get('concept', 'row')}")
    return value


def _integer(row: dict, column: str) -> int:
    value = _number(row, column)
    if not value.is_integer():
        raise ValueError(f"Expected an integer in {column}: {value}")
    return int(value)


def _load_tables(directory: Path) -> tuple[list[dict], list[dict], list[dict]]:
    main = _read_csv(directory, "main_results.csv", (
        "concept", "category", "fixed_rank", "adaptive_rank",
        "fixed_target_clip_mean", "fixed_target_clip_std",
        "adaptive_target_clip_mean", "adaptive_target_clip_std",
        "relative_improvement_pct",
    ))
    if len(main) != len(CONCEPTS) or {r["concept"] for r in main} != set(CONCEPTS):
        raise ValueError("Expected exactly one main-results row for each frozen concept")
    main = sorted(main, key=lambda row: CONCEPTS.index(row["concept"]))
    for row in main:
        if _integer(row, "fixed_rank") != 1 or _integer(row, "adaptive_rank") not in RANKS:
            raise ValueError("Unexpected fixed/adaptive rank in the frozen main table")
        for column in ("fixed_target_clip_mean", "adaptive_target_clip_mean",
                       "fixed_target_clip_std", "adaptive_target_clip_std",
                       "relative_improvement_pct"):
            _number(row, column)
        if min(_number(row, "fixed_target_clip_std"),
               _number(row, "adaptive_target_clip_std")) < 0:
            raise ValueError("A sample standard deviation cannot be negative")

    style = _read_csv(directory, "style_tradeoff.csv", (
        "concept", "rank", "target_residual", "preservation_score",
        "preservation_degradation_pct", "original_preservation_score",
    ))
    keys = [(r["concept"], _integer(r, "rank")) for r in style]
    expected = {(concept, rank) for concept in STYLE_CONCEPTS for rank in RANKS}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected exactly four frozen candidate ranks for each style")
    for row in style:
        for column in ("target_residual", "preservation_score",
                       "preservation_degradation_pct", "original_preservation_score"):
            _number(row, column)

    budgets = _read_csv(directory, "preservation_budget_sensitivity.csv", (
        "concept", "budget_pct", "selected_rank",
    ))
    keys = [(r["concept"], _integer(r, "budget_pct")) for r in budgets]
    expected = {(concept, budget) for concept in STYLE_CONCEPTS for budget in BUDGETS}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected the frozen 10%, 15%, and 20% budgets for each style")
    for row in budgets:
        if _integer(row, "selected_rank") not in RANKS:
            raise ValueError("Budget selector returned a rank outside the frozen candidates")
    return main, style, budgets


def _initialize_matplotlib():
    # Set this before importing Matplotlib: even font caches stay under reporting.
    config = _inside_reporting(REPORTING_ROOT / ".mplconfig")
    config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9CA3AF", "axes.labelcolor": "#273244",
        "text.color": "#273244", "xtick.color": "#273244",
        "ytick.color": "#273244", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300,
    })
    return plt


def _label(concept: str) -> str:
    return {"Taylor Swift": "Taylor\nSwift", "Vincent van Gogh": "Vincent\nvan Gogh",
            "Claude Monet": "Claude\nMonet"}.get(concept, concept)


def _grid(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D8DDE3", linewidth=0.7, alpha=0.8)


def _footnote(figure, text: str) -> None:
    figure.text(0.055, 0.025, text, ha="left", va="bottom", fontsize=8.1,
                linespacing=1.45, color="#4B5563")


def _main_figure(plt, rows: list[dict]):
    figure, axis = plt.subplots(figsize=(12.8, 6.4))
    x = list(range(len(rows)))
    fixed = [_number(row, "fixed_target_clip_mean") for row in rows]
    adaptive = [_number(row, "adaptive_target_clip_mean") for row in rows]
    fixed_std = [_number(row, "fixed_target_clip_std") for row in rows]
    adaptive_std = [_number(row, "adaptive_target_clip_std") for row in rows]
    width = 0.34
    axis.bar([v - width / 2 for v in x], fixed, width, yerr=fixed_std,
             color=BLUE, label="Fixed rank 1", capsize=4,
             error_kw={"elinewidth": 1.1, "ecolor": "#273244"})
    axis.bar([v + width / 2 for v in x], adaptive, width, yerr=adaptive_std,
             color=ORANGE, label="Adaptive (selected rank labeled)", capsize=4,
             error_kw={"elinewidth": 1.1, "ecolor": "#273244"})
    upper = max(m + s for means, stds in ((fixed, fixed_std), (adaptive, adaptive_std))
                for m, s in zip(means, stds))
    for index, row in enumerate(rows):
        axis.text(index + width / 2, adaptive[index] + adaptive_std[index] + upper * 0.02,
                  f"r={_integer(row, 'adaptive_rank')}", ha="center", fontsize=9)
    axis.set_ylim(0, upper * 1.21)
    axis.set_xticks(x, [_label(row["concept"]) for row in rows])
    axis.set_ylabel("Target CLIP cosine similarity (lower is better)")
    axis.set_title("Fixed and adaptive ranks on the heldout target test", loc="left", pad=15)
    axis.legend(loc="upper right", ncol=2)
    _grid(axis)
    figure.subplots_adjust(left=0.075, right=0.98, top=0.87, bottom=0.27)
    _footnote(figure, "Source: frozen final_test_results.csv, summarized in main_results.csv. "
              "Bars: mean ± sample SD; n=5 test seeds per method and concept.\n"
              "Test seeds 3025–3029; 15 inference steps. Adaptive ranks were selected on calibration seeds 2025–2027.\n"
              "Airplane and dog reuse the fixed rank-1 scores in the frozen notebook protocol; error bars are descriptive, not confidence intervals.")
    return figure


def _improvement_figure(plt, rows: list[dict]):
    figure, axis = plt.subplots(figsize=(12.8, 6.2))
    values = []
    for row in rows:
        value = _number(row, "relative_improvement_pct")
        if _integer(row, "adaptive_rank") == 1:
            if abs(value) > 1e-8:
                raise ValueError("Rank-1 reused scores differ beyond serialization noise")
            value = 0.0
        values.append(value)
    colors = [TEAL if value > 0 else RED if value < 0 else GRAY for value in values]
    x = list(range(len(rows)))
    axis.bar(x, values, width=0.62, color=colors)
    span = max(values + [0.0]) - min(values + [0.0])
    padding = max(1.0, span * 0.035)
    for index, value in enumerate(values):
        axis.text(index, value + (padding if value >= 0 else -padding),
                  f"{value:+.2f}%" if value != 0 else "0.00%",
                  ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    axis.axhline(0, color="#273244", linewidth=1.0)
    axis.set_ylim(min(values + [0.0]) - 4 * padding, max(values + [0.0]) + 5 * padding)
    axis.set_xticks(x, [_label(row["concept"]) for row in rows])
    axis.set_ylabel("Relative target-score improvement (%)")
    axis.set_title("Adaptive rank changes the heldout target score", loc="left", pad=15)
    _grid(axis)
    figure.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.27)
    _footnote(figure, "Source: main_results.csv, derived from frozen final_test_results.csv; "
              "five test seeds per method and concept.\n"
              "Improvement = 100 × (fixed mean − adaptive mean) / fixed mean. "
              "Positive is better; negative indicates a higher target residual.\n"
              "Reused rank-1 conditions (airplane, dog) display as zero; negligible CSV serialization noise is retained in the source table.")
    return figure


def _style_limits(rows: list[dict]) -> tuple[tuple[float, float], ...]:
    target_max = max(_number(row, "target_residual") for row in rows)
    preservation_max = max(_number(row, column) for row in rows
                           for column in ("preservation_score", "original_preservation_score"))
    drops = [_number(row, "preservation_degradation_pct") for row in rows]
    drop_min, drop_max = min(drops + [0.0]), max(drops + list(BUDGETS))
    margin = (drop_max - drop_min) * 0.10
    return ((0.0, target_max * 1.15), (0.0, preservation_max * 1.15),
            (drop_min - margin, drop_max + margin))


def _style_figure(plt, all_rows: list[dict], concept: str):
    rows = sorted((row for row in all_rows if row["concept"] == concept),
                  key=lambda row: _integer(row, "rank"))
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 5.6))
    limits = _style_limits(all_rows)
    original = _number(rows[0], "original_preservation_score")
    if any(not math.isclose(_number(row, "original_preservation_score"), original,
                            rel_tol=0.0, abs_tol=1e-12) for row in rows):
        raise ValueError("Style rows disagree on their original preservation comparator")
    fields = ("target_residual", "preservation_score", "preservation_degradation_pct")
    titles = ("Target residual ↓", "Preservation score ↑", "Preservation degradation ↓")
    labels = ("CLIP cosine similarity", "CLIP cosine similarity", "Change from original (%)")
    for axis, field, title, ylabel, color, ylim in zip(
            axes, fields, titles, labels, (BLUE, TEAL, RED), limits):
        axis.plot(range(4), [_number(row, field) for row in rows],
                  marker="o", markersize=6, color=color, linewidth=1.7)
        axis.set_xticks(range(4), [str(rank) for rank in RANKS])
        axis.set_xlabel("Candidate rank")
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        _grid(axis)
    axes[1].axhline(original, linestyle="--", color=GRAY, linewidth=1.1,
                    label=f"Original: {original:.6f}")
    axes[1].legend(loc="lower left", fontsize=8.5)
    axes[2].axhline(0, color=GRAY, linewidth=0.8)
    for budget, color, dash in zip(BUDGETS, ("#8291A8", "#AA8548", "#8D70A2"),
                                    ((4, 2), (2, 2), (6, 2))):
        axes[2].axhline(budget, color=color, linestyle=(0, dash), linewidth=1.0,
                        label=f"{budget}% budget")
    axes[2].legend(loc="upper left", fontsize=8)
    figure.suptitle(f"{concept}: rank tradeoff on calibration and preservation data",
                     x=0.06, ha="left", fontsize=14, y=0.97)
    figure.subplots_adjust(left=0.065, right=0.98, top=0.83, bottom=0.32, wspace=0.35)
    precision = ("Van Gogh preservation uses full-precision CSV / saved aggregate means; "
                 "target residuals and degradation retain six-decimal notebook precision."
                 if concept == "Vincent van Gogh" else
                 "Monet means and degradation retain six-decimal notebook precision; "
                 "the original preservation comparator comes from the frozen CSV.")
    _footnote(figure, "Source: style_tradeoff.csv; frozen adaptive_rank_experiment.ipynb "
              "and van_gogh_preservation.csv. Identical axis limits in both style figures.\n"
              "Target: calibration mean, n=3 seeds (2025–2027). Preservation: "
              "4 prompts × 3 seeds (4025–4027), n=12; 15 inference steps throughout.\n"
              + precision + "\n"
              "Negative degradation: no observed degradation / sampling variation; "
              "treated as zero for budgets. Points show recorded ranks only.")
    return figure


def _budget_figure(plt, rows: list[dict]):
    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    for concept, color, marker, offset in zip(STYLE_CONCEPTS, (BLUE, ORANGE),
                                               ("o", "s"), (-0.025, 0.025)):
        subset = sorted((row for row in rows if row["concept"] == concept),
                        key=lambda row: _integer(row, "budget_pct"))
        selected = [_integer(row, "selected_rank") for row in subset]
        x = [index + offset for index in range(len(BUDGETS))]
        y = [RANKS.index(rank) for rank in selected]
        axis.plot(x, y, color=color, marker=marker, linewidth=1.7,
                  markersize=7, label=concept)
        for xpos, ypos, rank in zip(x, y, selected):
            axis.annotate(str(rank), (xpos, ypos), xytext=(-12 if offset < 0 else 12, 9),
                          textcoords="offset points", ha="center", color=color, fontsize=11)
    axis.set_xticks(range(len(BUDGETS)), [f"{budget}%" for budget in BUDGETS])
    axis.set_yticks(range(len(RANKS)), [str(rank) for rank in RANKS])
    axis.set_xlim(-0.25, 2.25)
    axis.set_ylim(-0.4, len(RANKS) - 0.55)
    axis.set_xlabel("Allowed preservation degradation")
    axis.set_ylabel("Selected rank (categorical candidates)")
    axis.set_title("Preservation budget changes the selected style rank", loc="left", pad=15)
    axis.legend(loc="upper right")
    _grid(axis)
    figure.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.27)
    _footnote(figure, "Source: preservation_budget_sensitivity.csv; frozen notebook selections. "
              "Candidate ranks: {1, 2, 4, 8}.\n"
              "Choose the lowest calibration target residual among ranks within budget; "
              "negative observed degradation is clipped to zero.\n"
              "Markers show only the three recorded budgets; connecting segments guide the eye. "
              "No new experiment or independent heldout claim.")
    return figure


def generate(tables_dir: str | Path | None = None,
             output_dir: str | Path | None = None) -> list[Path]:
    """Create five 300-dpi PNGs; refuse to replace different existing contents.

    Both optional directories must resolve under repository/results/reporting.
    Identical existing files are left byte-for-byte unchanged.
    """
    tables = _inside_reporting(tables_dir if tables_dir is not None else REPORTING_ROOT)
    output = _inside_reporting(output_dir if output_dir is not None else REPORTING_ROOT / "figures")
    main, style, budgets = _load_tables(tables)
    plt = _initialize_matplotlib()
    producers = (
        ("fixed_vs_adaptive_by_concept.png", lambda: _main_figure(plt, main)),
        ("relative_improvement_by_concept.png", lambda: _improvement_figure(plt, main)),
        ("style_rank_tradeoff_van_gogh.png", lambda: _style_figure(plt, style, STYLE_CONCEPTS[0])),
        ("style_rank_tradeoff_monet.png", lambda: _style_figure(plt, style, STYLE_CONCEPTS[1])),
        ("preservation_budget_selection.png", lambda: _budget_figure(plt, budgets)),
    )
    pending = []
    for name, producer in producers:
        figure = producer()
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=300,
                           metadata={"Software": "concept-erasure frozen reporting"})
            content = buffer.getvalue()
        finally:
            plt.close(figure)
        destination = _inside_reporting(output / name)
        if destination.exists() and destination.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite different existing figure: {destination}")
        pending.append((destination, content))
    # Check every intended output before writing any new figures.
    output.mkdir(parents=True, exist_ok=True)
    for destination, content in pending:
        if not destination.exists():
            with destination.open("xb") as handle:
                handle.write(content)
    return [destination for destination, _ in pending]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=REPORTING_ROOT,
                        help="Reporting CSV directory (must be under results/reporting)")
    parser.add_argument("--output-dir", type=Path, default=REPORTING_ROOT / "figures",
                        help="PNG destination (must be under results/reporting)")
    arguments = parser.parse_args()
    for path in generate(arguments.tables_dir, arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
