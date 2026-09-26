"""Render conservative research narratives from audited frozen result tables.

This module performs no file I/O, model loading, inference, or rank selection.
The reporting builder owns source validation and writing returned Markdown.
"""

from __future__ import annotations


def _number(value, digits=6):
    return f"{float(value):.{digits}f}"


def _percent(value):
    return f"{float(value):.2f}%"


def _selected(value):
    return "none" if value is None or value == "" else str(int(value))


def render_narratives(main_rows, aggregate_rows, style_rows, budget_rows, evidence):
    """Return the three requested Markdown narratives without writing files."""
    main = {row["concept"]: row for row in main_rows}
    aggregate = {row["group"]: row for row in aggregate_rows}
    styles = {(row["concept"], int(row["rank"])): row for row in style_rows}
    car = main["car"]
    calibration = {int(row["rank"]): row for row in evidence["car_calibration"]}
    car_tradeoff = {int(row["rank"]): row for row in evidence["car_tradeoff"]}
    car_test = evidence["car_rank4_test"]
    original_preservation = float(evidence["baseline_preservation"])
    car_fixed_mean = float(car["fixed_target_clip_mean"])
    rank4_delta = float(car_test["mean"]) - car_fixed_mean
    rank4_reduction = 100 * (car_fixed_mean - float(car_test["mean"])) / car_fixed_mean

    calibration_table = "\n".join(
        f"| {rank} | {_number(row['target_clip_mean'])} | "
        f"{_number(row['target_clip_std'])} | {row['precision']} |"
        for rank, row in sorted(calibration.items())
    )
    car_preservation_table = "\n".join(
        f"| {rank} | {_number(row['preservation_score'])} | "
        f"{_percent(row['preservation_degradation_pct'])} |"
        for rank, row in sorted(car_tradeoff.items())
    )
    style_main_text = "; ".join(
        f"{concept}: {_number(main[concept]['fixed_target_clip_mean'])} to "
        f"{_number(main[concept]['adaptive_target_clip_mean'])} "
        f"({_percent(main[concept]['relative_improvement_pct'])} reduction, "
        f"rank {int(main[concept]['adaptive_rank'])})"
        for concept in ("Vincent van Gogh", "Claude Monet")
    )
    rank8_text = "; ".join(
        f"{concept}: {_percent(styles[(concept, 8)]['preservation_degradation_pct'])}"
        for concept in ("Vincent van Gogh", "Claude Monet")
    )
    budget_text = "; ".join(
        f"{concept}: " + ", ".join(
            f"{_number(row['budget_pct'], 0)}% → rank {_selected(row['selected_rank'])}"
            for row in sorted(
                (item for item in budget_rows if item["concept"] == concept),
                key=lambda item: float(item["budget_pct"]),
            )
        )
        for concept in ("Vincent van Gogh", "Claude Monet")
    )
    aggregate_text = "; ".join(
        f"{label}: {_number(aggregate[group]['fixed_target_clip_mean'])} to "
        f"{_number(aggregate[group]['adaptive_target_clip_mean'])} "
        f"({_percent(aggregate[group]['relative_improvement_pct'])} reduction)"
        for group, label in (
            ("all_concepts", "all seven concepts"),
            ("changed_rank", "concepts with a changed rank"),
            ("style", "the two style concepts"),
            ("non_style", "the five non-style concepts"),
        )
    )

    car_limitation = f"""# Car: frozen regression and exploratory follow-up

Lower target CLIP residual is better. Car remains a failure case in the frozen
main evaluation; none of the follow-up findings replace its selected rank or
held-out scores.

## Calibration and first-stage selection

The calibration prompt was `a photo of a car`, with seeds 2025–2027 and 15
inference steps. The candidate ranks were {{1, 2, 4, 8}}. Rank 2 had the lowest
observed calibration mean and is the frozen first-stage selection.

| Rank | Calibration mean | Sample SD | Source precision |
|---:|---:|---:|---|
{calibration_table}

Ranks 1, 2, and 4 come from [pilot_results.csv](../pilot_results.csv). Rank 8
is recovered from six-decimal saved notebook output (cells 49 and 92), so its
summary is approximate; it is not a new measurement.

## Held-out failure

On seeds 3025–3029, fixed rank 1 yielded
{_number(car['fixed_target_clip_mean'])} ± {_number(car['fixed_target_clip_std'])},
whereas selected rank 2 yielded
{_number(car['adaptive_target_clip_mean'])} ± {_number(car['adaptive_target_clip_std'])}
(mean ± sample SD). The signed relative improvement is
**{_percent(car['relative_improvement_pct'])}**, a regression. These frozen
[final-test rows](../final_test_results.csv) show calibration-to-test variability
and rank-selection instability for this concept.

## Exploratory preservation and rank 4

Saved notebook cells 87–94 compare the four ranks against original-pipeline
preservation {_number(original_preservation)}:

| Rank | Preservation CLIP ↑ | Degradation |
|---:|---:|---:|
{car_preservation_table}

This is a mixed prompt diagnostic: the four prompts include **`a photo of a
car` itself**, alongside dog, airplane, and person, each with seeds 4025–4027.
It therefore does not isolate preservation of unrelated concepts. These
extended observations are available only in saved notebook output, with
displayed precision; the original baseline comes from
[van_gogh_preservation.csv](../van_gogh_preservation.csv).

The existing 20% exploratory budget accepts ranks 1 and 4 and selects rank 4
(cell 94). Its subsequent test (cell 95) reports mean **{car_test['mean']}** and
sample SD **{car_test['std']}**, worse than fixed rank 1 by
{_number(rank4_delta)} (signed reduction {_percent(rank4_reduction)}).
This reuses the same five final-test seeds; it is not an independent test of a
revised selector. The exploratory rank-4 result does not repair the frozen
rank-2 failure.

Sources: [saved notebook](../../experiments/adaptive_rank_experiment.ipynb),
zero-based cells 26–28, 49, 70/73, 76, and 87–95. No new GPU experiment,
tuning, or change to the frozen protocol is part of this reporting pass.
"""

    experiment_summary = f"""# Frozen experiment summary

## Protocol and evidence

The experiments compare fixed rank 1 with a concept-specific rank selected
from {{1, 2, 4, 8}} using calibration seeds 2025–2027. The main evaluation uses
separate held-out seeds 3025–3029 for seven concepts: one identity, two styles,
two objects, and two animals. The frozen SD1.4 pipeline uses 15 inference
steps, the notebook's layer-0 projection with lambda 4, and its CLIP evaluator.
Lower target CLIP residual indicates stronger measured erasure; higher
preservation CLIP indicates better retained prompt alignment. Neither metric
alone establishes complete erasure or general image quality.

The main tables use only [final_test_results.csv](../final_test_results.csv)
and [selected_ranks.csv](../selected_ranks.csv). Per-concept variability is
sample SD over five seeds. Aggregate percentages compare equal-weight
concept means: 100 × (fixed mean − adaptive mean) / fixed mean; they are not
averages of per-concept percentages. Calibration, preservation, and
exploratory follow-up remain distinct from the main held-out dataset.

## Main quantitative findings

{aggregate_text.capitalize()}.

The strongest held-out gains are observed for the tested artistic styles:
{style_main_text}. Airplane and dog retain rank 1. Car regresses from
{_number(car['fixed_target_clip_mean'])} to {_number(car['adaptive_target_clip_mean'])}
with rank 2 ({_percent(car['relative_improvement_pct'])} signed improvement).
The small non-style aggregate gain should not obscure that failure.
See [main results](main_results.md) and [aggregate results](aggregate_results.md).

## Preservation and rank sensitivity

The style analyses show an erasure-preservation trade-off: optimizing target
residual alone can incur collateral preservation loss. Rank 8 has substantial
observed preservation degradation ({rank8_text}). The existing budget
sensitivity yields {budget_text}. These are existing calibration/preservation
diagnostics, not new held-out performance claims for those budget-selected
ranks. Negative degradation denotes **no observed degradation / sampling
variation**, not evidence that erasure improves unrelated generation.

Sources combine frozen CSVs with explicitly identified saved notebook output.
Notebook-only extended results are limited to saved output precision;
rounded reconstructed values are marked in the tables. The Car preservation
prompt set includes the Car target prompt, and its exploratory rank-4 test
reuses the same held-out seeds and also regresses. See
[Car limitation](car_limitation.md), [style trade-off](style_tradeoff.md), and
[budget sensitivity](preservation_budget_sensitivity.md).

## Supported conclusions and limits

The observations support **concept-dependent rank sensitivity**, meaningful
gains for the **tested style concepts**, an **erasure-preservation trade-off**,
and **non-monotonic behavior as rank increases**. Fixed rank 1 is not
universally optimal within these observations. Spectrum summaries provide
descriptive information; cumulative-energy thresholds do not directly
correspond to all empirically selected ranks.

These experiments do **not** establish a universal optimal-rank rule,
universal improvement for every concept, prediction of the best rank from
cumulative spectral energy alone, or broad generalization from seven
concepts and five test seeds. Only two styles received the detailed style
preservation analysis, using a small prompt set. No statistical significance
claim or causal explanation of the Car regression is established here.
Reporting uses existing evidence only; no GPU experiment or protocol change
is introduced.
"""

    paper_results_draft = f"""# Results draft: adaptive rank and preservation

We evaluated the frozen first-stage adaptive selector on seven concepts,
comparing its selected rank with fixed rank 1. Selection used three
calibration seeds (2025–2027) and candidates {{1, 2, 4, 8}}; the final comparison
used five separate held-out seeds (3025–3029). All comparisons retain the
existing SD1.4 generation and CLIP scoring protocol. Target CLIP residual is
reported with lower values indicating stronger measured erasure; per-concept
standard deviations summarize sample variation across test seeds.

Across the evaluated concepts, macro-average residual decreased from
{_number(aggregate['all_concepts']['fixed_target_clip_mean'])} to
{_number(aggregate['all_concepts']['adaptive_target_clip_mean'])}, an observed
{_percent(aggregate['all_concepts']['relative_improvement_pct'])} reduction.
For the {int(aggregate['changed_rank']['n_concepts'])} concepts where selection
changed the rank, the reduction was
{_percent(aggregate['changed_rank']['relative_improvement_pct'])}.
These percentages are relative changes in equal-weight concept means, not
mean per-concept percentage changes ([main results](main_results.md);
[aggregate results](aggregate_results.md)).

The largest gains occurred for the tested artistic-style concepts:
{style_main_text}. Their aggregate reduction was
{_percent(aggregate['style']['relative_improvement_pct'])}, compared with
{_percent(aggregate['non_style']['relative_improvement_pct'])} for non-style
concepts. These results support concept-dependent rank sensitivity and
meaningful gains for the two evaluated styles, rather than uniform benefit
from increasing projection rank. Airplane and dog retain the baseline rank.

Car provides a counterexample to universal improvement. Although rank 2
minimized its calibration residual, held-out residual increased from
{_number(car['fixed_target_clip_mean'])} ± {_number(car['fixed_target_clip_std'])}
at rank 1 to
{_number(car['adaptive_target_clip_mean'])} ± {_number(car['adaptive_target_clip_std'])}
at rank 2 (mean ± sample SD; signed improvement
{_percent(car['relative_improvement_pct'])}). This regression is consistent
with calibration-to-test variability and unstable selection for some
concepts; the available evidence does not determine its cause.

The style preservation analyses reveal a trade-off between lower target
residual and retained alignment on preservation prompts. Rank 8 produces
substantially larger preservation degradation ({rank8_text}). For the existing
10%, 15%, and 20% budgets, the observed selections are {budget_text}.
Budget-selected ranks describe the saved trade-off analysis and do not
replace ranks in the frozen main comparison. Negative measured degradation
is treated as no observed degradation / sampling variation, without claiming
improved unrelated generation ([style trade-off](style_tradeoff.md)).

The Car follow-up remains exploratory. Its mixed preservation set includes
the target Car prompt itself. A preservation-aware rank-4 follow-up also
exceeds the fixed-rank-1 target residual (mean {_number(car_test['mean'])},
sample SD {_number(car_test['std'])}; signed reduction
{_percent(rank4_reduction)}). Reuse of the same held-out seeds precludes
presenting it as independent validation of a revised selector.

Overall, the evidence supports non-monotonic rank behavior and an
erasure-preservation trade-off. It does not establish a universal optimal
rank, universal concept-wise improvement, or a rule predicting selected rank
from cumulative spectral energy alone. Seven concepts, five test seeds, and
two detailed style studies are insufficient to establish broad
generalization. No statistical significance claim is made. Extended
trade-off evidence available only in saved notebook outputs retains its
displayed precision; no missing result is inferred or newly generated.
"""

    return {
        "car_limitation.md": car_limitation,
        "experiment_summary.md": experiment_summary,
        "paper_results_draft.md": paper_results_draft,
    }
