# Frozen experiment summary

## Protocol and evidence

The experiments compare fixed rank 1 with a concept-specific rank selected
from {1, 2, 4, 8} using calibration seeds 2025–2027. The main evaluation uses
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

All seven concepts: 0.145357 to 0.125353 (13.76% reduction); concepts with a changed rank: 0.157304 to 0.129299 (17.80% reduction); the two style concepts: 0.195618 to 0.127710 (34.71% reduction); the five non-style concepts: 0.125252 to 0.124410 (0.67% reduction).

The strongest held-out gains are observed for the tested artistic styles:
Vincent van Gogh: 0.201334 to 0.122011 (39.40% reduction, rank 4); Claude Monet: 0.189903 to 0.133408 (29.75% reduction, rank 4). Airplane and dog retain rank 1. Car regresses from
0.112894 to 0.120180
with rank 2 (-6.45% signed improvement).
The small non-style aggregate gain should not obscure that failure.
See [main results](main_results.md) and [aggregate results](aggregate_results.md).

## Preservation and rank sensitivity

The style analyses show an erasure-preservation trade-off: optimizing target
residual alone can incur collateral preservation loss. Rank 8 has substantial
observed preservation degradation (Vincent van Gogh: 36.66%; Claude Monet: 25.61%). The existing budget
sensitivity yields Vincent van Gogh: 10% → rank 1, 15% → rank 1, 20% → rank 4; Claude Monet: 10% → rank 1, 15% → rank 4, 20% → rank 4. These are existing calibration/preservation
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
