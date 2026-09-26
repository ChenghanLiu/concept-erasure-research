# Results draft: adaptive rank and preservation

We evaluated the frozen first-stage adaptive selector on seven concepts,
comparing its selected rank with fixed rank 1. Selection used three
calibration seeds (2025–2027) and candidates {1, 2, 4, 8}; the final comparison
used five separate held-out seeds (3025–3029). All comparisons retain the
existing SD1.4 generation and CLIP scoring protocol. Target CLIP residual is
reported with lower values indicating stronger measured erasure; per-concept
standard deviations summarize sample variation across test seeds.

Across the evaluated concepts, macro-average residual decreased from
0.145357 to
0.125353, an observed
13.76% reduction.
For the 5 concepts where selection
changed the rank, the reduction was
17.80%.
These percentages are relative changes in equal-weight concept means, not
mean per-concept percentage changes ([main results](main_results.md);
[aggregate results](aggregate_results.md)).

The largest gains occurred for the tested artistic-style concepts:
Vincent van Gogh: 0.201334 to 0.122011 (39.40% reduction, rank 4); Claude Monet: 0.189903 to 0.133408 (29.75% reduction, rank 4). Their aggregate reduction was
34.71%, compared with
0.67% for non-style
concepts. These results support concept-dependent rank sensitivity and
meaningful gains for the two evaluated styles, rather than uniform benefit
from increasing projection rank. Airplane and dog retain the baseline rank.

Car provides a counterexample to universal improvement. Although rank 2
minimized its calibration residual, held-out residual increased from
0.112894 ± 0.017916
at rank 1 to
0.120180 ± 0.023593
at rank 2 (mean ± sample SD; signed improvement
-6.45%). This regression is consistent
with calibration-to-test variability and unstable selection for some
concepts; the available evidence does not determine its cause.

The style preservation analyses reveal a trade-off between lower target
residual and retained alignment on preservation prompts. Rank 8 produces
substantially larger preservation degradation (Vincent van Gogh: 36.66%; Claude Monet: 25.61%). For the existing
10%, 15%, and 20% budgets, the observed selections are Vincent van Gogh: 10% → rank 1, 15% → rank 1, 20% → rank 4; Claude Monet: 10% → rank 1, 15% → rank 4, 20% → rank 4.
Budget-selected ranks describe the saved trade-off analysis and do not
replace ranks in the frozen main comparison. Negative measured degradation
is treated as no observed degradation / sampling variation, without claiming
improved unrelated generation ([style trade-off](style_tradeoff.md)).

The Car follow-up remains exploratory. Its mixed preservation set includes
the target Car prompt itself. A preservation-aware rank-4 follow-up also
exceeds the fixed-rank-1 target residual (mean 0.126381,
sample SD 0.015324; signed reduction
-11.95%). Reuse of the same held-out seeds precludes
presenting it as independent validation of a revised selector.

Overall, the evidence supports non-monotonic rank behavior and an
erasure-preservation trade-off. It does not establish a universal optimal
rank, universal concept-wise improvement, or a rule predicting selected rank
from cumulative spectral energy alone. Seven concepts, five test seeds, and
two detailed style studies are insufficient to establish broad
generalization. No statistical significance claim is made. Extended
trade-off evidence available only in saved notebook outputs retains its
displayed precision; no missing result is inferred or newly generated.
