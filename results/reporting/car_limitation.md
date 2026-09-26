# Car: frozen regression and exploratory follow-up

Lower target CLIP residual is better. Car remains a failure case in the frozen
main evaluation; none of the follow-up findings replace its selected rank or
held-out scores.

## Calibration and first-stage selection

The calibration prompt was `a photo of a car`, with seeds 2025–2027 and 15
inference steps. The candidate ranks were {1, 2, 4, 8}. Rank 2 had the lowest
observed calibration mean and is the frozen first-stage selection.

| Rank | Calibration mean | Sample SD | Source precision |
|---:|---:|---:|---|
| 1 | 0.144314 | 0.057119 | full precision CSV |
| 2 | 0.117348 | 0.013524 | full precision CSV |
| 4 | 0.141699 | 0.027585 | full precision CSV |
| 8 | 0.157106 | 0.024810 | approximate; recovered six decimal samples |

Ranks 1, 2, and 4 come from [pilot_results.csv](../pilot_results.csv). Rank 8
is recovered from six-decimal saved notebook output (cells 49 and 92), so its
summary is approximate; it is not a new measurement.

## Held-out failure

On seeds 3025–3029, fixed rank 1 yielded
0.112894 ± 0.017916,
whereas selected rank 2 yielded
0.120180 ± 0.023593
(mean ± sample SD). The signed relative improvement is
**-6.45%**, a regression. These frozen
[final-test rows](../final_test_results.csv) show calibration-to-test variability
and rank-selection instability for this concept.

## Exploratory preservation and rank 4

Saved notebook cells 87–94 compare the four ranks against original-pipeline
preservation 0.209174:

| Rank | Preservation CLIP ↑ | Degradation |
|---:|---:|---:|
| 1 | 0.175334 | 16.18% |
| 2 | 0.138311 | 33.88% |
| 4 | 0.178239 | 14.79% |
| 8 | 0.155374 | 25.72% |

This is a mixed prompt diagnostic: the four prompts include **`a photo of a
car` itself**, alongside dog, airplane, and person, each with seeds 4025–4027.
It therefore does not isolate preservation of unrelated concepts. These
extended observations are available only in saved notebook output, with
displayed precision; the original baseline comes from
[van_gogh_preservation.csv](../van_gogh_preservation.csv).

The existing 20% exploratory budget accepts ranks 1 and 4 and selects rank 4
(cell 94). Its subsequent test (cell 95) reports mean **0.126381017267704** and
sample SD **0.015324158214602928**, worse than fixed rank 1 by
0.013487 (signed reduction -11.95%).
This reuses the same five final-test seeds; it is not an independent test of a
revised selector. The exploratory rank-4 result does not repair the frozen
rank-2 failure.

Sources: [saved notebook](../../experiments/adaptive_rank_experiment.ipynb),
zero-based cells 26–28, 49, 70/73, 76, and 87–95. No new GPU experiment,
tuning, or change to the frozen protocol is part of this reporting pass.
