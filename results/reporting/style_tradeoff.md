# Style erasure–preservation trade-off

Target residual: **lower is better**, calibration seeds 2025–2027. Preservation CLIP: **higher is better**, four prompts × seeds 4025–4027 (12 observations per rank). Original preservation mean: 0.209174395849.

## Vincent van Gogh

| Rank | Target residual | Preservation | Degradation % | 10% budget | 15% budget | 20% budget |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.175338 | 0.211644 | -1.180452 | Yes | Yes | Yes |
| 2 | 0.191937 | 0.217647 | -4.050390 | Yes | Yes | Yes |
| 4 | 0.115241 | 0.176170 | 15.778491 | No | No | Yes |
| 8 | 0.137270 | 0.132490 | 36.660355 | No | No | No |

## Claude Monet

| Rank | Target residual | Preservation | Degradation % | 10% budget | 15% budget | 20% budget |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.155533 | 0.212159 | -1.426951 | Yes | Yes | Yes |
| 2 | 0.158137 | 0.211344 | -1.037359 | Yes | Yes | Yes |
| 4 | 0.133511 | 0.178029 | 14.889500 | No | Yes | Yes |
| 8 | 0.164403 | 0.155615 | 25.605185 | No | No | No |

Degradation = 100 × (original preservation − projected preservation) / original preservation. Negative values mean **no observed degradation / sampling variation**, not evidence that erasure improves unrelated generation. Only budget feasibility uses max(0, degradation), matching the notebook.

The target columns are calibration summaries, not the held-out means in the main table. The notebook combines these with the separate preservation seed set for exploratory constrained selection.

Sources: [Van Gogh preservation CSV](../van_gogh_preservation.csv) and saved outputs in [adaptive_rank_experiment.ipynb](../../experiments/adaptive_rank_experiment.ipynb), zero-based cells 50–51, 76–77, 80–82. Van Gogh rank 2/8 full-precision aggregate means survive in cell 76; Monet means and all style target/degradation summaries survive only at six decimal places. Recorded degradation values are retained instead of back-solving missing precision or recomputing them from rounded means. Full provenance is in [style_tradeoff.csv](style_tradeoff.csv) and [notebook_evidence.json](notebook_evidence.json).
