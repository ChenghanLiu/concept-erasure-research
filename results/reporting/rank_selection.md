# Recorded ranks and activation spectra

| Concept | Category | Selected rank | PC1 energy | 70% rank | 80% rank | 90% rank | 95% rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Taylor Swift | Identity | 2 | 0.284781 | 10 | 16 | 25 | 31 |
| Vincent van Gogh | Style | 4 | 0.287163 | 7 | 14 | 23 | 30 |
| Claude Monet | Style | 4 | 0.302325 | 7 | 14 | 23 | 29 |
| car | Object | 2 | 0.185784 | 15 | 20 | 27 | 35 |
| airplane | Object | 1 | 0.285750 | 12 | 18 | 26 | 33 |
| dog | Animal | 1 | 0.185772 | 15 | 20 | 27 | 35 |
| cat | Animal | 2 | 0.186648 | 15 | 20 | 27 | 35 |

Selections come from [selected_ranks.csv](../selected_ranks.csv). Spectrum statistics use [spectrum_summary.csv](../spectrum_summary.csv) for its four recorded concepts and the existing cached `.pt` ratio/threshold metadata for the other three; all seven caches were read on CPU, without recomputing SVD. Per-row provenance is in [rank_selection.csv](rank_selection.csv).

Energy ranks are descriptive thresholds of the uncentered activation SVD, not candidate ranks selected for erasure. They differ from the empirically selected ranks; this does not establish a universal spectrum-to-rank rule.
