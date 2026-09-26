# Held-out target CLIP residual

Lower is better. Mean ± sample SD (ddof=1), five paired test seeds 3025–3029 per concept and method.

| Concept | Category | Fixed rank | Adaptive rank | Fixed mean ± SD | Adaptive mean ± SD | Relative improvement |
| --- | --- | --- | --- | --- | --- | --- |
| Taylor Swift | Identity | 1 | 2 | 0.169960 ± 0.018578 | 0.161609 ± 0.021833 | 4.91% |
| Vincent van Gogh | Style | 1 | 4 | 0.201334 ± 0.013775 | 0.122011 ± 0.025540 | 39.40% |
| Claude Monet | Style | 1 | 4 | 0.189903 ± 0.007136 | 0.133408 ± 0.022998 | 29.75% |
| car | Object | 1 | 2 | 0.112894 ± 0.017916 | 0.120180 ± 0.023593 | -6.45% |
| airplane | Object | 1 | 1 | 0.127617 ± 0.021802 | 0.127617 ± 0.021802 | 0.00% |
| dog | Animal | 1 | 1 | 0.103359 ± 0.030526 | 0.103359 ± 0.030526 | 0.00% |
| cat | Animal | 1 | 2 | 0.112432 ± 0.027367 | 0.109284 ± 0.012089 | 2.80% |

Relative improvement = 100 × (fixed mean − adaptive mean) / fixed mean. Negative values indicate regression; Car remains visible. Source: [frozen final test](../final_test_results.csv), ranks from [recorded selections](../selected_ranks.csv). No calibration or smoke-test rows enter this table.

Airplane and dog use rank 1 for both methods and reuse baseline scores. Their negligible final-digit CSV serialization differences are retained in calculations and display as 0.00%. SD describes seed variation, not a confidence interval or significance test.
