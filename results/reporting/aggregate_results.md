# Macro target-residual results

Lower target CLIP residual is better.

| Group | Concepts | Fixed rank 1 | Adaptive | Relative improvement |
| --- | --- | --- | --- | --- |
| All concepts | 7 | 0.145357 | 0.125353 | 13.76% |
| Changed-rank concepts | 5 | 0.157304 | 0.129299 | 17.80% |
| Style concepts | 2 | 0.195618 | 0.127710 | 34.71% |
| Non-style concepts | 5 | 0.125252 | 0.124410 | 0.67% |

Each method's macro average is the equally weighted mean of its per-concept five-seed means. Relative improvement is computed from those two macro averages, **not** by averaging per-concept percentages. All rows use only [final_test_results.csv](../final_test_results.csv).

Changed-rank: Taylor Swift, Vincent van Gogh, Claude Monet, car, cat. Style: Vincent van Gogh and Claude Monet. Non-style: the other five concepts. Car is retained in every applicable group.
