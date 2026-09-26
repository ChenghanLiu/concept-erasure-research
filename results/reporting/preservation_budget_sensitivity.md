# Preservation-budget sensitivity

| Concept | Budget % | Eligible ranks | Selected rank | Target residual | Preservation degradation % |
| --- | --- | --- | --- | --- | --- |
| Vincent van Gogh | 10 | 1, 2 | 1 | 0.175338 | -1.180452 |
| Vincent van Gogh | 15 | 1, 2 | 1 | 0.175338 | -1.180452 |
| Vincent van Gogh | 20 | 1, 2, 4 | 4 | 0.115241 | 15.778491 |
| Claude Monet | 10 | 1, 2 | 1 | 0.155533 | -1.426951 |
| Claude Monet | 15 | 1, 2, 4 | 4 | 0.133511 | 14.889500 |
| Claude Monet | 20 | 1, 2, 4 | 4 | 0.133511 | 14.889500 |

Select the lowest calibration target residual among ranks with max(0, recorded preservation degradation) ≤ budget. Negative degradation is interpreted as no observed degradation / sampling variation. The 10%, 15%, and 20% budgets and resulting choices exactly reproduce saved notebook cell 82 (20% also checked in cells 84–85). These are sensitivity analyses of the existing trade-off data, not a new independently tested method; they do not replace the frozen first-stage selections.
