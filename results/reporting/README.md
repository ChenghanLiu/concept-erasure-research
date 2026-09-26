# Frozen experiment reporting

Start with [experiment_summary.md](experiment_summary.md) and the
[paper results draft](paper_results_draft.md). All artifacts in this directory
describe existing evidence. No notebook cell or GPU experiment is executed.

## Tables and figures

| Artifact | Contents |
| --- | --- |
| [data_inventory.csv](data_inventory.csv) | Every existing `results/` artifact and the saved notebook output blocks used for extended analyses; splits, units, counts, and SHA-256 hashes |
| [main_results.csv](main_results.csv), [Markdown](main_results.md) | Seven held-out concept comparisons; mean and sample SD over five seeds |
| [aggregate_results.csv](aggregate_results.csv), [Markdown](aggregate_results.md) | Equally weighted concept means for all, changed-rank, style, and non-style groups |
| [rank_selection.csv](rank_selection.csv), [Markdown](rank_selection.md) | Frozen selected ranks and recorded spectral-energy statistics |
| [style_tradeoff.csv](style_tradeoff.csv), [Markdown](style_tradeoff.md) | Separate calibration residuals and preservation summaries for both styles |
| [preservation_budget_sensitivity.csv](preservation_budget_sensitivity.csv), [Markdown](preservation_budget_sensitivity.md) | Recorded 10%, 15%, and 20% budget selections |
| [car_limitation.md](car_limitation.md) | Calibration, frozen regression, and exploratory preservation/rank-4 limitations |
| [audit_notes.md](audit_notes.md) | Missing precision, incomplete source tables, and interpretation boundaries |

Five 300-dpi figures are saved in `figures/`:

- [Fixed versus adaptive](figures/fixed_vs_adaptive_by_concept.png)
- [Relative improvement, including Car regression](figures/relative_improvement_by_concept.png)
- [Van Gogh trade-off](figures/style_rank_tradeoff_van_gogh.png)
- [Monet trade-off](figures/style_rank_tradeoff_monet.png)
- [Preservation-budget selections](figures/preservation_budget_selection.png)

## Sources

Primary CSVs: `results/final_test_results.csv`, `results/selected_ranks.csv`,
`results/spectrum_summary.csv`, `results/pilot_results.csv`, and
`results/van_gogh_preservation.csv`. Cached `results/subspaces/*.pt` records
provide existing spectral metadata for all seven concepts, including three
absent from the spectrum CSV. They are deserialized on CPU; no SVD is run.

Saved outputs in `experiments/adaptive_rank_experiment.ipynb` supply the
extended style and exploratory Car evidence. Cell references are zero-based.
Per-field source and precision information are included in the CSV tables;
[notebook_evidence.json](notebook_evidence.json) retains the extracted source
excerpts and Car summaries. The earlier GPU-parity files in `results/runs/`
are inventoried as implementation validation, excluded from research averages,
and never rerun by these scripts.

## Rebuild and validate

From the repository root, using the existing Docker container:

```powershell
docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w /workspace concept-erasure python -B results/reporting/generate_report.py
docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w /workspace concept-erasure python -B results/reporting/generate_figures.py
docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w /workspace concept-erasure python -B results/reporting/validate_reporting.py
```

These use Python standard libraries plus the existing CPU-capable PyTorch,
pandas, Pillow, and Matplotlib installations. CUDA visibility is disabled.
The reporting scripts import no diffusion pipeline or CLIP evaluator.
`report_narratives.py` renders the three research narratives from the audited
numeric tables. `generate_figures.py` reads only reporting CSVs.

The builders accept `--output-dir` only inside `results/reporting/` and refuse
to overwrite a nonidentical existing artifact. Identical reruns preserve file
contents. Validation rebuilds all 20 generated table, narrative, evidence, and
figure files in a fresh temporary child directory, compares them byte for
byte, then removes only that temporary directory. No frozen source is changed.
Matplotlib's non-result font cache stays under `.mplconfig/`.

The before-reporting [source snapshot](source_snapshot.json) records 48
protected file hashes. [validation_report.json](validation_report.json)
records source checks, independent numerical checks, figure dimensions, and
the clean rebuild comparison. [git_diff_stat.txt](git_diff_stat.txt) records
the requested Git status: pre-existing notebook changes are unrelated to this
reporting pass; newly created reporting files are untracked and therefore do
not appear in ordinary `git diff --stat`.
