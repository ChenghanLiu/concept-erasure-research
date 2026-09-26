# Frozen experiment reporting

Start with [experiment_summary.md](experiment_summary.md) and the
[paper results draft](paper_results_draft.md). All artifacts in this directory
describe existing evidence. No notebook cell or GPU experiment is executed.

## Tables and figures

| Artifact | Contents |
| --- | --- |
| [data_inventory.csv](data_inventory.csv) | Explicitly selected frozen source artifacts and saved notebook output blocks; splits, units, counts, and historical SHA-256 hashes |
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
and never rerun by these scripts. Their saved smoke harness is archival, not a
reusable fresh-run entry point.

There are two separate provenance records:

- [input_manifest.json](input_manifest.json) is the portable regeneration
  allowlist: five frozen CSVs, seven cached subspaces, the adaptive-rank notebook,
  and eight archived smoke files needed to reproduce the frozen inventory.
  Each entry identifies committed source bytes. Verification permits only
  CRLF/LF checkout conversion for text; altered or missing required content
  fails. New experiments, audit outputs, and other files under `results/` do
  not expand this input set.
- [source_snapshot.json](source_snapshot.json) is the unchanged historical
  protection snapshot of 48 files from the original working tree. It includes
  ignored scratch files and checkout-specific hashes; it is archival evidence,
  not a portable regeneration gate. Missing scratch files are never fabricated.

The historical hashes printed in `data_inventory.csv` remain unchanged as
recorded provenance. Portable input verification uses the new manifest; it
does not require a current checkout to reproduce historical line endings.

## Rebuild and validate

From a fresh clone's repository root, build the existing Dockerfile once.
The build downloads its base image and Python packages. These commands work in
PowerShell and Bash, and explicitly mount the current clone rather than relying
on a previously started container:

```text
docker build -t concept-erasure-reporting .
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 --mount "type=bind,source=$PWD,target=/workspace" -w /workspace concept-erasure-reporting python -B results/reporting/generate_report.py --output-dir results/reporting/.rebuild
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 --mount "type=bind,source=$PWD,target=/workspace" -w /workspace concept-erasure-reporting python -B results/reporting/generate_figures.py --tables-dir results/reporting/.rebuild --output-dir results/reporting/.rebuild/figures
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 --mount "type=bind,source=$PWD,target=/workspace" -w /workspace concept-erasure-reporting python -B results/reporting/validate_reporting.py
```

These use Python standard libraries plus the existing CPU-capable PyTorch,
pandas, Pillow, and Matplotlib installations. Network and CUDA are disabled;
there is no model download or GPU requirement. `/workspace` is an explicit
container mount, not an assumed host path.
The reporting scripts import no diffusion pipeline or CLIP evaluator.
`report_narratives.py` renders the three research narratives from the audited
numeric tables. `generate_figures.py` reads only reporting CSVs.

The builders accept `--output-dir` only inside `results/reporting/` and refuse
to overwrite a nonidentical existing artifact. Identical reruns preserve file
contents. Validation rebuilds all 20 generated table, narrative, evidence, and
figure files in a fresh temporary child directory, compares them against the
delivered artifacts, then removes only that temporary directory. Text comparison
permits CRLF/LF conversion; figure comparison is byte-for-byte. No frozen source
is changed. The `.rebuild/` created by the commands above is retained.
Matplotlib's non-result font cache stays under `.mplconfig/`.

`python -B results/reporting/test_reporting_inputs.py` runs seven offline
checks for missing/changed inputs, EOL portability, unrelated future results,
and refusal to overwrite changed artifacts. It uses temporary synthetic files.

The validator prints fresh validation JSON to stdout. Capture it in a new audit
log if needed; do not replace the archived
[validation_report.json](validation_report.json), which records the original
local reporting pass. That record, the source snapshot, and
[git_diff_stat.txt](git_diff_stat.txt) describe historical state and are not
regenerated outputs. The 20 generated files comprise six CSV tables (including
the inventory), five Markdown table summaries, `notebook_evidence.json`, three
narratives (`car_limitation.md`, `experiment_summary.md`, and
`paper_results_draft.md`), and five figures. `audit_notes.md` is a reviewed
archival note, not a generated file.

## Tested environment

[reproducibility_environment.json](reproducibility_environment.json) records
the successful 2026-09-25/26 clean-clone execution and parity environment,
including all observed installed Python packages, the resolved Docker base
digest, and verified SD1.4 and CLIP revisions. It is an environment record,
not an installation lock; the existing Dockerfile and requirements remain
unchanged. These revisions were observed during the clean-clone audit and were
not established as historical experiment pins. That audit obtained exact GPU
parity and identical bytes for all five figures, although complete reporting
regeneration was then blocked by the old snapshot gate.

Exact PNG equality depends on the recorded rendering environment (including
Matplotlib, Pillow, and fonts). A future unpinned build may resolve differently.
The tested base image has an inherited Ninja wheel-metadata warning in
`pip check`; the audit's imports, regression tests, figure rendering, and GPU
parity passed without changing that package.
