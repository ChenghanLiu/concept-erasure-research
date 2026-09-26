# Reproducibility audit after delivery fixes

**Verdict: READY FOR DELIVERY for the reviewed working-tree candidate.**

B1, B2, W1, W3, and W4 are resolved. Complete reporting regeneration and
validation pass from Git-derived project contents in both LF and CRLF checkout
representations. No scientific methodology, result, notebook, rank, seed,
prompt, metric, or protected artifact was changed. No new GPU experiment ran.

This verdict applies to the proposed files listed below. They have **not** been
committed or pushed; origin still contains the pre-fix commit. The canonical
staging area is unchanged and empty.

## Scope and validation source

- Authoritative audit: `results/reproducibility_audit.md`, preserved unchanged.
- Date: 2026-09-26; validation completed at approximately 04:23 UTC.
- Canonical repository: `D:\Projects\concept-erasure`.
- Base commit: `41763a5bcfa38065f85c61b2bceb3dd24e1b21f7`.
- Retained fresh clone: `D:\Projects\concept-erasure-delivery-validation-20260926`.
- Tested proposed Git tree: `7c9f7a589c4f0d75acef4b32d321338f0bf61799`.
- Runtime: the unchanged, previously audited Docker image
  `concept-erasure-clean-audit:41763a5`, identity
  `sha256:213a99b281ff28f4edb1c0d24f620e682d23c54f6dcf0e8b89e9e0245c81be51`.

To reconcile fresh-checkout testing with the instruction not to commit, the
validation repository was cloned from origin. An alternate temporary Git index
was initialized from the base commit and populated with only the explicit
proposed code/documentation files. `git write-tree` created the candidate tree
without a commit. `git archive` exported that tree into the fresh clone, first
with `core.autocrlf=false`, then with Windows CRLF conversion enabled. No
scientific input was copied from an untracked or ignored canonical file.

The alternate index and archive files are retained under the validation clone's
`.git/`. The canonical index was never used to stage these changes. This final
audit document was written after validation and is not itself part of the tested
tree; it is outside the reporting input allowlist.

All execution used only the validation clone as the project mount, with Docker
networking and CUDA visibility disabled. No canonical model cache or scratch
files were mounted. The prior tested image supplies Python/packages, not project
source. The Dockerfile and requirements were not modified or rebuilt for these
code-only fixes.

## Resolved findings

| Finding | Status | Change and evidence |
| --- | --- | --- |
| B1: ignored historical snapshot dependencies | RESOLVED | Reporting verifies 21 explicit committed inputs; no scratch/checkpoint/Untitled dependency. Fresh clone regeneration passes. |
| B2: stale and checkout-dependent hashes | RESOLVED | Verification uses SHA-256 derived from committed Git blobs. Declared text permits only CRLF-to-LF conversion; binary remains exact. LF and CRLF validations pass. |
| W1: growing `results/` inventory | RESOLVED | Inventory and validation use the manifest allowlist, never a recursive scan of future result files. Extra audit/run files do not change outputs. |
| W3: hardcoded setup path | RESOLVED | Root README supplies portable Windows and Linux clone-directory examples. |
| W4: absent root reporting instructions | RESOLVED | Root README links the reporting guide and gives CPU/offline rebuild and validation commands for the current clone. |

### Archival provenance versus portable inputs

`results/reporting/source_snapshot.json` remains byte-for-byte unchanged as the
historical protection record. Neither reporting entry point uses it as a
regeneration prerequisite. The historical `validation_report.json` is also
unchanged; current validation prints a new report to stdout.

`results/reporting/input_manifest.json` includes only the inputs consumed by
the frozen report's calculations and inventory:

- Five frozen CSVs.
- Seven cached subspaces.
- The adaptive-rank notebook with saved evidence.
- Eight archived GPU-smoke files included in the original inventory, excluded
  from scientific aggregates and never executed by reporting.

These 21 inputs correspond to 20 whole-file result inventory entries plus the
notebook source. Their hashes were computed from Git blobs at the base commit,
not from the canonical working-tree line endings. Verification requires no Git
executable inside Docker, checks file existence/content before creating output,
and verifies inputs again afterward.

Three smoke inventory entries historically recorded CRLF-byte hashes. Those
values remain explicitly labeled `historical_inventory_sha256` metadata in the
new manifest so `data_inventory.csv` can be reproduced unchanged. They are never
used to verify current files; `committed_sha256` is the integrity check. No
old hash was made to match by editing a notebook or result.

## Validation results

| Check | Result |
| --- | --- |
| Syntax compilation of source, runner, reporting | PASS |
| Imports of production and reporting modules | PASS |
| Original offline regression suite | PASS, 9/9 |
| New provenance/EOL tests | PASS, 7/7 |
| Cached subspace loading | PASS, 7/7 on CPU, both SVD entry points patched to fail |
| Complete regeneration into a new directory | PASS, all 20 generated artifacts |
| LF output comparison | PASS, 20/20 byte-identical to committed reporting artifacts |
| Reporting validation | PASS, 140 independent numerical checks, 21 verified inputs |
| CRLF checkout generation | PASS, existing text artifacts accepted without rewriting |
| CRLF checkout validation | PASS, same complete rebuild; only text EOL representation ignored |
| Five figure comparisons | PASS, exact PNG bytes in LF and CRLF validation |
| Future result/audit independence | PASS, unrelated files ignored; inventory and outputs unchanged |
| Missing required CSV/notebook | PASS, rejected before output-directory creation |
| Altered required CSV/binary cache | PASS, rejected before output-directory creation |
| Non-EOL alteration of generated artifact | PASS, overwrite refused |
| Canonical protected-file integrity | PASS, 74 baseline files checked, zero differences |
| GPU experiments during fixes | 0 |

Negative end-to-end tests inject missing/altered read behavior in memory; they
do not edit scientific files. The seven proposed provenance regression tests
use temporary synthetic fixtures. The full CRLF test uses Git's archive checkout
conversion only in the validation clone, not normalization of canonical evidence.

The complete generated package comprises these 20 unchanged outputs:

```text
aggregate_results.csv
aggregate_results.md
car_limitation.md
data_inventory.csv
experiment_summary.md
figures/fixed_vs_adaptive_by_concept.png
figures/preservation_budget_selection.png
figures/relative_improvement_by_concept.png
figures/style_rank_tradeoff_monet.png
figures/style_rank_tradeoff_van_gogh.png
main_results.csv
main_results.md
notebook_evidence.json
paper_results_draft.md
preservation_budget_sensitivity.csv
preservation_budget_sensitivity.md
rank_selection.csv
rank_selection.md
style_tradeoff.csv
style_tradeoff.md
```

Archival records such as the historical snapshot, validation JSON, audit notes,
and Git-status capture are preserved records, not outputs to regenerate. There
were no numerical, narrative, inventory, or figure differences in the complete
LF rebuild. Windows comparisons ignore only CRLF/LF text representation;
subspace binaries and PNGs remain byte-exact.

### Commands and retained evidence

All Python commands below ran from the fresh validation clone through:

```powershell
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 --mount 'type=bind,source=D:\Projects\concept-erasure-delivery-validation-20260926,target=/workspace' -w /workspace concept-erasure-clean-audit:41763a5 python -B <arguments>
```

```text
-m compileall -q src run_experiment.py results/reporting
-m src.checks
results/reporting/test_reporting_inputs.py
results/reporting/generate_report.py --output-dir results/reporting/.delivery-rebuild
results/reporting/generate_figures.py --tables-dir results/reporting/.delivery-rebuild --output-dir results/reporting/.delivery-rebuild/figures
results/reporting/validate_reporting.py
```

The clone-only acceptance harness additionally imports production/reporting
modules, loads all seven caches with SVD disabled, compares all output bytes,
and exercises the negative/inventory cases. Its exact commands, per-artifact
SHA-256 values, and results are retained under
`results/runs/delivery-validation/` in the validation clone:

```text
check_delivery.py
acceptance_results.json
acceptance.log
syntax.log
imports.log
regression.log
reporting_tests.log
rebuild_tables.log
rebuild_figures.log
validation-lf.log
generation-crlf.log
validation-crlf.log
```

After Git exported the CRLF representation, the default generator was run
against the existing reporting directory and the validator was run again.
The already-created new run/audit files remained present throughout, testing
that they do not expand the frozen inventory. The rebuilt package is retained
in `results/reporting/.delivery-rebuild/` in the clone. The clone was not deleted.

## Protected-file integrity

All 74 existing files in the preservation baseline have identical SHA-256
values after the work. The baseline covers tracked files outside the four
authorized code/documentation edits, existing ignored files named by the old
snapshot, and the authoritative original audit. No notebook, experimental CSV,
cached subspace, production `src/` module, runner, baseline artifact, generated
reporting result, figure, Dockerfile, or requirements file changed.

Two explicit archival checks:

| File | Unchanged SHA-256 |
| --- | --- |
| `results/reporting/source_snapshot.json` | `531c341577c7abc40d9958d358c798b89874a42a6bcb7cf17e099b1c32673f53` |
| `results/reporting/validation_report.json` | `713d6e22c5f34c34be04d5dd332c12d1bcb71a9807177ea983b3cbece189a76e` |

## Reproducibility record and remaining warnings

`results/reporting/reproducibility_environment.json` records the prior successful
clean-clone audit's 196 installed distributions, Python/GPU details, Docker base
digest, and observed model revisions:

- Base digest:
  `sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.
- SD1.4: `133a221b8aa7292a167afc5127cb63fb5005638b`.
- CLIP: `32bd64288804d66eefd0ccbe215aa642df71cc41`.

These are **audit-verified observations**, not claims that historical experiments
pinned these revisions. No model-loading behavior or scientific environment was
changed. The existing exact GPU parity result remains evidence:
`0.20923097431659698`, difference `0.0`. It was not rerun for these fixes.

Four original warning categories remain documented and non-blocking for the
supported production/reporting delivery:

| Warning | Remaining limitation |
| --- | --- |
| W2 | The tested environment is now recorded, but it is not an installation lock; future unpinned packages/models and rendering stacks may differ. Historical extraction provenance remains incomplete. |
| W5 | Some historical notebooks need unavailable helpers/templates/tensors; they are explicitly research records, not supported standalone execution entry points. |
| W6 | The saved smoke harness assumes its original cache/output paths and remains archival. |
| W7 | Inherited Ninja wheel metadata still causes the documented `pip check` warning; no replacement was made. |

No remaining blocker was found in the current supported workflow. The delivery
verdict does not promise arbitrary future unpinned-environment byte equality or
standalone execution of every historical notebook.

## Files modified or created

| Path | Action | Purpose |
| --- | --- | --- |
| `README.md` | Modified | Portable setup, reporting reproduction, archival scope and tested environment |
| `results/reporting/README.md` | Modified | Provenance split, allowlist, rebuild/validation and EOL behavior |
| `results/reporting/generate_report.py` | Modified | Portable verification, fixed inventory, preserved historical metadata and text comparison |
| `results/reporting/validate_reporting.py` | Modified | Portable input checks, fixed inventory, EOL-safe comparison, fresh stdout report |
| `results/reporting/input_manifest.json` | Created | 21 explicit Git-derived required inputs and separate historical inventory hashes |
| `results/reporting/reporting_inputs.py` | Created | Shared committed-content verification without runtime Git dependency |
| `results/reporting/test_reporting_inputs.py` | Created | Seven meaningful provenance/EOL/overwrite regression checks |
| `results/reporting/reproducibility_environment.json` | Created | Self-contained audit-observed environment, base digest and model revisions |
| `results/reproducibility_audit_after_fixes.md` | Created | This after-fixes audit and delivery evidence |

`results/reproducibility_audit.md` was already untracked at the start of this
task and remains unchanged; it is included among the documentation proposed for
eventual delivery. No canonical generated scientific artifact was created or
modified. Temporary validation outputs exist only in the separate clone.

**Final result: 0 remaining blockers; B1/B2/W1/W3/W4 resolved; complete reporting
regeneration and validation PASS; 9/9 original regression tests PASS; protected
files unchanged; READY FOR DELIVERY pending review and an explicitly authorized
commit/push.**
