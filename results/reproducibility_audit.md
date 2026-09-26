# Clean-clone reproducibility audit

**Overall verdict: NOT YET READY FOR DELIVERY**

**2 BLOCKER findings; 7 WARNING findings.** The production experiment path passes
offline checks and one exact GPU parity check. Complete reporting regeneration
fails because its required snapshot contains unavailable files and hashes that
do not describe the committed repository. No fixes were implemented.

## Scope and provenance

- Audit dates: 2026-09-25 to 2026-09-26 America/Toronto; final verification at
  2026-09-26 04:00 UTC.
- Canonical repository: `D:\Projects\concept-erasure`.
- Origin: `https://github.com/ChenghanLiu/concept-erasure-research.git`.
- Audited commit: `41763a5bcfa38065f85c61b2bceb3dd24e1b21f7`.
- Commit subject: `Freeze final experiments and reporting artifacts`.
- Local HEAD and origin `refs/heads/main` were verified equal using
  `git rev-parse HEAD` and `git ls-remote origin refs/heads/main`.
- Retained clean clone: `D:\Projects\concept-erasure-clean-audit-20260925`.
- The clone came directly from origin. No project files, model caches,
  credentials, or untracked/ignored sources were copied from the canonical tree.
- LF checkout was selected explicitly at clone time to inspect Git blob bytes
  without this Windows host's global `core.autocrlf=true` conversion. This is
  equivalent to the relevant text representation in a normal Linux checkout;
  no notebook or source file was subsequently normalized or rewritten.
- A new Docker image was built from the clone's unchanged Dockerfile and
  requirements, with `--pull --no-cache`. The previously existing project image
  and container were not used. Only the clean clone was bind-mounted.
- Offline checks ran with Docker networking disabled. The GPU attempt used a
  newly populated clone-local Hugging Face cache and anonymous model downloads.
- All 70 tracked clone files were checked against Git blob hashes with zero
  byte differences. Tracked source, notebooks, frozen results, and reporting
  artifacts remained unchanged. Audit harnesses are additional diagnostic files,
  not project fixes. The only canonical-tree addition is this report.
- No batch GPU experiment was run. Exactly one image was generated, followed
  by a resume that generated zero additional images. No methodology was tuned.

## PASS: required Git contents

| Component | Result |
| --- | --- |
| `src/`, `run_experiment.py` | Present, including offline regression checks |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `README.md` | Present |
| Current extraction templates | `1000_short_t2i_prompts.txt` present |
| Cached subspaces | All seven required `.pt` files present and loadable |
| Frozen result CSVs | Final test, selected ranks, pilot, spectra, and Van Gogh preservation present |
| Research notebook | Full 97-cell `experiments/adaptive_rank_experiment.ipynb` present |
| Reporting package | All 29 committed reporting files present |
| Reporting scripts | `generate_report.py`, `generate_figures.py`, `report_narratives.py`, `validate_reporting.py` present |

Missing historical-notebook dependencies are distinguished from the current
workflow in WARNING W5. Missing reporting preconditions are BLOCKER B1.

## Environment actually tested

| Item | Observed value |
| --- | --- |
| Host GPU | NVIDIA GeForce RTX 2070 SUPER, 8 GiB |
| Host driver | 616.64 |
| Container Python | 3.11.10 |
| PyTorch / torchvision | 2.5.1+cu124 / 0.20.1+cu124 |
| diffusers / transformers | 0.40.0 / 5.17.0 |
| accelerate / huggingface_hub | 1.15.0 / 1.33.0 |
| NumPy / pandas | 2.1.2 / 3.0.6 |
| Matplotlib / Pillow | 3.11.2 / 10.2.0 |
| Base image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| Resolved base digest | `sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755` |
| Audit image tag | `concept-erasure-clean-audit:41763a5` |
| Audit image identity from Docker inspect | `sha256:213a99b281ff28f4edb1c0d24f620e682d23c54f6dcf0e8b89e9e0245c81be51` |

Full installed package versions and build output are retained in the clone's
`results/runs/clean-clone-audit/offline_results.json` and `docker-build.log`.

## Offline execution results

| Check | Classification | Observation |
| --- | --- | --- |
| Fresh Docker build | PASS | Unchanged Dockerfile built successfully using external registry/PyPI downloads |
| Compose configuration | PASS | Bind paths resolve to the clean clone, with the declared GPU request |
| Syntax compilation | PASS | `src`, runner, and reporting Python files compiled |
| Imports | PASS | Production modules, runner, SD pipeline class, CLIP classes, pandas, Matplotlib, Pillow imported offline |
| `python -m src.checks` | PASS | All 9 tests passed |
| CLI help and full dry-run | PASS | 84 calibration, 70 final-test, 36 preservation rows; ranks/seeds unchanged; 15 steps |
| Seven cached subspaces | PASS | CPU float32 loading succeeded with both SVD entry points disabled |
| API checkpoint/resume | PASS | Synthetic 1-row checkpoint completed its 9 missing jobs; second resume skipped all jobs and preserved CSV bytes |
| CLI checkpoint/resume | PASS | Completed synthetic checkpoint resumed from `/tmp`, with repository mounted at `/audit-clone` |
| Invalid output destinations | PASS | Five destinations outside permitted run children rejected |
| `python -m pip check` | WARNING | Exit 1: `ninja 1.11.1.1 is not supported on this platform`; see W7 |
| Reporting table/narrative rebuild | BLOCKER | Exit 1 before tables were generated; see B1/B2 |
| Reporting validator | BLOCKER | Exit 1 at the same initial snapshot check |
| Figure-only rebuild | PASS | All five PNGs byte-identical and pixel-identical to committed figures |

The nine committed regression tests cover notebook-equivalent seeded latents
and generation arguments, CLIP scoring, all cache/rank projections, projection
weight/bias handling, rank selection completeness/ties, output boundaries,
append/resume manifests, and invalid checkpoint rows.

The checkpoint fixtures use explicitly synthetic scores, not research results.
They reside only in the audit's `results/runs/clean-clone-audit/` subtree. The
CLI test rejected `results`, `results/subspaces`, `results/runs`, `../outside`,
and `/tmp/outside`, all with exit 1. Changing the mount and current directory
confirmed production paths do not depend on the literal `/workspace` location.

### Cached subspaces

| Concept | Cache | U shape |
| --- | --- | --- |
| Taylor Swift | `results/subspaces/taylor_swift.pt` | 768 x 167 |
| Vincent van Gogh | `results/subspaces/vincent_van_gogh.pt` | 768 x 187 |
| Claude Monet | `results/subspaces/claude_monet.pt` | 768 x 167 |
| car | `results/subspaces/car.pt` | 768 x 147 |
| airplane | `results/subspaces/airplane.pt` | 768 x 147 |
| dog | `results/subspaces/dog.pt` | 768 x 147 |
| cat | `results/subspaces/cat.pt` | 768 x 147 |

Loading used `load_cached_subspace`, CPU placement, and `weights_only=True`.
`torch.svd` and `torch.linalg.svd` were patched only in the audit harness to
raise if called; all seven loads succeeded. No SVD was recomputed.

## Reporting regeneration and comparison

Both committed entry points failed at the first unavailable snapshot file:

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/workspace/.~code_for_zexian.ipynb'
```

An independent audit of all 48 snapshot entries found **34 matching files,
7 missing files, and 7 hash mismatches**. The historical committed
`validation_report.json` records a prior local validation; it does not establish
clean-clone reproducibility.

No snapshot verification was bypassed, no missing source was fabricated, and
no full table/narrative rebuild is claimed. A figure-only rebuild using the
committed reporting CSVs succeeded in a new output directory:

| Figure | Byte comparison | Pixel comparison |
| --- | --- | --- |
| `fixed_vs_adaptive_by_concept.png` | Identical | Identical |
| `relative_improvement_by_concept.png` | Identical | Identical |
| `preservation_budget_selection.png` | Identical | Identical |
| `style_rank_tradeoff_monet.png` | Identical | Identical |
| `style_rank_tradeoff_van_gogh.png` | Identical | Identical |

The remaining 15 generated table, narrative, inventory, and evidence artifacts
could not be regenerated through the official builder because it stopped at
input verification. There are no observed figure differences; comparison of
the complete 20-artifact rebuilt package is blocked.

## Findings requiring review

### BLOCKER B1: mandatory snapshot entries are absent from Git

**Exact cause:** `generate_report.py:69-74,313` and
`validate_reporting.py:38-39` unconditionally hash every historical protection
snapshot entry. Seven entries are ignored local artifacts rather than available
Git sources:

```text
.~code_for_zexian.ipynb
experiments/.ipynb_checkpoints/adaptive_rank_experiment-checkpoint.ipynb
experiments/temp_folder/0668f09a-b79c-11f1-b7cb-c20abd0a5b2d.png
experiments/temp_folder/0668f09a-b79c-11f1-b7cb-c20abd0a5b2d.pth
experiments/temp_folder/c01105e4-b7b7-11f1-b7cb-c20abd0a5b2d.png
experiments/temp_folder/c01105e4-b7b7-11f1-b7cb-c20abd0a5b2d.pth
Untitled.ipynb
```

**Affected paths:** `results/reporting/source_snapshot.json`, both reporting
entry points, and the seven paths above. `.gitignore` excludes these files.

**Fresh-user impact:** Every clean clone fails reporting generation and
validation before numerical calculations. These incidental files are not
otherwise consumed by the table calculations.

**Minimal recommended fix:** Preserve the historical protection snapshot as
evidence, but introduce a separate explicit manifest of committed inputs
actually required for regeneration. Do not copy private scratch files or
fabricate them merely to satisfy this check.

### BLOCKER B2: snapshot hashes describe stale/local bytes, not the commit

**Exact cause:** Six expected hashes match an in-memory LF-to-CRLF conversion
of the committed files, rather than their committed LF bytes:

```text
Dockerfile
docker-compose.yml
requirements.txt
results/runs/car-gpu-parity-20260925/checkpoint/calibration_results.csv
results/runs/car-gpu-parity-20260925/run.log
results/runs/car-gpu-parity-20260925/resume.log
```

The seventh mismatch is `.gitignore`. Its recorded hash matches the parent
commit's ten-line contents with mixed LF/CRLF endings. The audited commit adds
`results/reporting/.mplconfig/`, so neither LF nor CRLF current contents match.
This was established from Git objects and in-memory transformations only.

**Affected paths:** The seven files above including `.gitignore`,
`results/reporting/source_snapshot.json`, and raw-byte hash checks in
`generate_report.py` / `validate_reporting.py`. No `.gitattributes` defines a
portable checkout representation.

**Fresh-user impact:** LF clones encounter the six EOL mismatches; all fresh
clones encounter the stale `.gitignore` entry. Supplying B1's missing files
alone would not solve regeneration.

**Minimal recommended fix:** Separate archival protection hashes from the
portable regeneration manifest. Derive the latter from committed source bytes
with an explicit EOL policy, or verify Git blobs. Review provenance before
updating any manifest; do not normalize protected notebooks/results as a fix.

### WARNING W1: frozen reporting inventory expands with future runs

**Exact cause and paths:** `generate_report.py:270-292` scans every file beneath
`results/` except `results/reporting/`; `validate_reporting.py:121-124` requires
the current set to appear in the frozen inventory and disallows unclassified
rows. New run files change the regenerated inventory; a file such as this audit
report also introduces an unclassified entry.

**Fresh-user impact:** A pristine clone has the expected 20 nonreporting result
artifacts, but following the README's experiment commands creates more. Later
reporting validation then conflicts with the frozen inventory. The current
audit logs also expose this condition. This is a source-inspection finding;
the observed validator failure occurred earlier at B1.

**Minimal recommended fix:** Inventory an explicit frozen-source allowlist and
keep new reproduction/audit outputs outside that frozen report input set.

### WARNING W2: dependency and model revisions are not fixed

**Exact cause and paths:** All packages in `requirements.txt` are unpinned;
`Dockerfile:1` specifies a tag without a digest. `src/generation.py:15-19` and
`src/evaluation.py:57-58` use model identifiers without revision arguments.
`run_experiment.py:79-83,123` records only a subset of dependency versions and
does not record model revisions. Reporting requires exact PNG bytes without a
locked rendering environment.

**Fresh-user impact:** This build selected transformers 5.17.0, while the README
records 5.16.1 for the earlier environment. GPU parity and PNG equality passed
today, but later installations are not constrained to the same software or
weights. Historical SVD caches lack complete extraction revision provenance.

**Minimal recommended fix:** Publish a tested dependency lock, base-image
digest, rendering versions, and verified model revision IDs; record these in
future manifests. Preserve historical provenance limitations and methodology.

### WARNING W3: local setup command hardcodes the author's checkout path

**Exact cause and path:** `README.md:68` contains
`cd D:\Projects\concept-erasure`.

**Fresh-user impact:** A user cloning elsewhere, or using Linux/macOS, cannot
copy this setup command literally. The production code itself is portable.

**Minimal recommended fix:** Replace the literal setup destination with an
explicit clone-directory placeholder and platform-appropriate instructions.

### WARNING W4: root README omits reporting reproduction instructions

**Exact cause and paths:** Root `README.md` neither links to nor describes the
reporting package. Rebuild commands exist only in
`results/reporting/README.md:45-51`, targeting an existing `concept-erasure`
container at `/workspace`.

**Fresh-user impact:** Following the root README does not tell a user how to
regenerate the delivered reporting artifacts. On a host with multiple clones,
the fixed container name also requires ensuring the correct clone is mounted.

**Minimal recommended fix:** Link the reporting README from the root and give
the reporting sequence after Docker setup, clearly identifying the target clone.

### WARNING W5: historical notebook execution needs unavailable dependencies

**Exact cause and paths:** `code_for_zexian.ipynb:22` imports absent
`clip_score_cal`; `sec-5_style_monet.ipynb:19,62,172,236` imports absent
`unlearn_utils`, reads absent `prompt_template3.txt`, and loads ignored UUID
`.pth` files from `temp_folder/`. Several notebook setup cells force
`https://hf-mirror.com` and remove proxy environment settings. The baseline
notebook's relative template path assumes a repository-root working directory.

**Fresh-user impact:** Rerunning all historical notebooks from Git alone is
not supported. Current production CLI execution and figure generation do not
require those missing helpers/templates/tensors. The README already describes
notebooks as records and acknowledges execution-history dependence.

**Minimal recommended fix:** Document precisely which notebooks are archival
and not standalone reproductions. If complete historical execution is promised,
provide the genuine original dependencies and execution instructions without
rewriting protected notebook evidence.

### WARNING W6: archived GPU smoke script is not a reusable fresh-run command

**Exact cause and path:**
`results/runs/car-gpu-parity-20260925/smoke_test.py:19-24,104,227-229` forces
offline loading from clone-local `hf-cache/`, targets an already committed
checkpoint, and exclusively creates already committed report filenames.

**Fresh-user impact:** Direct execution on a clean clone cannot create a new
smoke run at that occupied destination; offline models are also absent. The
production CLI exposes whole phases, not a single-sample option. This audit
therefore used a new diagnostic harness calling the unchanged production
runner's one-job execution path.

**Minimal recommended fix:** Label the saved harness as historical evidence
and provide a separately reusable one-sample entry point with fresh output
paths and documented model preparation. Do not overwrite the saved run.

### WARNING W7: base-image Ninja metadata fails package consistency checking

**Exact cause and affected dependency:** In the freshly built Docker image,
`python -m pip check` exits 1 for inherited `ninja 1.11.1.1`. Its installed
`ninja-1.11.1.1.dist-info/WHEEL` has a blank line before its `Tag:` fields, so
header parsing returns no wheel tags. The raw text contains Linux-compatible
tags, but the installed metadata is not parsed as supported. The package was
already in the base environment, not installed by the project requirements.

**Fresh-user impact:** The audited base-image digest reproduces this package
check warning. Project imports, nine regression tests, reporting figure
rendering, and GPU execution nevertheless passed; no relevant functional
failure was observed.

**Minimal recommended fix:** Validate a base image or explicit Ninja wheel
installation with correct metadata, then record that environment and rerun
`pip check`. No package replacement was performed during this audit.

## External and local dependencies not supplied by Git

| Dependency | Required for / observed behavior |
| --- | --- |
| Git and origin access | Fetching the project; origin was reachable |
| Docker Engine/Desktop and Compose | Recommended setup; external installation required |
| Docker Hub base image | Downloaded/resolved from the registry; digest recorded above |
| Python 3.11 and CUDA-compatible PyTorch | Required separately for local setup; supplied by Docker base here |
| Python packages | Twelve direct requirements plus transitive dependencies; full resolved list retained in audit JSON |
| torchvision | Present in Docker base; historical notebooks import it; not explicitly in requirements |
| NVIDIA GPU, compatible driver, GPU container runtime | Required for recorded CUDA/float16 inference; CPU sufficient for offline checks and reporting |
| Hugging Face SD1.4 assets | Weights, scheduler/configuration, tokenizer and encoder assets downloaded from `CompVis/stable-diffusion-v1-4` |
| Hugging Face CLIP assets | Weights/tokenizer/processor downloaded from `openai/clip-vit-large-patch14` |
| Network and download storage | Docker/PyPI/Hugging Face access required for first setup; new model cache occupied about 5.98 GB in this audit |
| Hugging Face credentials | No credentials forwarded or required for the successful anonymous downloads; a user may supply an external token for limits/access policy |
| `HF_HOME` and related HF variables | Runner defaults to clone `hf-cache/`; explicitly set existing settings are honored |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | Used for offline tests; historical smoke script forces them and therefore assumes cached models |
| Proxy variables / `HF_ENDPOINT` | Production honors external settings; historical notebooks override proxies and force a mirror |
| `CUDA_VISIBLE_DEVICES`, Docker GPU exposure | Reporting disables CUDA; inference needs a visible CUDA device |
| `PYTHONDONTWRITEBYTECODE`, `CUDA_CACHE_DISABLE`, `TMPDIR` | Diagnostic/runtime cache controls; not scientific hyperparameters |
| `hf-cache/` | Ignored, regenerable external model cache; audit started with no cache and downloaded independently |
| `results/reporting/.mplconfig/` | Ignored, regenerable Matplotlib font cache; no prior cache required |
| Snapshot's seven ignored files | Genuine hidden reporting preconditions, unavailable from Git; B1 |
| Archived notebook helpers/template/tensors | Genuine historical execution dependencies missing from Git; W5 |

No project secret or credential file was supplied to the clean containers.
Ordinary external packages/models are expected setup dependencies, distinct
from the unavailable local artifacts that cause the blockers.

## Absolute-path audit

- `README.md:68` is an executable machine-specific setup example (W3).
- `/workspace` in Dockerfile, Compose, and reporting commands is an intentional
  bind-mount contract. It does not point back to the canonical repository.
- `/root/.cache/huggingface` in Compose is an intentional mount of that clone's
  `hf-cache/`; production's explicit `HF_HOME` also remains clone-local.
- Production input paths and reporting roots derive from `__file__`. The CLI
  passed from `/tmp` with a different mount root, `/audit-clone`.
- `/home/huiqiangchen/miniconda3/...` in the Monet notebook is saved warning
  output, not an executable source dependency.
- `D:\Projects\concept-erasure` strings in the archived smoke validation JSON
  are historical evidence keys, not runtime file reads.
- Missing relative local files and notebook working-directory assumptions are
  documented in B1/W5/W6; no undocumented canonical-path access was used.

## PASS: one real-GPU parity sample

| Field | Result |
| --- | --- |
| Concept / rank / seed | car / 1 / 2025 |
| Prompt | `a photo of a car` |
| Inference steps | Exactly 15 |
| Model | `CompVis/stable-diffusion-v1-4`, float16, PNDMScheduler |
| Projection | Lambda 4, text encoder layer 0 output projection; unchanged production code |
| Evaluator | `openai/clip-vit-large-patch14`, notebook-equivalent preprocessing and normalized cosine scoring |
| Cached subspace | Loaded `results/subspaces/car.pt`; SVD recomputation forbidden |
| Images generated | Exactly 1 |
| Produced CLIP score | **0.20923097431659698** |
| Frozen reference | **0.20923097431659698** |
| Signed difference | **0.0** |
| Parity | **PASS**, exact equality; diagnostic tolerance 0.00001 |
| Resume | PASS; no model/scoring call, no additional image, CSV bytes unchanged |

The audit compared the actual GPU latent and every generation argument with
the frozen notebook function using a capture-only pipeline, which generated
no second image. It also evaluated the same generated image with the frozen
notebook scoring function and obtained exactly the same score.

Resolved model revisions, recorded from the new cache's `refs/main`:

- SD1.4: `133a221b8aa7292a167afc5127cb63fb5005638b`.
- CLIP: `32bd64288804d66eefd0ccbe215aa642df71cc41`.

These are audit observations, not newly installed revision pins. The model
object itself did not expose a commit hash in the inspected UNet config.

## Commands and retained audit evidence

The initial verification used `git status --short`, `git rev-parse HEAD`,
`git remote -v`, `git branch -vv`, and `git ls-remote origin`. The canonical
working tree and index were clean at audit start.

```powershell
git -c core.autocrlf=false clone https://github.com/ChenghanLiu/concept-erasure-research.git D:\Projects\concept-erasure-clean-audit-20260925
cd D:\Projects\concept-erasure-clean-audit-20260925
docker build --pull --no-cache --progress plain -t concept-erasure-clean-audit:41763a5 .

docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -e HF_HOME=/workspace/hf-cache --mount 'type=bind,source=D:\Projects\concept-erasure-clean-audit-20260925,target=/workspace' -w /workspace concept-erasure-clean-audit:41763a5 python -B results/runs/clean-clone-audit/offline_audit.py

docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --mount 'type=bind,source=D:\Projects\concept-erasure-clean-audit-20260925,target=/audit-clone' -w /tmp concept-erasure-clean-audit:41763a5 python -B /audit-clone/results/runs/clean-clone-audit/cli_resume_audit.py

docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_DISABLE_PROGRESS_BARS=1 --mount 'type=bind,source=D:\Projects\concept-erasure-clean-audit-20260925,target=/workspace' -w /workspace concept-erasure-clean-audit:41763a5 python -B results/runs/clean-clone-audit/gpu_audit.py
```

Host command output was captured with `Tee-Object` in the audit log directory.
The offline harness ran the committed checks plus these unmodified reporting
entry points, capturing each exit code independently:

```text
python -m compileall -q src run_experiment.py results/reporting
python -B -m src.checks
python -B run_experiment.py --help
python -B run_experiment.py --phase all --dry-run
python -m pip check
python -B results/reporting/generate_report.py --output-dir results/reporting/.clean-clone-audit-rebuild
python -B results/reporting/validate_reporting.py
python -B results/reporting/generate_figures.py --output-dir results/reporting/.clean-clone-audit-figures
```

The expanded module-import command is retained verbatim in `offline_results.json`.
Audit harnesses add instrumentation, synthetic fixtures, and the user-requested
single-job plan; they do not replace project inputs or alter protected code.

Retained additions in the clean clone:

- `results/runs/clean-clone-audit/`: three audit harnesses, build/test/GPU logs,
  `offline_results.json`, `cli_resume_results.json`, `gpu_results.json`,
  `tracked-integrity.json`, two synthetic checkpoint directories, and the new
  one-row GPU checkpoint. Completed run locks were removed by normal runner exit.
- `results/reporting/.clean-clone-audit-figures/`: five new comparison PNGs.
- `results/reporting/.clean-clone-audit-rebuild/`: empty directory created before
  the blocked report build stopped.
- Ignored clone-local `hf-cache/`, Matplotlib font cache, and compilation
  bytecode caches. None was imported from the canonical repository.
- The newly built Docker image is retained. Temporary audit containers exited
  normally; the clone directory was not deleted.

Scientific run CSVs/manifests stayed under this clone's `results/runs/`.
Diagnostic figures are under `results/reporting/`, and model downloads under
`hf-cache/`; this is not a claim that all runtime cache writes are run outputs.
No canonical notebook, source, frozen result, or reporting artifact was edited.
No file was staged, committed, or pushed during the audit.

## Delivery decision

| Requested outcome | Audit result |
| --- | --- |
| Commit audited | `41763a5bcfa38065f85c61b2bceb3dd24e1b21f7`, verified pushed |
| Clean clone | `D:\Projects\concept-erasure-clean-audit-20260925`, retained |
| Offline test status | Syntax/imports and 9 regression checks PASS; package consistency WARNING |
| Reporting regeneration | BLOCKED; five figure-only rebuilds identical, full rebuild unavailable |
| Cached subspace status | PASS, 7/7, no SVD recomputation |
| GPU parity | PASS, exact score 0.20923097431659698, difference 0.0 |
| Resume/output paths | PASS with synthetic CLI checks and actual one-sample GPU resume |
| BLOCKER count | **2** |
| WARNING count | **7** |
| Overall verdict | **NOT YET READY FOR DELIVERY** |

The reporting blockers must be reviewed and corrected before Git-only delivery
can be claimed. All recommendations above remain proposals; no fixes were made.
