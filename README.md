# Concept erasure research

Production code extracted from `experiments/adaptive_rank_experiment.ipynb`.
The notebooks remain experiment records. Existing files in `results/`, cached
subspaces, and baseline artifacts are read-only inputs to this runner.

## Recorded experiment

The runner preserves these settings; they are not command-line tuning options:

| Setting | Recorded value |
| --- | --- |
| Generator | `CompVis/stable-diffusion-v1-4`, CUDA, float16, `safety_checker=None` |
| Projection | `P = I - 4 U_r U_r.T`; `W_new = W_original @ P` |
| Projection location | Text encoder layer **0**, `self_attn.out_proj.weight`; bias unchanged |
| Subspace extraction | First **20** nonempty lines of `1000_short_t2i_prompts.txt` |
| Activations | Input to the layer's output projection; unpadded tokens including special tokens |
| SVD | Uncentered `torch.svd(X.T)` on CPU float32, embedding width 768 |
| Candidate ranks | **1, 2, 4, 8** |
| Calibration seeds | **2025, 2026, 2027** |
| Final test seeds | **3025, 3026, 3027, 3028, 3029** |
| Generation | One image per sample, **15** inference steps, explicit `(1,4,64,64)` latent |
| Scheduler / guidance | Pretrained Stable Diffusion pipeline defaults |
| Metric | Normalized image/text cosine similarity from `openai/clip-vit-large-patch14` |
| Calibration selection | Lowest mean CLIP similarity across the three calibration seeds |

The original negative prompt is stored verbatim in `src/protocol.py`. Each score
uses a fresh diffusion pipeline, so projections never accumulate between ranks.
The generation helper retains the notebook's 50-step default and `seed + i*100`
rule for multiple images; the experiment runner explicitly uses one image and
15 steps. Generation and CLIP scoring use the same prompt in the final protocols:

| Concept | Prompt | Recorded adaptive rank |
| --- | --- | --- |
| Taylor Swift | a portrait of Taylor Swift | 2 |
| Vincent van Gogh | a painting in the style of Vincent van Gogh | 4 |
| Claude Monet | a painting in the style of Claude Monet | 4 |
| car | a photo of a car | 2 |
| airplane | a photo of an airplane | 1 |
| dog | a photo of a dog | 1 |
| cat | a photo of a cat | 2 |

Final tests compare fixed rank 1 with those recorded selections from
`results/selected_ranks.csv`. Airplane and dog reuse the baseline score as their
adaptive score, exactly as the notebook does. A complete test produces **70 CSV
rows from 60 generated images**. A fresh calibration produces **84 rows**.
Its computed choices are reported separately and never replace the recorded
final-test ranks, even if numerical differences change the new calibration winner.

The `preservation` phase reproduces the recorded Van Gogh comparison: original
model, fixed rank 1, and adaptive rank 4. It uses seeds **4025, 4026, 4027** and
four prompts: `a photo of a dog`, `a photo of a car`, `a photo of an airplane`,
and `a portrait of a person`. Scoring uses each preservation prompt itself.
This produces **36 rows**. The `all` phase produces **190 rows** across the three
protocols for all seven concepts.

## Setup

Run commands from this repository root. The existing Docker environment is the
recommended reproduction environment: its base image supplies PyTorch
`2.5.1` with CUDA `12.4` and installs `requirements.txt`.
An NVIDIA GPU and NVIDIA-compatible Docker GPU support are required for inference.

For a local Python environment, use Python 3.11, install a CUDA-compatible
PyTorch build, then install the existing requirements:

```powershell
cd D:\Projects\concept-erasure
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_experiment.py --help
```

`requirements.txt` does not install PyTorch; install it in that environment
before running the experiments. Model access and enough GPU memory for SD1.4
and the CLIP evaluator are needed. Model loading may download weights on first
use. The runner defaults `HF_HOME` to the repository's `hf-cache/` if it is unset;
existing Hugging Face environment settings are honored. It does not clear proxy
settings or force the notebook's optional Hugging Face mirror.

The existing dependencies are unpinned. The working container used for the
refactor checks had Python 3.11.10, torch 2.5.1+cu124, diffusers 0.40.0, and
transformers 5.16.1. Each new run records its installed versions and input/code
hashes in `manifest.json`. Historical caches do not contain model revisions or
extraction provenance, so bitwise equality across environments is not asserted.

## Docker usage

The existing `Dockerfile` and `docker-compose.yml` are unchanged. Compose mounts
this repository at `/workspace`, mounts `hf-cache/` as the Hugging Face cache,
requests one NVIDIA GPU, and exposes Jupyter on port 8888.

```powershell
docker compose build
docker compose up -d
docker compose exec concept-erasure python run_experiment.py --help
docker compose exec concept-erasure python run_experiment.py --phase all --dry-run
```

For the existing notebook interface, use the local Jupyter address and token
reported by `docker compose logs concept-erasure`. The runner does not execute,
rewrite, or clear notebook cells.

## Reproduction

Inspect the plan without loading models, using a GPU, or creating output files:

```powershell
python run_experiment.py --phase all --dry-run
```

Run the final test with the seven cached subspaces and original selected ranks:

```powershell
python run_experiment.py --phase test --run-dir results/runs/final-reproduction
```

Run fresh calibration and the original preservation comparison separately:

```powershell
python run_experiment.py --phase calibration --run-dir results/runs/calibration-reproduction
python run_experiment.py --phase preservation --run-dir results/runs/preservation-reproduction
```

Run all three protocols, or select a subset of the recorded concepts:

```powershell
python run_experiment.py --phase all --run-dir results/runs/full-reproduction
python run_experiment.py --phase test --concept "Vincent van Gogh" car --run-dir results/runs/two-concepts
```

Prefix any of these Python commands with `docker compose exec concept-erasure`
to run them in Docker. Paths are relative to the repository root regardless of
the shell's working directory. If `--run-dir` is omitted, a new UTC timestamped
directory is created under `results/runs/`. `all` includes preservation only
when Vincent van Gogh is among the selected concepts; `preservation` itself
accepts only Vincent van Gogh.

### Resume

Repeat the same phase, concept subset, and run directory with `--resume`:

```powershell
python run_experiment.py --phase test --run-dir results/runs/final-reproduction --resume
```

Each completed score is appended and flushed to disk before the next sample.
Resume validates the manifest and CSV schema, skips completed jobs, and checks
rank/seed/method keys. It rejects duplicate, incomplete, nonfinite, or conflicting
rows instead of treating them as completed results. A killed in-flight sample
is regenerated. A torn CSV write is reported and left untouched for inspection.

A directory must be new unless `--resume` is supplied. Resumption requires the
same code, protocol, input hashes, and recorded dependency versions. A `.lock`
file prevents concurrent writers. If a process is forcibly killed, remove that
run's stale `.lock` only after confirming the process is no longer active.

Historical result files cannot be selected as destinations or resumed in place.
Only runner-created directories beneath `results/runs/` are eligible. Start a
new run for reproduction; the original CSVs remain available for comparison.

## Code and artifacts

| Path | Purpose |
| --- | --- |
| `src/embedding.py` | Activation extraction, uncentered SVD, validated cached `.pt` loading |
| `src/projection.py` | Exact projection construction and weight update; both CLIP encoder layouts supported |
| `src/generation.py` | Fresh SD1.4 loading and explicit seeded latent generation |
| `src/evaluation.py` | CLIP cosine scoring and single-sample evaluation |
| `src/rank_selector.py` | Complete calibration validation and recorded-rank loading |
| `src/protocol.py` | Recorded prompts, ranks, seeds, and constants |
| `src/results.py` | New run directories, manifests, locks, and resumable CSV storage |
| `src/checks.py` | Offline parity and checkpoint checks |
| `src/__init__.py` | Research package marker |
| `run_experiment.py` | Command-line orchestration |
| `experiments/`, root notebooks | Historical experiment records, unchanged |
| `baseline/` | Baseline reproduction artifacts, unchanged |
| `results/subspaces/*.pt` | Existing CPU SVD cache records, read-only |
| `results/pilot_results.csv` | Early pilot data, read-only |
| `results/selected_ranks.csv` | Original seven selected ranks, read-only |
| `results/final_test_results.csv` | Original 70-row test, read-only |
| `results/van_gogh_preservation.csv` | Original 36-row preservation comparison, read-only |
| `results/spectrum_summary.csv` | Original spectral summary, read-only |
| `results/runs/<run>/` | New reproduction outputs only |

New run outputs are `manifest.json` and the selected phase's
`calibration_results.csv`, `final_test_results.csv`, and/or
`van_gogh_preservation.csv`. Calibration also creates
`calibration_selected_ranks.csv`. Scores retain the historical `clip_score`
column name; it represents cosine similarity. Generated images are evaluated
in memory; this runner writes no image galleries or replacement subspaces.

Cache records contain `concept`, `U`, `S`, `ratio`, `cumulative`, and
`threshold_ranks`. Loading uses CPU placement and `weights_only=True`; the runner
requires existing valid caches and never silently recomputes them. For an
explicit in-memory extraction using the original procedure:

```python
from src.embedding import analyze_spectrum, get_embedding
from src.generation import load_pipeline

pipe = load_pipeline()
embedding = get_embedding(pipe, "1000_short_t2i_prompts.txt", "car")
analysis = analyze_spectrum(embedding, "car")
# Existing results/subspaces/car.pt is not written or replaced.
```

### Notebook review notes

The notebook contains exploratory cells and depends on execution history.
The refactor follows embedding/SVD cells 1–2 and 35, cache schema cell 55,
calibration cell 45, final-test cells 52 and 56–57, and preservation cells 65–71
(zero-based indices).

`pilot_results.csv` contains 27 early rows, including Van Gogh rank 7, and is
not a complete `{1,2,4,8}` calibration matrix. Earlier Taylor Swift pilots also
use a different scoring prompt. Later cells explore a preservation budget and
car rank 4; those exploratory choices do not replace the recorded car rank 2
in the final experiment. The runner avoids these historical ambiguities while
leaving every notebook and result intact.

## Verification

Run syntax, import, and offline parity checks in the configured environment:

```powershell
python -m compileall -q src run_experiment.py
python -c "import src.embedding, src.projection, src.generation, src.evaluation, src.rank_selector; import run_experiment"
python -m src.checks
```

The checks load the existing small subspace caches and compare generation,
projection, and scoring against function definitions extracted from the notebook
using test doubles. They also exercise rank validation and CSV resumption.
They do not download models, generate experiment images, or write historical
results. Temporary fixtures stay under `src/` and are cleaned up.
