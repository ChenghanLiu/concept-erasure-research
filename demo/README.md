# Concept erasure demo

Docker Compose is the primary supported execution path. Python is required
inside the container only; no Python installation on the Windows host is
assumed. Run these commands from the repository root.

Ensure the existing Compose service is running before executing or validating
the demo:

```powershell
docker compose up -d
docker compose ps concept-erasure
```

Run the demo through the container:

```powershell
docker compose exec concept-erasure python demo/run_demo.py --concept "Vincent van Gogh"
```

Run without generation:

```powershell
docker compose exec concept-erasure python demo/run_demo.py --concept "Vincent van Gogh" --no-generate
```

Docker mounts the current repository at `/workspace`. Generation requires the
configured NVIDIA GPU support. Use these Compose commands for validation;
do not substitute a host `python demo/run_demo.py` invocation.

The normal mode generates a comparison using the first recorded held-out seed,
**3025**: fixed rank **1** versus the recorded adaptive rank (**4** for Van Gogh).
It reuses the production Stable Diffusion v1.4 and CLIP code, the cached subspace,
the original prompt/negative prompt, **15 steps**, projection lambda **4**, and
text-encoder layer **0**. Each generated method uses a fresh pipeline. Concepts
whose adaptive rank is 1 reuse the fixed result, matching the original protocol.
This is a single-seed illustration, not a replacement for the five-seed experiment.

The command prints a new `results/runs/demo-<UTC timestamp>/` directory containing
`index.html`, the generated PNGs, `demo_results.csv`, `summary.json`, and
`manifest.json`. Open `index.html` on the Windows host for the side-by-side view.
Each invocation creates a new directory; existing results are never overwritten.
The minimal demo does not resume an interrupted run; invoke it again for a new
directory. First generation may download models into the repository's ignored
`hf-cache/`; existing Hugging Face environment settings are respected.

`--no-generate` only prints the frozen five-seed mean and sample standard
deviation from `results/final_test_results.csv`, with recorded ranks. It loads
no models, needs no GPU or model cache, makes no downloads, and writes no files.
It does not pretend that reporting charts are generated sample images.

For first-time Docker setup, run `docker compose build` before
`docker compose up -d`. See the [root README](../README.md) for Docker/GPU setup
and the full recorded experiment workflow.
