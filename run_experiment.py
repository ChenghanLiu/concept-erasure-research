"""Run the recorded calibration, final-test, or Van Gogh preservation protocol."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from src import protocol
from src.rank_selector import load_selected_ranks, select_adaptive_rank
from src.results import CsvResults, open_run, resolve_run_dir, sha256

REPOSITORY = Path(__file__).resolve().parent
CSV_FIELDS = {
    "calibration": ("concept", "rank", "seed", "clip_score"),
    "test": ("concept", "method", "rank", "seed", "clip_score"),
    "preservation": ("target_concept", "prompt", "seed", "method", "clip_score"),
}
CSV_NAMES = {
    "calibration": "calibration_results.csv",
    "test": "final_test_results.csv",
    "preservation": "van_gogh_preservation.csv",
}


def build_plan(phase: str, concepts: list[str], ranks: dict) -> dict[str, list[dict]]:
    """Preserve notebook iteration order, including baseline score reuse."""
    plans = {}
    if phase in ("calibration", "all"):
        plans["calibration"] = [
            {"concept": concept, "rank": rank, "seed": seed}
            for concept in concepts
            for rank in protocol.CANDIDATE_RANKS
            for seed in protocol.CALIBRATION_SEEDS
        ]
    if phase in ("test", "all"):
        plans["test"] = [
            {"concept": concept, "method": method, "rank": rank, "seed": seed}
            for concept in concepts
            for seed in protocol.TEST_SEEDS
            for method, rank in (("fixed_rank_1", 1), ("adaptive", ranks[concept]))
        ]
    if phase in ("preservation", "all") and "Vincent van Gogh" in concepts:
        plans["preservation"] = [
            {
                "target_concept": "Vincent van Gogh", "prompt": prompt,
                "seed": seed, "method": method,
            }
            for prompt in protocol.PRESERVATION_PROMPTS
            for seed in protocol.PRESERVATION_SEEDS
            for method in ("original", "fixed_rank_1", "adaptive_rank_4")
        ]
    return plans


def input_paths(concepts: list[str]) -> dict[str, Path]:
    relative = ["1000_short_t2i_prompts.txt", "results/selected_ranks.csv"]
    relative.extend(
        f"results/subspaces/{concept.lower().replace(' ', '_')}.pt"
        for concept in concepts
    )
    paths = {name: REPOSITORY / name for name in relative}
    for name, path in paths.items():
        if not path.resolve().is_relative_to(REPOSITORY):
            raise ValueError(f"Input resolves outside the repository: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Required recorded input is missing: {name}")
    return paths


def build_manifest(phase: str, concepts: list[str], ranks: dict, inputs: dict) -> dict:
    """Capture immutable scientific settings, code, input hashes and versions."""
    versions = {}
    for package in ("torch", "diffusers", "transformers", "numpy", "pillow", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    sources = [REPOSITORY / "run_experiment.py"] + [
        REPOSITORY / "src" / f"{name}.py"
        for name in (
            "protocol", "embedding", "projection", "generation", "evaluation",
            "rank_selector", "results",
        )
    ]
    return {
        "format_version": 1,
        "phase": phase,
        "concepts": concepts,
        "protocol": {
            "model": protocol.MODEL_NAME,
            "clip_model": protocol.CLIP_MODEL_NAME,
            "pipeline_dtype": "float16",
            "device": "cuda",
            "safety_checker": None,
            "scheduler_and_guidance": "StableDiffusionPipeline pretrained defaults",
            "negative_prompt": protocol.NEGATIVE_PROMPT,
            "candidate_ranks": list(protocol.CANDIDATE_RANKS),
            "calibration_seeds": list(protocol.CALIBRATION_SEEDS),
            "test_seeds": list(protocol.TEST_SEEDS),
            "preservation_seeds": list(protocol.PRESERVATION_SEEDS),
            "preservation_prompts": list(protocol.PRESERVATION_PROMPTS),
            "inference_steps": protocol.NUM_INFERENCE_STEPS,
            "lambda": protocol.LAMBDA_VALUE,
            "layer": protocol.LAYER_ID,
            "embedding_templates": protocol.NUM_EMBEDDING_PROMPTS,
            "recorded_selected_ranks": ranks,
            "prompts": protocol.CONCEPT_PROMPTS,
            "metric": "mean normalized CLIP image/text cosine similarity",
            "latent_shape": [1, 4, 64, 64],
            "images_per_job": 1,
            "adaptive_rank_one": "copy baseline score",
        },
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "source_sha256": {
            path.relative_to(REPOSITORY).as_posix(): sha256(path) for path in sources
        },
        "environment": {"python": platform.python_version(), "packages": versions},
    }


def save_calibration_selection(run_dir: Path, rows: list[dict], concepts: list[str]):
    """Report new calibration choices without replacing final-test selections."""
    selections = [
        {"concept": concept, "selected_rank": select_adaptive_rank(rows, concept)}
        for concept in concepts
    ]
    path = run_dir / "calibration_selected_ranks.csv"
    fields = ["concept", "selected_rank"]
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            previous = list(reader)
            expected = [{k: str(v) for k, v in row.items()} for row in selections]
            if reader.fieldnames != fields or previous != expected:
                raise ValueError("Existing calibration selections disagree with completed rows")
    else:
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selections)
    print("Calibration selections:", selections, flush=True)
    if any(row["selected_rank"] != protocol.SELECTED_RANKS[row["concept"]] for row in selections):
        print("New calibration differs from the record; final tests still use recorded ranks.", flush=True)


def execute_plans(plans: dict, stores: dict, analyses: dict, ranks: dict, *, evaluator=None):
    """Run only missing samples; injection supports offline resume checks."""
    scorer = None
    for phase, jobs in plans.items():
        store = stores[phase]
        # A copied rank-one row must be backed by exactly the baseline score.
        if phase == "test":
            for job in jobs:
                if job["method"] == "adaptive" and job["rank"] == 1:
                    adaptive = store.get(job)
                    baseline = store.get({**job, "method": "fixed_rank_1"})
                    if adaptive is not None and (
                        baseline is None or adaptive["clip_score"] != baseline["clip_score"]
                    ):
                        raise ValueError("Adaptive rank-one checkpoint must copy its baseline")
        for index, job in enumerate(jobs, start=1):
            if store.get(job) is not None:
                continue
            if phase == "test" and job["method"] == "adaptive" and job["rank"] == 1:
                baseline = store.get({**job, "method": "fixed_rank_1"})
                score = baseline["clip_score"]
            else:
                if phase == "preservation":
                    concept = job["target_concept"]
                    prompt = job["prompt"]
                    rank = {"original": None, "fixed_rank_1": 1, "adaptive_rank_4": 4}[job["method"]]
                else:
                    concept = job["concept"]
                    prompt = protocol.CONCEPT_PROMPTS[concept]
                    rank = job["rank"]
                if evaluator is None:
                    from src.evaluation import CLIPScorer, evaluate_single

                    if scorer is None:
                        scorer = CLIPScorer(device="cuda")
                    score = evaluate_single(
                        analyses[concept]["U"], rank, prompt, prompt,
                        job["seed"], scorer, device="cuda",
                    )
                else:
                    score = evaluator(concept, rank, prompt, job["seed"])
            store.append({**job, "clip_score": score})
            print(f"{phase} [{index}/{len(jobs)}] {job} CLIP={score:.6f}", flush=True)
        print(f"{phase}: {len(store.rows)}/{len(jobs)} rows complete", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "test", "preservation", "all"), default="test")
    parser.add_argument("--concept", nargs="+", choices=list(protocol.CONCEPT_PROMPTS), help="Subset of recorded concepts")
    parser.add_argument("--run-dir", help="New directory below results/runs/ (default: UTC timestamp)")
    parser.add_argument("--resume", action="store_true", help="Resume an existing runner-created --run-dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate input paths and print the fixed plan without loading models or writing files")
    args = parser.parse_args(argv)
    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir")
    requested = args.concept or (
        ["Vincent van Gogh"] if args.phase == "preservation" else list(protocol.CONCEPT_PROMPTS)
    )
    if len(requested) != len(set(requested)):
        parser.error("--concept must not contain duplicates")
    concepts = [concept for concept in protocol.CONCEPT_PROMPTS if concept in requested]
    if args.phase == "preservation" and concepts != ["Vincent van Gogh"]:
        parser.error("The recorded preservation comparison is for Vincent van Gogh only")
    try:
        run_name = args.run_dir or (
            "results/runs/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        )
        run_dir = resolve_run_dir(REPOSITORY, run_name)
        inputs = input_paths(concepts)
        ranks = load_selected_ranks(inputs["results/selected_ranks.csv"])
        plans = build_plan(args.phase, concepts, ranks)
        if args.dry_run:
            print(json.dumps({
                "run_dir": str(run_dir), "concepts": concepts,
                "rows": {phase: len(jobs) for phase, jobs in plans.items()},
                "candidate_ranks": protocol.CANDIDATE_RANKS,
                "calibration_seeds": protocol.CALIBRATION_SEEDS,
                "test_seeds": protocol.TEST_SEEDS,
                "preservation_seeds": protocol.PRESERVATION_SEEDS,
                "selected_ranks": ranks, "inference_steps": protocol.NUM_INFERENCE_STEPS,
            }, indent=2))
            return 0

        # Keep new Hugging Face downloads within the repository's existing cache.
        os.environ.setdefault("HF_HOME", str(REPOSITORY / "hf-cache"))
        import torch
        from src.embedding import cache_filename, load_cached_subspace

        manifest = build_manifest(args.phase, concepts, ranks, inputs)
        analyses = {
            concept: load_cached_subspace(
                REPOSITORY / "results" / "subspaces" / cache_filename(concept), concept
            )
            for concept in concepts
        }
        with open_run(run_dir, manifest, resume=args.resume):
            stores = {
                phase: CsvResults(run_dir / CSV_NAMES[phase], CSV_FIELDS[phase], jobs)
                for phase, jobs in plans.items()
            }
            if any(len(stores[phase].rows) != len(jobs) for phase, jobs in plans.items()):
                if not torch.cuda.is_available():
                    raise RuntimeError("The recorded fp16 protocol requires CUDA; use --dry-run for a CPU-only plan")
            execute_plans(plans, stores, analyses, ranks)
            if "calibration" in stores:
                save_calibration_selection(run_dir, list(stores["calibration"].rows.values()), concepts)
        print(f"Results: {run_dir}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, csv.Error) as exc:
        parser.exit(1, f"Experiment stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
