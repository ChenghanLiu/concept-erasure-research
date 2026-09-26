"""One frozen-protocol comparison, or a read-only view of frozen results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import protocol
from src.rank_selector import load_selected_ranks


def frozen_results(concept, adaptive_rank):
    """Read the original five-seed test without loading models or writing files."""
    source = ROOT / "results/final_test_results.csv"
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["concept"] == concept]
    expected = {(method, seed) for method in ("fixed_rank_1", "adaptive") for seed in protocol.TEST_SEEDS}
    keys = [(row["method"], int(row["seed"])) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"Incomplete or duplicate frozen final-test rows for {concept}")
    for row in rows:
        expected_rank = 1 if row["method"] == "fixed_rank_1" else adaptive_rank
        if int(row["rank"]) != expected_rank or not math.isfinite(float(row["clip_score"])):
            raise ValueError(f"Invalid frozen rank or score for {concept}")
    summary = []
    for method, rank in (("fixed_rank_1", 1), ("adaptive", adaptive_rank)):
        selected = [row for row in rows if row["method"] == method]
        scores = [float(row["clip_score"]) for row in selected]
        summary.append({
            "method": method, "rank": rank, "mean": statistics.mean(scores),
            "sample_std": statistics.stdev(scores), "seeds": list(protocol.TEST_SEEDS),
            "first_seed_score": next(float(row["clip_score"]) for row in selected
                                     if int(row["seed"]) == protocol.TEST_SEEDS[0]),
        })
    return summary


def show_frozen(concept, summary):
    print(f"Concept: {concept}")
    print(f"Prompt: {protocol.CONCEPT_PROMPTS[concept]}")
    print("Frozen held-out results: mean +/- sample SD across seeds "
          + ", ".join(map(str, protocol.TEST_SEEDS)))
    print("CLIP cosine similarity to the target prompt; lower means less target similarity.")
    for row in summary:
        print(f"  {row['method']} (rank {row['rank']}): {row['mean']:.6f} +/- {row['sample_std']:.6f}")


def write_gallery(path, concept, rows, frozen):
    cards = []
    for row in rows:
        cards.append(
            f"<article><h2>{html.escape(row['method'])} · rank {row['rank']}</h2>"
            f"<img src='{row['image']}' alt='{html.escape(concept, quote=True)} comparison'>"
            f"<p>Generated CLIP: {row['clip_score']:.6f}<br>"
            f"Frozen same-seed CLIP: {row['frozen_same_seed_score']:.6f}</p></article>"
        )
    frozen_text = " · ".join(f"{row['method']}: {row['mean']:.6f}" for row in frozen)
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(concept)} — concept erasure demo</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;background:#f4f5f7;color:#17212b}}h1{{line-height:1.2}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px}}article{{background:white;padding:20px;border:1px solid #dce1e6;border-radius:12px}}img{{width:100%;height:auto}}h2{{font-size:20px}}.note{{color:#425466}}</style>
<h1>{html.escape(concept)}</h1><p>{html.escape(protocol.CONCEPT_PROMPTS[concept])}</p>
<p class="note">Single-seed illustration: seed {protocol.TEST_SEEDS[0]}, {protocol.NUM_INFERENCE_STEPS} inference steps.
CLIP measures similarity to the target prompt; lower means less target similarity.
This illustration does not replace the frozen five-seed experiment.</p>
<main class="cards">{''.join(cards)}</main>
<p>Frozen five-seed means: {html.escape(frozen_text)}.</p>
<p><a href="demo_results.csv">Generated scores</a> · <a href="manifest.json">Run manifest</a> · <a href="summary.json">Comparison details</a></p></html>
"""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(document)


def generate_demo(concept, ranks, frozen):
    # This branch alone imports GPU/model dependencies. Existing HF settings win.
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    import torch
    import run_experiment as runner
    from src.embedding import cache_filename, load_cached_subspace
    from src.evaluation import CLIPScorer, evaluate_single
    from src.results import CsvResults, open_run, resolve_run_dir, sha256

    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires CUDA in the Compose container; use --no-generate for frozen results")
    seed = protocol.TEST_SEEDS[0]
    plan = [row for row in runner.build_plan("test", [concept], ranks)["test"] if row["seed"] == seed]
    inputs = runner.input_paths([concept])
    inputs["results/final_test_results.csv"] = ROOT / "results/final_test_results.csv"
    manifest = runner.build_manifest("demo_single_test_seed", [concept], ranks, inputs)
    manifest["source_sha256"]["demo/run_demo.py"] = sha256(Path(__file__))
    manifest["demo"] = {"jobs": plan, "scope": "single-seed illustration; not a full experiment"}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = resolve_run_dir(ROOT, f"results/runs/demo-{timestamp}")
    cache = ROOT / "results/subspaces" / cache_filename(concept)
    analysis = load_cached_subspace(cache, concept)
    print(f"Loaded cached subspace: {cache.relative_to(ROOT)} (no SVD recomputation)", flush=True)
    print(f"Generating one pair at seed {seed}, {protocol.NUM_INFERENCE_STEPS} steps; "
          f"lambda {protocol.LAMBDA_VALUE}, layer {protocol.LAYER_ID}.", flush=True)
    rows = []
    with open_run(output, manifest):
        print(f"Output directory: {output.relative_to(ROOT)}", flush=True)
        store = CsvResults(output / "demo_results.csv", runner.CSV_FIELDS["test"], plan)
        scorer = CLIPScorer(device="cuda")

        class SaveScorer:
            """Capture the image at the existing evaluator's scoring boundary."""
            image_path = None

            def score(self, images, prompt):
                if len(images) != 1:
                    raise ValueError("The frozen demo requires exactly one image per sample")
                score = scorer.score(images, prompt)
                with self.image_path.open("xb") as handle:
                    images[0].save(handle, format="PNG")
                return score

        capture = SaveScorer()
        prompt = protocol.CONCEPT_PROMPTS[concept]
        for job in plan:
            method, rank = job["method"], job["rank"]
            image_name = f"{method}.png"
            if method == "adaptive" and rank == 1:
                # The original final-test protocol reuses the rank-one baseline.
                score = rows[0]["clip_score"]
                image_name = rows[0]["image"]
            else:
                capture.image_path = output / image_name
                score = evaluate_single(analysis["U"], rank, prompt, prompt, seed, capture, device="cuda")
            store.append({**job, "clip_score": score})
            reference = next(row["first_seed_score"] for row in frozen if row["method"] == method)
            rows.append({**job, "clip_score": score, "image": image_name,
                         "frozen_same_seed_score": reference, "difference_from_frozen": score - reference})
            print(f"{method}, rank {rank}: CLIP={score!r}; frozen same-seed={reference!r}", flush=True)
        with (output / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump({"concept": concept, "prompt": prompt, "scope": manifest["demo"]["scope"],
                       "frozen_five_seed_summary": frozen, "generated": rows}, handle, indent=2)
            handle.write("\n")
        write_gallery(output / "index.html", concept, rows, frozen)
    print(f"Open on the host: {(output / 'index.html').relative_to(ROOT)}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", choices=list(protocol.CONCEPT_PROMPTS), default="Vincent van Gogh")
    parser.add_argument("--no-generate", action="store_true", help="Print frozen results only; no models, GPU, downloads, or output files")
    args = parser.parse_args(argv)
    try:
        ranks = load_selected_ranks(ROOT / "results/selected_ranks.csv")
        frozen = frozen_results(args.concept, ranks[args.concept])
        show_frozen(args.concept, frozen)
        if args.no_generate:
            print("No generation requested; no models loaded or files written.")
            return 0
        generate_demo(args.concept, ranks, frozen)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, csv.Error) as error:
        parser.exit(1, f"Demo stopped: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
