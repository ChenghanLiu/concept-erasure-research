"""One frozen-condition GPU check through the production runner; no batch."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from functools import wraps
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HOME"] = str(ROOT / "hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_CACHE_DISABLE"] = "1"
os.environ["TMPDIR"] = str(HERE / "tmp")
(HERE / "tmp").mkdir(exist_ok=True)

WRITES = set()


def audit_writes(event, args):
    paths = []
    if event == "open":
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            paths = [path]
    elif event in ("os.mkdir", "os.remove", "os.rmdir"):
        paths = [args[0]]
    elif event in ("os.rename", "os.link", "os.symlink"):
        paths = [args[0], args[1]]
    for value in paths:
        if not isinstance(value, (str, bytes, os.PathLike)):
            continue
        path = Path(os.fsdecode(value)).resolve()
        if str(path) == "/dev/null":
            continue
        if not path.is_relative_to(ROOT / "results" / "runs"):
            raise RuntimeError(f"Smoke test attempted a write outside results/runs: {path}")
        WRITES.add(str(path.relative_to(ROOT)))


sys.addaudithook(audit_writes)

import torch
import run_experiment as runner
from src import embedding, evaluation, generation, protocol
from src.projection import get_target_projection_layer
from src.rank_selector import load_selected_ranks
from src.results import CsvResults, open_run, resolve_run_dir, sha256


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def notebook_function(name):
    notebook = json.loads((ROOT / "experiments/adaptive_rank_experiment.ipynb").read_text())
    tree = ast.parse("".join(notebook["cells"][1]["source"]))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "frozen-notebook-cell1", "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available()
    assert protocol.NUM_INFERENCE_STEPS == 15
    assert protocol.LAMBDA_VALUE == 4 and protocol.LAYER_ID == 0
    assert protocol.MODEL_NAME == "CompVis/stable-diffusion-v1-4"
    assert protocol.CLIP_MODEL_NAME == "openai/clip-vit-large-patch14"
    assert protocol.CONCEPT_PROMPTS["car"] == "a photo of a car"
    frozen_reference = 0.209231
    absolute_tolerance = 0.00001
    with (ROOT / "results/pilot_results.csv").open(newline="") as handle:
        frozen_rows = [row for row in csv.DictReader(handle)
                       if (row["concept"], row["rank"], row["seed"]) == ("car", "1", "2025")]
    assert len(frozen_rows) == 1
    frozen_exact = float(frozen_rows[0]["clip_score"])
    inputs = runner.input_paths(["car"])
    ranks = load_selected_ranks(inputs["results/selected_ranks.csv"])
    full_plan = runner.build_plan("calibration", ["car"], ranks)
    job = {"concept": "car", "rank": 1, "seed": 2025}
    plans = {"calibration": [row for row in full_plan["calibration"] if row == job]}
    assert plans == {"calibration": [job]}
    manifest = runner.build_manifest("single_sample_gpu_parity", ["car"], ranks, inputs)
    manifest["smoke_test"] = {
        "jobs": plans, "frozen_reference": frozen_reference,
        "absolute_tolerance": absolute_tolerance,
        "harness_sha256": sha256(Path(__file__)),
        "gpu": torch.cuda.get_device_name(0),
    }
    run_dir = resolve_run_dir(ROOT, HERE / "checkpoint")
    cache = ROOT / "results/subspaces/car.pt"
    trace = {"resume": args.resume, "gpu": torch.cuda.get_device_name(0),
             "frozen_csv_score": frozen_exact}
    # Both SVD entry points fail immediately if anything tries to recompute a basis.
    with patch.object(torch, "svd", side_effect=AssertionError("SVD recomputation forbidden")), \
         patch.object(torch.linalg, "svd", side_effect=AssertionError("SVD recomputation forbidden")):
        analysis = embedding.load_cached_subspace(cache, "car")
        trace["cache"] = {"path": str(cache.relative_to(ROOT)), "sha256": sha256(cache),
                          "U_shape": list(analysis["U"].shape), "recomputed": False}
        original_load = generation.load_pipeline
        original_generate = generation.generate_img
        original_project = evaluation.project_text_conder_attn
        original_score = evaluation.CLIPScorer.score
        old_clip = notebook_function("clip_car_scores")
        old_generate = notebook_function("generate_img")

        def checked_load(device="cuda"):
            trace["pipeline_load_count"] = trace.get("pipeline_load_count", 0) + 1
            pipe = original_load(device)
            trace["pipeline"] = {"class": type(pipe).__name__, "model": protocol.MODEL_NAME,
                "dtype": str(pipe.unet.dtype), "device": str(pipe.unet.device),
                "scheduler": type(pipe.scheduler).__name__, "safety_checker": pipe.safety_checker,
                "text_encoder_class": type(pipe.text_encoder).__name__}
            return pipe

        def checked_project(pipe, proj_matrix, layer_id=0):
            target = get_target_projection_layer(pipe, layer_id)
            reference_matrix = torch.eye(768) - 4 * (analysis["U"][:, :1] @ analysis["U"][:, :1].T)
            assert torch.equal(proj_matrix, reference_matrix)
            with torch.no_grad():
                expected_weight = target.weight.clone() @ reference_matrix.to(target.weight.device).to(target.weight.dtype)
                before_bias = target.bias.detach().clone()
                original_project(pipe, proj_matrix, layer_id=layer_id)
                assert torch.equal(target.weight, expected_weight)
                assert torch.equal(target.bias, before_bias)
            trace["projection"] = {"rank": 1, "lambda": 4, "layer_id": layer_id,
                "module": "text_encoder.encoder.layers[0].self_attn.out_proj"
                    if hasattr(pipe.text_encoder, "encoder")
                    else "text_encoder.text_model.encoder.layers[0].self_attn.out_proj",
                "matrix_matches_notebook": True, "weight_matches_notebook": True,
                "bias_unchanged": True}

        def checked_generate(pipe, prompt, **kwargs):
            assert prompt == "a photo of a car"
            assert kwargs == {"num_img": 1, "negative_prompt": protocol.NEGATIVE_PROMPT,
                              "seed": 2025, "num_inference_steps": 15}

            class CaptureOnly:
                unet = pipe.unet
                def set_progress_bar_config(self, **kwargs):
                    pass
                def __call__(self, **call_kwargs):
                    self.kwargs = call_kwargs
                    return ([None],)

            # Execute the notebook latent-generation function on the same GPU,
            # but use a capture-only pipeline so this produces no second image.
            capture = CaptureOnly()
            old_generate(capture, prompt, **kwargs)
            notebook_latent = capture.kwargs["latents"]
            pipe_type = type(pipe)
            real_call = pipe_type.__call__

            @wraps(real_call)
            def checked_call(instance, *call_args, **call_kwargs):
                assert instance is pipe and not call_args
                assert set(call_kwargs) == set(capture.kwargs)
                for key, value in capture.kwargs.items():
                    if key == "latents":
                        assert torch.equal(value, call_kwargs[key])
                    else:
                        assert value == call_kwargs[key]
                trace["generation"] = {"seed": 2025, "num_inference_steps": call_kwargs["num_inference_steps"],
                    "prompt": call_kwargs["prompt"], "latent_shape": list(notebook_latent.shape),
                    "latent_dtype": str(notebook_latent.dtype), "latent_device": str(notebook_latent.device),
                    "latent_sha256": tensor_hash(notebook_latent), "notebook_latent_exact_match": True,
                    "negative_prompt_exact_match": True, "generator_passed_to_pipeline": "generator" in call_kwargs}
                result = real_call(instance, *call_args, **call_kwargs)
                trace["generation"]["images_generated"] = len(result.images)
                return result

            with patch.object(pipe_type, "__call__", checked_call):
                return original_generate(pipe, prompt, **kwargs)

        def checked_score(scorer, images, prompt):
            score = original_score(scorer, images, prompt)
            notebook_score = old_clip(scorer.model, scorer.processor, scorer.device, images, prompt).mean().item()
            assert score == notebook_score
            trace["clip"] = {"model": protocol.CLIP_MODEL_NAME, "score": score,
                "frozen_function_same_image_score": notebook_score, "exact_scoring_match": True,
                "processor": type(scorer.processor).__name__,
                "image_processor": type(scorer.processor.image_processor).__name__,
                "image_processor_config": scorer.processor.image_processor.to_dict(),
                "model_dtype": str(next(scorer.model.parameters()).dtype),
                "model_training": scorer.model.training, "scoring_prompt": prompt}
            return score

        with open_run(run_dir, manifest, resume=args.resume):
            path = run_dir / runner.CSV_NAMES["calibration"]
            store = CsvResults(path, runner.CSV_FIELDS["calibration"], plans["calibration"])
            before_hash = sha256(path)
            with patch.object(generation, "load_pipeline", checked_load), \
                 patch.object(generation, "generate_img", checked_generate), \
                 patch.object(evaluation, "project_text_conder_attn", checked_project), \
                 patch.object(evaluation.CLIPScorer, "score", checked_score):
                runner.execute_plans(plans, {"calibration": store}, {"car": analysis}, ranks)
            assert len(store.rows) == 1
            score = store.get(job)["clip_score"]
            after_hash = sha256(path)
            if args.resume:
                assert trace.get("pipeline_load_count", 0) == 0
                assert before_hash == after_hash
            trace["checkpoint"] = {"rows": 1, "before_sha256": before_hash,
                "after_sha256": after_hash, "resume_unchanged": args.resume and before_hash == after_hash,
                "pipeline_load_count": trace.get("pipeline_load_count", 0)}

    delta = score - frozen_reference
    trace.update({"clip_score": score, "reference": frozen_reference, "signed_difference": delta,
                  "difference_from_frozen_csv": score - frozen_exact,
                  "absolute_difference": abs(delta), "absolute_tolerance": absolute_tolerance,
                  "gpu_parity": "PASS" if abs(delta) <= absolute_tolerance else "FAIL",
                  "write_paths": sorted(WRITES)})
    report_path = HERE / ("resume_report.json" if args.resume else "report.json")
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: trace[key] for key in (
        "clip_score", "reference", "signed_difference", "absolute_difference", "gpu_parity", "checkpoint"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
