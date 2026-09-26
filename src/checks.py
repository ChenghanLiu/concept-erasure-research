"""Offline regression checks: ``python -m src.checks``.

These checks use recorded notebook functions, cached subspaces, and small CPU
fakes. They never load model weights or generate experiment results. Temporary
CSV fixtures are created under src/ and removed after each check.
"""

from __future__ import annotations

import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from .embedding import cache_filename, load_cached_subspace
from .evaluation import clip_car_scores
from .generation import generate_img
from .projection import build_projection, project_text_conder_attn
from .protocol import CALIBRATION_SEEDS, CANDIDATE_RANKS, SELECTED_RANKS
from .rank_selector import select_adaptive_rank
from .results import CsvResults, open_run, resolve_run_dir


REPOSITORY = Path(__file__).resolve().parents[1]


def notebook_function(name):
    """Compile just one named function, without executing notebook cells."""
    path = REPOSITORY / "experiments" / "adaptive_rank_experiment.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][1]["source"]))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one notebook definition of {name}")
    module = ast.Module(body=matches, type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, f"{path}:cell1", "exec"), namespace)
    return namespace[name]


class RecordingPipeline:
    def __init__(self):
        self.unet = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float32,
            config=SimpleNamespace(in_channels=4),
        )
        self.calls = []
        self.progress = None

    def set_progress_bar_config(self, **kwargs):
        self.progress = kwargs

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return ([f"image-{len(self.calls)}"],)


class ProcessorInputs(dict):
    def to(self, device):
        self.device = device
        return self


class RecordingProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return ProcessorInputs()


class FakeCLIP:
    def __init__(self, pooled):
        self.pooled = pooled

    def features(self, values):
        tensor = torch.tensor(values, dtype=torch.float32)
        return SimpleNamespace(pooler_output=tensor) if self.pooled else tensor

    def get_text_features(self, **kwargs):
        return self.features([[3, 4, 0], [0, 2, 1]])

    def get_image_features(self, **kwargs):
        return self.features([[4, 3, 2], [1, 0, 2]])


class NotebookEquivalenceChecks(unittest.TestCase):
    def test_generation_latents_seeds_and_unseeded_arguments(self):
        original = notebook_function("generate_img")
        for seed, count in ((2025, 3), (None, 1)):
            with self.subTest(seed=seed):
                reference, current = RecordingPipeline(), RecordingPipeline()
                kwargs = dict(
                    prompt="a photo of a car", num_img=count,
                    negative_prompt="negative", seed=seed, num_inference_steps=15,
                )
                self.assertEqual(original(reference, **kwargs), generate_img(current, **kwargs))
                self.assertEqual(reference.progress, current.progress)
                self.assertEqual(len(current.calls), count)
                for index, (before, after) in enumerate(zip(reference.calls, current.calls)):
                    self.assertEqual(before.keys(), after.keys())
                    for key in before:
                        if key == "latents":
                            self.assertTrue(torch.equal(before[key], after[key]))
                            expected = torch.randn(
                                (1, 4, 64, 64),
                                generator=torch.Generator(device="cpu").manual_seed(seed + index * 100),
                                dtype=torch.float32,
                            )
                            self.assertTrue(torch.equal(after[key], expected))
                        else:
                            self.assertEqual(before[key], after[key])
        pipeline = RecordingPipeline()
        generate_img(pipeline, "prompt", num_img=1)
        self.assertEqual(pipeline.calls[0]["num_inference_steps"], 50)

    def test_clip_cosine_and_pooler_compatibility(self):
        original = notebook_function("clip_car_scores")
        for pooled in (False, True):
            with self.subTest(pooler_output=pooled):
                before, after = RecordingProcessor(), RecordingProcessor()
                model = FakeCLIP(pooled)
                images, prompts = ["image-a", "image-b"], ["a car", "a vehicle"]
                expected = original(model, before, "cpu", images, prompts)
                actual = clip_car_scores(model, after, "cpu", images, prompts)
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(actual.shape, (2,))
                self.assertEqual(actual.device.type, "cpu")
                self.assertEqual(before.calls, after.calls)

    def test_all_cached_subspaces_and_candidate_projections(self):
        self.assertEqual(len(SELECTED_RANKS), 7)
        for concept in SELECTED_RANKS:
            cache = load_cached_subspace(
                REPOSITORY / "results" / "subspaces" / cache_filename(concept), concept
            )
            basis = cache["U"]
            self.assertEqual(basis.dtype, torch.float32)
            for rank in CANDIDATE_RANKS:
                with self.subTest(concept=concept, rank=rank):
                    retained = basis[:, :rank]
                    expected = torch.eye(basis.shape[0]) - 4 * torch.matmul(retained, retained.T)
                    self.assertTrue(torch.equal(build_projection(basis, rank), expected))

    def test_right_multiplication_fp16_conversion_and_unchanged_bias(self):
        original = notebook_function("project_text_conder_attn")
        basis = load_cached_subspace(REPOSITORY / "results/subspaces/car.pt", "car")["U"]
        projection = build_projection(basis, 2)
        generator = torch.Generator().manual_seed(123)
        weight = torch.randn(16, 768, generator=generator).to(torch.float16)
        bias = torch.randn(16, generator=generator).to(torch.float16)

        def pipe_with_layer():
            layer = torch.nn.Linear(768, 16, dtype=torch.float16)
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            pipe = SimpleNamespace(text_encoder=SimpleNamespace(encoder=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=SimpleNamespace(out_proj=layer))]
            )))
            return pipe, layer

        reference, before = pipe_with_layer()
        current, after = pipe_with_layer()
        original(reference, projection, layer_id=0)
        project_text_conder_attn(current, projection, layer_id=0)
        self.assertTrue(torch.equal(before.weight, after.weight))
        self.assertTrue(torch.equal(after.weight, weight @ projection.to(torch.float16)))
        self.assertTrue(torch.equal(after.bias, bias))
        self.assertEqual(after.weight.dtype, torch.float16)


class RankSelectionChecks(unittest.TestCase):
    @staticmethod
    def rows(scores):
        return [
            {"concept": "car", "rank": rank, "seed": seed, "clip_score": scores[rank]}
            for rank in CANDIDATE_RANKS for seed in CALIBRATION_SEEDS
        ]

    def test_complete_calibration_and_lowest_rank_tie(self):
        rows = self.rows({1: 0.4, 2: 0.1, 4: 0.2, 8: 0.3})
        self.assertEqual(len(rows), 12)
        self.assertEqual(select_adaptive_rank(reversed(rows), "car"), 2)
        self.assertEqual(select_adaptive_rank(self.rows(dict.fromkeys(CANDIDATE_RANKS, 0.2)), "car"), 1)

    def test_incomplete_duplicate_and_nonfinite_calibration_rejected(self):
        rows = self.rows(dict.fromkeys(CANDIDATE_RANKS, 0.2))
        for invalid in (rows[:-1], rows + [rows[0]], [dict(rows[0], clip_score=float("nan"))] + rows[1:]):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                select_adaptive_rank(invalid, "car")


class CheckpointChecks(unittest.TestCase):
    fields = ("concept", "rank", "seed", "clip_score")
    jobs = [{"concept": "car", "rank": 2, "seed": 3025}]
    row = dict(jobs[0], clip_score=0.25)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".checks-", dir=REPOSITORY / "src")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_historical_and_outside_paths_rejected(self):
        for path in (REPOSITORY, REPOSITORY.parent, "results", "results/subspaces", "results/runs", "results/runs/../../experiments"):
            with self.subTest(path=str(path)), self.assertRaises(ValueError):
                resolve_run_dir(REPOSITORY, path)
        self.assertEqual(
            resolve_run_dir(REPOSITORY, "results/runs/check-name"),
            REPOSITORY / "results/runs/check-name",
        )

    def test_append_resume_and_manifest(self):
        path = resolve_run_dir(self.root, "results/runs/example")
        manifest = {"protocol": "recorded", "phase": "test"}
        with open_run(path, manifest):
            result = CsvResults(path / "scores.csv", self.fields, self.jobs)
            self.assertIsNone(result.get(self.jobs[0]))
            result.append(self.row)
            content = result.path.read_bytes()
            with self.assertRaises(ValueError):
                result.append(self.row)
            self.assertEqual(result.path.read_bytes(), content)
        with open_run(path, manifest, resume=True):
            result = CsvResults(path / "scores.csv", self.fields, self.jobs)
            self.assertEqual(result.get(self.jobs[0]), self.row)
        with self.assertRaises(ValueError), open_run(path, {"protocol": "changed"}, resume=True):
            pass
        self.assertFalse((path / ".lock").exists())
        with self.assertRaises(FileExistsError), open_run(path, manifest):
            pass

    def test_invalid_checkpoint_rows_rejected(self):
        cases = {
            "wrong-header": "concept,rank,seed,score\n",
            "wrong-rank": "concept,rank,seed,clip_score\ncar,4,3025,0.25\n",
            "malformed": "concept,rank,seed,clip_score\ncar,2,3025\n",
            "extra-field": "concept,rank,seed,clip_score\ncar,2,3025,0.25,extra\n",
            "partial": "concept,rank,seed,clip_score\ncar,2,3025,0.25",
            "nonfinite": "concept,rank,seed,clip_score\ncar,2,3025,nan\n",
            "duplicate": "concept,rank,seed,clip_score\ncar,2,3025,0.25\ncar,2,3025,0.25\n",
        }
        for name, contents in cases.items():
            with self.subTest(case=name):
                path = self.root / f"{name}.csv"
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    CsvResults(path, self.fields, self.jobs)
        path = self.root / "append.csv"
        result = CsvResults(path, self.fields, self.jobs)
        for invalid in (dict(self.row, rank=4), dict(self.row, rank=2.5), dict(self.row, clip_score=float("inf"))):
            with self.assertRaises(ValueError):
                result.append(invalid)
        with path.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
