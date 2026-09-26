"""Independently check frozen-source numbers and rebuild reports on CPU."""

from __future__ import annotations

import ast
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from reporting_inputs import inventory_digest, text_bytes, verify_inputs

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def assert_close(actual, expected, tolerance=1e-12):
    assert math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance), (actual, expected)


def main():
    import pandas as pd
    import torch
    from PIL import Image

    assert not torch.cuda.is_initialized()
    inputs = verify_inputs()
    final = pd.read_csv(ROOT / "results/final_test_results.csv", float_precision="round_trip")
    assert len(final) == 70 and not final.duplicated(["concept", "method", "seed"]).any()
    source_summary = final.groupby(["concept", "method"]).clip_score.agg(["mean", "std", "count"])
    main_table = pd.read_csv(HERE / "main_results.csv", float_precision="round_trip").set_index("concept")
    selected = pd.read_csv(ROOT / "results/selected_ranks.csv").set_index("concept").selected_rank
    numeric_checks = 0
    for concept, row in main_table.iterrows():
        assert row.fixed_rank == 1 and row.adaptive_rank == selected[concept]
        for method, label in (("fixed_rank_1", "fixed"), ("adaptive", "adaptive")):
            source = source_summary.loc[(concept, method)]
            assert source["count"] == 5
            for statistic in ("mean", "std"):
                assert_close(row[f"{label}_target_clip_{statistic}"], source[statistic])
                numeric_checks += 1
        assert_close(row.relative_improvement_pct,
                     100 * (row.fixed_target_clip_mean - row.adaptive_target_clip_mean) / row.fixed_target_clip_mean)
        numeric_checks += 1
    assert main_table.loc["car", "relative_improvement_pct"] < 0
    aggregate = pd.read_csv(HERE / "aggregate_results.csv").set_index("group")
    for group, selection in (
        ("all_concepts", main_table), ("changed_rank", main_table[main_table.adaptive_rank != 1]),
        ("style", main_table[main_table.category == "Style"]),
        ("non_style", main_table[main_table.category != "Style"]),
    ):
        fixed = selection.fixed_target_clip_mean.mean()
        adaptive = selection.adaptive_target_clip_mean.mean()
        assert aggregate.loc[group, "n_concepts"] == len(selection)
        for column, expected in (("fixed_target_clip_mean", fixed), ("adaptive_target_clip_mean", adaptive),
                                 ("relative_improvement_pct", 100 * (fixed - adaptive) / fixed)):
            assert_close(aggregate.loc[group, column], expected)
            numeric_checks += 1
    ranks = pd.read_csv(HERE / "rank_selection.csv").set_index("concept")
    for concept, row in ranks.iterrows():
        path = ROOT / "results/subspaces" / (concept.lower().replace(" ", "_") + ".pt")
        cached = torch.load(path, map_location="cpu", weights_only=True)
        assert_close(row.pc1_energy, cached["ratio"][0].item())
        for threshold, value in cached["threshold_ranks"].items():
            assert row[f"rank_{int(threshold * 100)}"] == value
        numeric_checks += 5
    notebook = json.loads((ROOT / "experiments/adaptive_rank_experiment.ipynb").read_text(encoding="utf-8"))
    # Cross-check against a DIFFERENT saved location: literal tradeoff_data in
    # cell82, rather than the cell77/81 printed tables used by the builder.
    source_tree = ast.parse("".join(notebook["cells"][82]["source"]))
    assignment = next(n for n in source_tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "tradeoff_data" for t in n.targets))
    style = pd.read_csv(HERE / "style_tradeoff.csv").set_index(["concept", "rank"])
    for key, value in zip(assignment.value.keys, assignment.value.values):
        concept = ast.literal_eval(key)
        columns = ast.literal_eval(value.args[0])
        for index, rank in enumerate(columns["rank"]):
            row = style.loc[(concept, rank)]
            assert_close(row.target_residual, columns["target_residual"][index])
            assert_close(row.preservation_degradation_pct, columns["preservation_drop_pct"][index])
            for budget in (10, 15, 20):
                assert bool(row[f"satisfies_{budget}pct_budget"]) == (max(0, row.preservation_degradation_pct) <= budget)
            numeric_checks += 5
    pres = pd.read_csv(ROOT / "results/van_gogh_preservation.csv", float_precision="round_trip")
    for rank, method in ((1, "fixed_rank_1"), (4, "adaptive_rank_4")):
        assert_close(style.loc[("Vincent van Gogh", rank), "preservation_score"],
                     pres.loc[pres.method == method, "clip_score"].mean())
        numeric_checks += 1
    for _, row in style.iterrows():
        recomputed = 100 * (row.original_preservation_score - row.preservation_score) / row.original_preservation_score
        # Saved Monet means have only six decimals; this bound reflects that
        # display precision and is not an experimental threshold.
        assert_close(recomputed, row.preservation_degradation_pct, tolerance=0.0003)
        numeric_checks += 1
    budget_table = pd.read_csv(HERE / "preservation_budget_sensitivity.csv")
    saved_budget_text = "".join("".join(o.get("text", [])) for o in notebook["cells"][82]["outputs"])
    for row in budget_table.itertuples():
        block = saved_budget_text.split(f"===== {row.concept} =====")[1].split("=====")[0]
        saved = dict((int(b), int(r)) for b, r in re.findall(r"Preservation budget (\d+)% -> selected rank = (\d+)", block))
        assert row.selected_rank == saved[row.budget_pct]
        numeric_checks += 1
    evidence = json.loads((HERE / "notebook_evidence.json").read_text(encoding="utf-8"))
    source95 = "".join("".join(o.get("text", [])) for o in notebook["cells"][95]["outputs"])
    assert_close(evidence["car_rank4_test"]["mean"], float(re.search(r"(?m)^Mean:\s*([0-9.]+)", source95)[1]))
    assert_close(evidence["car_rank4_test"]["std"], float(re.search(r"(?m)^Std:\s*([0-9.]+)", source95)[1]))
    assert evidence["car_rank4_test"]["mean"] > main_table.loc["car", "fixed_target_clip_mean"]
    numeric_checks += 2
    inventory = pd.read_csv(HERE / "data_inventory.csv").fillna("")
    files_in_results = {relative for relative in inputs if relative.startswith("results/")}
    assert set(inventory.source_file) == set(inputs)
    for row in inventory.itertuples():
        assert row.sha256 == inventory_digest(inputs[row.source_file])
    assert not (inventory.partition == "unclassified").any()
    image_metadata = []
    for path in sorted((HERE / "figures").glob("*.png")):
        with Image.open(path) as image:
            assert image.width >= 2800 and min(image.info.get("dpi", (0, 0))) >= 299
            image_metadata.append(dict(file=path.name, width=image.width, height=image.height,
                                       dpi=list(image.info["dpi"])))
    assert len(image_metadata) == 5
    rebuilt = []
    # Rebuild into a fresh temporary child of reporting, compare every artifact,
    # then clean only that temporary child. No existing output is overwritten.
    with tempfile.TemporaryDirectory(prefix=".validation-rebuild-", dir=HERE) as directory:
        temporary = Path(directory)
        subprocess.run([sys.executable, "-B", str(HERE / "generate_report.py"), "--output-dir", directory], check=True)
        subprocess.run([sys.executable, "-B", str(HERE / "generate_figures.py"), "--tables-dir", directory,
                        "--output-dir", str(temporary / "figures")], check=True)
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                relative = path.relative_to(temporary)
                original = (HERE / relative).read_bytes()
                # Git may convert text checkouts to CRLF. PNGs remain byte-exact.
                if path.suffix in (".csv", ".md", ".json"):
                    original = text_bytes(original)
                assert path.read_bytes() == original, relative
                rebuilt.append(relative.as_posix())
    assert len(rebuilt) == 20
    assert not torch.cuda.is_initialized()
    assert verify_inputs() == inputs
    for path in HERE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
        assert not any(name.startswith(("diffusers", "transformers", "src.generation", "src.evaluation")) for name in imports)
    report = dict(status="PASS", independent_numeric_checks=numeric_checks,
        committed_inputs_verified=len(inputs), main_source_rows=70, style_rows=8,
        budget_rows=6, existing_result_artifacts_inventoried=len(files_in_results),
        rebuilt_artifacts_identical=rebuilt, figures=image_metadata,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], cuda_initialized=False,
        gpu_experiments_launched=0, notebook_cells_executed=0,
        precision_note="Monet rounded aggregates checked at saved precision; no unavailable digits reconstructed")
    # validation_report.json is the original local audit record, not a template
    # to overwrite with today's verification of the portable input manifest.
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
