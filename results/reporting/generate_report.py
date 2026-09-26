"""Build frozen research tables and prose on CPU; never execute notebook cells.

All output is confined to results/reporting. Existing output is accepted only
when byte-identical, so reruns cannot silently replace a changed result.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics as stats
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
REPORTING = ROOT / "results/reporting"
NOTEBOOK = "experiments/adaptive_rank_experiment.ipynb"
CONCEPTS = ("Taylor Swift", "Vincent van Gogh", "Claude Monet", "car", "airplane", "dog", "cat")
CATEGORIES = dict(zip(CONCEPTS, ("Identity", "Style", "Style", "Object", "Object", "Animal", "Animal")))
RANKS = (1, 2, 4, 8)
SEEDS = (3025, 3026, 3027, 3028, 3029)
SELECTED = dict(zip(CONCEPTS, (2, 4, 4, 2, 1, 1, 2)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_directory(value):
    path = Path(value).resolve()
    if not path.is_relative_to(REPORTING.resolve()):
        raise ValueError("Reporting output must stay beneath results/reporting")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write(path, text):
    content = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite nonidentical artifact: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(content)


def write_csv(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write(path, buffer.getvalue())


def read_csv(relative):
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify_frozen_inputs():
    snapshot = json.loads((REPORTING / "source_snapshot.json").read_text())
    for path, expected in snapshot["files"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen source changed since the reporting audit: {path}")
    return len(snapshot["files"])


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]) + "\n"


def build_main():
    final = read_csv("results/final_test_results.csv")
    selected = {row["concept"]: int(row["selected_rank"]) for row in read_csv("results/selected_ranks.csv")}
    assert selected == SELECTED
    expected = {(c, m, s) for c in CONCEPTS for m in ("fixed_rank_1", "adaptive") for s in SEEDS}
    actual = [(r["concept"], r["method"], int(r["seed"])) for r in final]
    assert len(final) == 70 and len(set(actual)) == 70 and set(actual) == expected
    for row in final:
        assert int(row["rank"]) == (1 if row["method"] == "fixed_rank_1" else selected[row["concept"]])
        assert math.isfinite(float(row["clip_score"]))
    rows = []
    for concept in CONCEPTS:
        fixed = [float(r["clip_score"]) for r in final if r["concept"] == concept and r["method"] == "fixed_rank_1"]
        adaptive = [float(r["clip_score"]) for r in final if r["concept"] == concept and r["method"] == "adaptive"]
        fixed_mean, adaptive_mean = stats.mean(fixed), stats.mean(adaptive)
        rows.append(dict(
            concept=concept, category=CATEGORIES[concept], fixed_rank=1, adaptive_rank=selected[concept],
            fixed_target_clip_mean=fixed_mean, fixed_target_clip_std=stats.stdev(fixed),
            adaptive_target_clip_mean=adaptive_mean, adaptive_target_clip_std=stats.stdev(adaptive),
            relative_improvement_pct=100 * (fixed_mean - adaptive_mean) / fixed_mean,
            n_test_seeds=5, source_file="results/final_test_results.csv",
        ))
    aggregate = []
    for group, members in (
        ("all_concepts", rows),
        ("changed_rank", [r for r in rows if r["adaptive_rank"] != 1]),
        ("style", [r for r in rows if r["category"] == "Style"]),
        ("non_style", [r for r in rows if r["category"] != "Style"]),
    ):
        fixed = stats.mean(r["fixed_target_clip_mean"] for r in members)
        adaptive = stats.mean(r["adaptive_target_clip_mean"] for r in members)
        aggregate.append(dict(group=group, n_concepts=len(members), fixed_target_clip_mean=fixed,
            adaptive_target_clip_mean=adaptive, relative_improvement_pct=100 * (fixed - adaptive) / fixed,
            concepts="; ".join(r["concept"] for r in members), source_file="results/final_test_results.csv"))
    return rows, aggregate


def cell_stdout(notebook, index):
    return "".join("".join(output.get("text", [])) for output in notebook["cells"][index].get("outputs", [])
                   if output.get("output_type") == "stream" and output.get("name") == "stdout")


def parse_tradeoff(text):
    values = re.findall(r"(?m)^\s*\d+\s+([1248])\s+([0-9.]+)\s+([0-9.]+)\s+(-?[0-9.]+)\s*$", text)
    assert len(values) == 4
    return [dict(rank=int(rank), target_residual=float(target), preservation_score=float(preservation),
                 preservation_degradation_pct=float(drop)) for rank, target, preservation, drop in values]


def build_notebook_evidence():
    notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding="utf-8"))
    pres = read_csv("results/van_gogh_preservation.csv")
    assert len(pres) == 36
    prompts = {"a photo of a dog", "a photo of a car", "a photo of an airplane", "a portrait of a person"}
    expected = {(p, s, m) for p in prompts for s in (4025, 4026, 4027)
                for m in ("original", "fixed_rank_1", "adaptive_rank_4")}
    assert {(r["prompt"], int(r["seed"]), r["method"]) for r in pres} == expected
    assert {r["target_concept"] for r in pres} == {"Vincent van Gogh"}
    means = {m: stats.mean(float(r["clip_score"]) for r in pres if r["method"] == m)
             for m in ("original", "fixed_rank_1", "adaptive_rank_4")}
    text76 = cell_stdout(notebook, 76)
    original_printed = float(re.search(r"Original preservation:\s*([0-9.]+)", text76)[1])
    assert math.isclose(means["original"], original_printed, abs_tol=1e-15)
    vg_means = {int(rank): float(value) for rank, value in re.findall(r"Rank ([1248]) preservation:\s*([0-9.]+)", text76)}
    assert math.isclose(vg_means[1], means["fixed_rank_1"], abs_tol=1e-15)
    assert math.isclose(vg_means[4], means["adaptive_rank_4"], abs_tol=1e-15)
    style = []
    for concept, cell, calibration_cell in (("Vincent van Gogh", 77, 50), ("Claude Monet", 81, 51)):
        calibration = cell_stdout(notebook, calibration_cell).split("===== Calibration Summary =====")[1]
        target_means = {int(rank): float(value) for rank, value, _ in
            re.findall(r"(?m)^\s*\d+\s+([1248])\s+([0-9.]+)\s+([0-9.]+)\s*$", calibration)}
        assert set(target_means) == set(RANKS)
        for row in parse_tradeoff(cell_stdout(notebook, cell)):
            assert row["target_residual"] == target_means[row["rank"]]
            if concept == "Vincent van Gogh":
                value = vg_means[row["rank"]]
                assert abs(value - row["preservation_score"]) <= 0.0000005
                row["preservation_score"] = value
                psource = ("results/van_gogh_preservation.csv" if row["rank"] in (1, 4)
                           else f"{NOTEBOOK}#cell-76")
                precision = "full precision aggregate"
            else:
                psource = f"{NOTEBOOK}#cell-80"
                precision = "six decimal saved aggregate"
            # The recorded degradation is authoritative. Recomputing from the
            # rounded Monet mean would manufacture unavailable precision.
            inferred = 100 * (means["original"] - row["preservation_score"]) / means["original"]
            assert abs(inferred - row["preservation_degradation_pct"]) < 0.0003
            effective = max(0.0, row["preservation_degradation_pct"])
            style.append(dict(concept=concept, **row, original_preservation_score=means["original"],
                effective_preservation_degradation_pct=effective,
                satisfies_10pct_budget=effective <= 10, satisfies_15pct_budget=effective <= 15,
                satisfies_20pct_budget=effective <= 20, target_split="calibration",
                target_seeds="2025; 2026; 2027", preservation_seeds="4025; 4026; 4027",
                preservation_observations_per_rank=12, target_precision="six decimal saved aggregate",
                preservation_precision=precision, degradation_precision="six decimal saved aggregate",
                target_source=f"{NOTEBOOK}#cell-{calibration_cell}", preservation_source=psource,
                degradation_source=f"{NOTEBOOK}#cell-{cell}"))
    budgets = []
    for concept in ("Vincent van Gogh", "Claude Monet"):
        candidates = [r for r in style if r["concept"] == concept]
        saved_block = cell_stdout(notebook, 82).split(f"===== {concept} =====")[1].split("=====")[0]
        saved = {int(b): int(r) for b, r in re.findall(r"Preservation budget (\d+)% -> selected rank = (\d+)", saved_block)}
        for budget in (10, 15, 20):
            valid = [r for r in candidates if r["effective_preservation_degradation_pct"] <= budget]
            winner = min(valid, key=lambda r: (r["target_residual"], r["rank"]))
            assert winner["rank"] == saved[budget]
            budgets.append(dict(concept=concept, budget_pct=budget, selected_rank=winner["rank"],
                target_residual=winner["target_residual"], preservation_score=winner["preservation_score"],
                preservation_degradation_pct=winner["preservation_degradation_pct"],
                valid_ranks="; ".join(str(r["rank"]) for r in valid),
                target_split="calibration", source=f"{NOTEBOOK}#cell-82"))
    pilot = read_csv("results/pilot_results.csv")
    car_calibration = []
    for rank in (1, 2, 4):
        source = [r for r in pilot if r["concept"] == "car" and int(r["rank"]) == rank]
        assert {int(r["seed"]) for r in source} == {2025, 2026, 2027}
        scores = [float(r["clip_score"]) for r in source]
        car_calibration.append(dict(rank=rank, target_clip_mean=stats.mean(scores), target_clip_std=stats.stdev(scores),
            precision="full precision CSV", source="results/pilot_results.csv"))
    tree = ast.parse("".join(notebook["cells"][92]["source"]))
    recovered_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "car_rank8_recovered" for t in node.targets))
    recovered = ast.literal_eval(recovered_node.value.args[0])
    assert {(r["concept"], r["rank"], r["seed"]) for r in recovered} == {("car", 8, s) for s in (2025, 2026, 2027)}
    scores = [r["clip_score"] for r in recovered]
    car_calibration.append(dict(rank=8, target_clip_mean=stats.mean(scores), target_clip_std=stats.stdev(scores),
        precision="approximate; recovered six decimal samples", source=f"{NOTEBOOK}#cell-92"))
    car_tradeoff = parse_tradeoff(cell_stdout(notebook, 93))
    text95 = cell_stdout(notebook, 95)
    rank4 = dict(mean=float(re.search(r"(?m)^Mean:\s*([0-9.]+)", text95)[1]),
                 std=float(re.search(r"(?m)^Std:\s*([0-9.]+)", text95)[1]),
                 seeds=list(SEEDS), source=f"{NOTEBOOK}#cell-95")
    raw_counts = {}
    for cell, count in ((75, 24), (79, 48), (87, 48)):
        matches = re.findall(r"(?s)(Vincent van Gogh|Claude Monet|car) \| rank=(\d+) \| ([^\r\n]+) \| seed=(\d+).*?Preservation CLIP=(\d+\.\d+)", cell_stdout(notebook, cell))
        assert len(matches) == count
        raw_counts[str(cell)] = count
    evidence = dict(car_calibration=car_calibration, car_tradeoff=car_tradeoff, car_rank4_test=rank4,
        baseline_preservation=means["original"], notebook_source=NOTEBOOK,
        cell_index_convention="zero-based", printed_preservation_observation_counts=raw_counts,
        car_rank8_recovered_samples=recovered,
        saved_output_excerpts={str(i): cell_stdout(notebook, i) for i in (50, 51, 76, 77, 80, 81, 82, 88, 92, 93, 94)},
        car_rank4_saved_summary=text95[text95.index("Car rank=4 test:"):])
    return style, budgets, evidence


def build_spectra():
    # CPU tensor deserialization only. No decomposition, model loading, or inference.
    import torch
    assert not torch.cuda.is_initialized()
    csv_stats = {r["concept"]: r for r in read_csv("results/spectrum_summary.csv")}
    rows = []
    caches = {}
    for concept in CONCEPTS:
        name = f"results/subspaces/{concept.lower().replace(' ', '_')}.pt"
        record = torch.load(ROOT / name, map_location="cpu", weights_only=True)
        assert record["concept"] == concept and record["U"].device.type == "cpu"
        cached = dict(pc1_energy=record["ratio"][0].item(), **{
            f"rank_{int(threshold * 100)}": rank for threshold, rank in record["threshold_ranks"].items()})
        if concept in csv_stats:
            authoritative = csv_stats[concept]
            assert math.isclose(float(authoritative["pc1_energy"]), cached["pc1_energy"], abs_tol=1e-15)
            assert all(int(authoritative[k]) == cached[k] for k in ("rank_70", "rank_80", "rank_90", "rank_95"))
            source = "results/spectrum_summary.csv"
            cached = {k: float(authoritative[k]) if k == "pc1_energy" else int(authoritative[k]) for k in cached}
        else:
            source = name
        rows.append(dict(concept=concept, category=CATEGORIES[concept], selected_rank=SELECTED[concept], **cached,
            available_singular_values=record["S"].numel(), spectrum_source=source,
            selection_source="results/selected_ranks.csv"))
        caches[name] = dict(concept=concept, modes=record["S"].numel())
    assert not torch.cuda.is_initialized()
    return rows, caches


def inventory(caches):
    rows = []
    specifications = {
        "final_test_results.csv": ("main rank comparison", "held-out final test", "; ".join(CONCEPTS), "1; 2; 4", "3025-3029", "target CLIP residual", 70, "sample rows", "Primary main/aggregate source; rank-one adaptive rows reuse baseline"),
        "pilot_results.csv": ("early rank stability pilot", "calibration / exploratory pilot", "Vincent van Gogh; car; dog", "Van Gogh:1,4,7; car/dog:1,2,4", "2025-2027", "target CLIP residual", 27, "sample rows", "Incomplete candidate grid; Van Gogh rank7 is exploratory, never pooled into main test"),
        "selected_ranks.csv": ("frozen first-stage selections", "calibration selection metadata", "; ".join(CONCEPTS), "1; 2; 4", "2025-2027 (selection protocol)", "selected rank", 7, "concept records", "Original mapping; exploratory Car rank4 does not replace Car rank2"),
        "spectrum_summary.csv": ("uncentered activation SVD", "spectrum", "Taylor Swift; Vincent van Gogh; car; dog", "energy thresholds70/80/90/95%", "not applicable", "PC1 energy; cumulative-energy ranks", 4, "concept records", "Supplement absent concepts from cached spectrum metadata; do not recompute SVD"),
        "van_gogh_preservation.csv": ("original vs projected preservation", "preservation", "Vincent van Gogh", "original; 1; 4", "4025-4027", "preservation CLIP", 36, "prompt-seed-method rows", "Four prompts x three seeds x three methods; full precision"),
    }
    for path in sorted((ROOT / "results").rglob("*")):
        if not path.is_file() or path.is_relative_to(REPORTING):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.name in specifications and path.parent == ROOT / "results":
            spec = specifications[path.name]
            assert len(read_csv(relative)) == spec[6]
        elif relative in caches:
            cache = caches[relative]
            spec = ("cached concept subspace", "spectrum", cache["concept"], "basis modes", "not applicable",
                    "U; S; energy ratios; threshold ranks", 1, "concept cache",
                    f"{cache['modes']} singular values; CPU read only; no SVD recomputation")
        elif "car-gpu-parity-20260925" in relative:
            count = len(read_csv(relative)) if path.suffix == ".csv" else (1 if path.suffix == ".json" else "")
            spec = ("previous GPU parity smoke test", "implementation validation (excluded from research aggregates)",
                    "car", "1", "2025", "target CLIP residual; parity/checkpoint metadata", count,
                    "sample rows" if path.suffix == ".csv" else "metadata object" if path.suffix == ".json" else "not a data table",
                    "Existing smoke test duplicate; read only, never rerun or added to research sample counts")
        else:
            spec = ("unclassified artifact", "unclassified", "", "", "", "", "", "unknown", "Not used; requires explicit source review")
        rows.append(dict(source_file=relative, source_locator="whole file", experiment_type=spec[0], partition=spec[1],
            concepts=spec[2], ranks=spec[3], seeds=spec[4], metrics=spec[5], row_count=spec[6], row_unit=spec[7],
            notes=spec[8], sha256=digest(path)))
    for cells, concepts, kind, ranks, seeds, count, unit, notes in (
        ("50;51", "Vincent van Gogh; Claude Monet", "calibration summary", "1;2;4;8", "2025-2027", 8, "rank means", "Six-decimal saved means; no complete full-precision calibration CSV"),
        ("75", "Vincent van Gogh", "preservation", "2;8", "4025-4027", 24, "printed sample rows", "Six-decimal samples; prefer full-precision aggregate means in cell76"),
        ("76", "Vincent van Gogh", "preservation aggregate", "original;1;2;4;8", "4025-4027", 5, "method/rank means", "Full-precision printed means; original/r1/r4 cross-checked against CSV"),
        ("79", "Claude Monet", "preservation", "1;2;4;8", "4025-4027", 48, "printed sample rows", "Six-decimal samples; means in cell80 and degradation in81 retained at recorded precision"),
        ("77;81", "Vincent van Gogh; Claude Monet", "calibration/preservation trade-off", "1;2;4;8", "target2025-2027;preservation4025-4027", 8, "rank summary rows", "Target and preservation splits remain distinct; percentages saved to six decimals"),
        ("82", "Vincent van Gogh; Claude Monet", "preservation-budget sensitivity", "1;4 selected", "same trade-off samples", 6, "concept-budget selections", "Budgets10/15/20%; negatives clamped to0 for feasibility only"),
        ("87", "car", "exploratory preservation", "1;2;4;8", "4025-4027", 48, "printed sample rows", "Mixed prompt diagnostic includes the Car target prompt"),
        ("92", "car", "exploratory calibration recovery", "8", "2025-2027", 3, "rounded recovered samples", "Six-decimal literals; not full-precision original CSV observations"),
        ("93;94", "car", "exploratory preservation-aware selection", "1;2;4;8", "target2025-2027;preservation4025-4027", 4, "rank summary rows", "20% budget selects4; original main selection remains2"),
        ("95", "car", "exploratory rank4 follow-up", "4", "3025-3029", 5, "printed sample rows", "Same held-out seeds reused; full-precision printed aggregate mean/SD, six-decimal sample scores"),
    ):
        rows.append(dict(source_file=NOTEBOOK, source_locator=f"zero-based cells {cells}", experiment_type=kind,
            partition=kind, concepts=concepts, ranks=ranks, seeds=seeds, metrics="CLIP cosine / selection",
            row_count=count, row_unit=unit, notes=notes, sha256=digest(ROOT / NOTEBOOK)))
    return rows


def build(output_dir):
    out = output_directory(output_dir)
    protected_count = verify_frozen_inputs()
    main_rows, aggregate_rows = build_main()
    style_rows, budget_rows, evidence = build_notebook_evidence()
    rank_rows, caches = build_spectra()
    for name, rows in (("main_results", main_rows), ("aggregate_results", aggregate_rows),
                       ("rank_selection", rank_rows), ("style_tradeoff", style_rows),
                       ("preservation_budget_sensitivity", budget_rows), ("data_inventory", inventory(caches))):
        write_csv(out / f"{name}.csv", rows)
    write(out / "notebook_evidence.json", json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write(out / "main_results.md", "# Held-out target CLIP residual\n\nLower is better. Mean ± sample SD (ddof=1), five paired test seeds 3025–3029 per concept and method.\n\n" +
        table(["Concept", "Category", "Fixed rank", "Adaptive rank", "Fixed mean ± SD", "Adaptive mean ± SD", "Relative improvement"],
        [[r["concept"], r["category"], 1, r["adaptive_rank"], f'{r["fixed_target_clip_mean"]:.6f} ± {r["fixed_target_clip_std"]:.6f}',
          f'{r["adaptive_target_clip_mean"]:.6f} ± {r["adaptive_target_clip_std"]:.6f}', f'{r["relative_improvement_pct"]:.2f}%'] for r in main_rows]) +
        "\nRelative improvement = 100 × (fixed mean − adaptive mean) / fixed mean. Negative values indicate regression; Car remains visible. Source: [frozen final test](../final_test_results.csv), ranks from [recorded selections](../selected_ranks.csv). No calibration or smoke-test rows enter this table.\n\nAirplane and dog use rank 1 for both methods and reuse baseline scores. Their negligible final-digit CSV serialization differences are retained in calculations and display as 0.00%. SD describes seed variation, not a confidence interval or significance test.\n")
    group_labels = dict(all_concepts="All concepts", changed_rank="Changed-rank concepts", style="Style concepts", non_style="Non-style concepts")
    write(out / "aggregate_results.md", "# Macro target-residual results\n\nLower target CLIP residual is better.\n\n" +
        table(["Group", "Concepts", "Fixed rank 1", "Adaptive", "Relative improvement"],
            [[group_labels[r["group"]], r["n_concepts"], f'{r["fixed_target_clip_mean"]:.6f}',
              f'{r["adaptive_target_clip_mean"]:.6f}', f'{r["relative_improvement_pct"]:.2f}%'] for r in aggregate_rows]) +
        "\nEach method's macro average is the equally weighted mean of its per-concept five-seed means. Relative improvement is computed from those two macro averages, **not** by averaging per-concept percentages. All rows use only [final_test_results.csv](../final_test_results.csv).\n\nChanged-rank: Taylor Swift, Vincent van Gogh, Claude Monet, car, cat. Style: Vincent van Gogh and Claude Monet. Non-style: the other five concepts. Car is retained in every applicable group.\n")
    write(out / "rank_selection.md", "# Recorded ranks and activation spectra\n\n" +
        table(["Concept", "Category", "Selected rank", "PC1 energy", "70% rank", "80% rank", "90% rank", "95% rank"],
            [[r["concept"], r["category"], r["selected_rank"], f'{r["pc1_energy"]:.6f}',
              r["rank_70"], r["rank_80"], r["rank_90"], r["rank_95"]] for r in rank_rows]) +
        "\nSelections come from [selected_ranks.csv](../selected_ranks.csv). Spectrum statistics use [spectrum_summary.csv](../spectrum_summary.csv) for its four recorded concepts and the existing cached `.pt` ratio/threshold metadata for the other three; all seven caches were read on CPU, without recomputing SVD. Per-row provenance is in [rank_selection.csv](rank_selection.csv).\n\nEnergy ranks are descriptive thresholds of the uncentered activation SVD, not candidate ranks selected for erasure. They differ from the empirically selected ranks; this does not establish a universal spectrum-to-rank rule.\n")
    text = "# Style erasure–preservation trade-off\n\nTarget residual: **lower is better**, calibration seeds 2025–2027. Preservation CLIP: **higher is better**, four prompts × seeds 4025–4027 (12 observations per rank). Original preservation mean: " + f'{evidence["baseline_preservation"]:.12f}.\n'
    for concept in ("Vincent van Gogh", "Claude Monet"):
        text += f"\n## {concept}\n\n" + table(["Rank", "Target residual", "Preservation", "Degradation %", "10% budget", "15% budget", "20% budget"],
            [[r["rank"], f'{r["target_residual"]:.6f}', f'{r["preservation_score"]:.6f}', f'{r["preservation_degradation_pct"]:.6f}',
              *("Yes" if r[f"satisfies_{b}pct_budget"] else "No" for b in (10, 15, 20))] for r in style_rows if r["concept"] == concept])
    text += "\nDegradation = 100 × (original preservation − projected preservation) / original preservation. Negative values mean **no observed degradation / sampling variation**, not evidence that erasure improves unrelated generation. Only budget feasibility uses max(0, degradation), matching the notebook.\n\nThe target columns are calibration summaries, not the held-out means in the main table. The notebook combines these with the separate preservation seed set for exploratory constrained selection.\n\nSources: [Van Gogh preservation CSV](../van_gogh_preservation.csv) and saved outputs in [adaptive_rank_experiment.ipynb](../../experiments/adaptive_rank_experiment.ipynb), zero-based cells 50–51, 76–77, 80–82. Van Gogh rank 2/8 full-precision aggregate means survive in cell 76; Monet means and all style target/degradation summaries survive only at six decimal places. Recorded degradation values are retained instead of back-solving missing precision or recomputing them from rounded means. Full provenance is in [style_tradeoff.csv](style_tradeoff.csv) and [notebook_evidence.json](notebook_evidence.json).\n"
    write(out / "style_tradeoff.md", text)
    write(out / "preservation_budget_sensitivity.md", "# Preservation-budget sensitivity\n\n" +
        table(["Concept", "Budget %", "Eligible ranks", "Selected rank", "Target residual", "Preservation degradation %"],
            [[r["concept"], r["budget_pct"], r["valid_ranks"].replace("; ", ", "), r["selected_rank"],
              f'{r["target_residual"]:.6f}', f'{r["preservation_degradation_pct"]:.6f}'] for r in budget_rows]) +
        "\nSelect the lowest calibration target residual among ranks with max(0, recorded preservation degradation) ≤ budget. Negative degradation is interpreted as no observed degradation / sampling variation. The 10%, 15%, and 20% budgets and resulting choices exactly reproduce saved notebook cell 82 (20% also checked in cells 84–85). These are sensitivity analyses of the existing trade-off data, not a new independently tested method; they do not replace the frozen first-stage selections.\n")
    from report_narratives import render_narratives
    for name, text in render_narratives(main_rows, aggregate_rows, style_rows, budget_rows, evidence).items():
        write(out / name, text)
    assert verify_frozen_inputs() == protected_count
    print(json.dumps({"tables": 6, "main_rows": len(main_rows), "style_rows": len(style_rows),
        "budget_rows": len(budget_rows), "protected_files_unchanged": protected_count,
        "gpu_experiments_launched": 0, "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(REPORTING))
    build(parser.parse_args().output_dir)
