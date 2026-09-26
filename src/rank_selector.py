"""Validated calibration rank selection and recorded final-test ranks.

This module preserves cells 45 and 52 of adaptive_rank_experiment.ipynb.
Calibration selects the smallest mean CLIP cosine score across three seeds;
final evaluation uses the separately validated, recorded rank mapping.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .protocol import CALIBRATION_SEEDS, CANDIDATE_RANKS, SELECTED_RANKS


def _integer(value: Any, field: str) -> int:
    """Read integer CSV values without silently truncating fractional values."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if not isinstance(value, str) and value != parsed:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return parsed


def select_adaptive_rank(
    rows: Iterable[Mapping[str, Any]], concept: str
) -> int:
    """Select a rank from one complete, unique 4-rank by 3-seed calibration.

    Rows for other concepts are ignored. For the requested concept every
    candidate rank and calibration seed must occur exactly once, with a finite
    ``clip_score``. Exact mean ties select the lowest rank, as the notebook's
    sorted pandas groupby followed by idxmin does. This function does not alter
    the recorded ranks used for final evaluation.
    """
    if concept not in SELECTED_RANKS:
        raise ValueError(f"Unknown concept: {concept!r}")

    expected = {
        (rank, seed) for rank in CANDIDATE_RANKS for seed in CALIBRATION_SEEDS
    }
    scores: dict[tuple[int, int], float] = {}
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Calibration row {row_number} must be a mapping")
        if row.get("concept") != concept:
            continue
        rank = _integer(row.get("rank"), "rank")
        seed = _integer(row.get("seed"), "seed")
        key = (rank, seed)
        if key not in expected:
            raise ValueError(
                f"Unexpected calibration rank/seed for {concept!r}: {key}"
            )
        if key in scores:
            raise ValueError(
                f"Duplicate calibration rank/seed for {concept!r}: {key}"
            )
        value = row.get("clip_score")
        if isinstance(value, bool):
            raise ValueError(f"Invalid clip_score for {concept!r}, {key}: {value!r}")
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid clip_score for {concept!r}, {key}: {value!r}"
            ) from exc
        if not math.isfinite(score):
            raise ValueError(
                f"Non-finite clip_score for {concept!r}, {key}: {value!r}"
            )
        scores[key] = score

    missing = expected - scores.keys()
    if missing:
        raise ValueError(
            f"Incomplete calibration for {concept!r}; missing rank/seed pairs: "
            f"{sorted(missing)}"
        )
    means = {
        rank: math.fsum(scores[(rank, seed)] for seed in CALIBRATION_SEEDS)
        / len(CALIBRATION_SEEDS)
        for rank in CANDIDATE_RANKS
    }
    return min(CANDIDATE_RANKS, key=lambda rank: (means[rank], rank))


def load_selected_ranks(path: str | Path) -> dict[str, int]:
    """Read the seven recorded ranks, rejecting any changes to their mapping.

    The historical selection CSV is a read-only input. A newly computed
    calibration must not silently replace those selections for the final test.
    """
    selected: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["concept", "selected_rank"]:
            raise ValueError(
                "Selected-rank CSV must have columns: concept,selected_rank"
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Extra fields in selected-rank CSV row {row_number}")
            concept = row["concept"]
            if concept not in SELECTED_RANKS:
                raise ValueError(f"Unknown concept in selected-rank CSV: {concept!r}")
            if concept in selected:
                raise ValueError(f"Duplicate concept in selected-rank CSV: {concept!r}")
            rank = _integer(row["selected_rank"], "selected_rank")
            expected_rank = SELECTED_RANKS[concept]
            if rank != expected_rank:
                raise ValueError(
                    f"Recorded rank for {concept!r} must remain {expected_rank}, "
                    f"got {rank}"
                )
            selected[concept] = rank
    missing = SELECTED_RANKS.keys() - selected.keys()
    if missing:
        raise ValueError(f"Missing concepts in selected-rank CSV: {sorted(missing)}")
    return {concept: selected[concept] for concept in SELECTED_RANKS}
