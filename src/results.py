"""Exclusive run directories and validated, append-only CSV checkpoints.

Only directories created by this runner under results/runs can be resumed.
Historical CSVs and cached subspaces are never opened for writing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_run_dir(repository: Path, run_dir: str | Path) -> Path:
    """Reject root, historical locations, escapes, and symlinks outside runs."""
    repository = repository.resolve()
    runs = repository / "results" / "runs"
    if runs.resolve() != runs:
        raise ValueError("results/runs must not resolve through a symlink")
    path = Path(run_dir)
    if not path.is_absolute():
        path = repository / path
    path = path.resolve()
    if not path.is_relative_to(runs) or path == runs:
        raise ValueError("--run-dir must be a child of this repository's results/runs/")
    return path


@contextmanager
def open_run(path: Path, manifest: dict, *, resume: bool = False):
    """Create once; resumption requires identical inputs and environment.

    An exclusive lock prevents concurrent appenders. After a forced process kill,
    the user must confirm that no runner is active before removing its stale lock.
    """
    manifest_path = path / "manifest.json"
    if resume:
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("Resume requires a runner-created manifest.json")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)

    lock_path = path / ".lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
            lock.write(f"pid={os.getpid()}\n")
        if resume:
            with manifest_path.open(encoding="utf-8") as handle:
                previous = json.load(handle)
            if previous != manifest:
                raise ValueError(
                    "Resume manifest differs: use the same phase, concepts, code, "
                    "input files and dependency versions, or start a new run."
                )
        else:
            with manifest_path.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
        yield path
    finally:
        lock_path.unlink()


class CsvResults:
    """One flushed row per completed job, with strict resumption validation."""

    def __init__(self, path: Path, fields: tuple[str, ...], expected: list[dict]):
        self.path = path
        self.fields = fields
        self.key_fields = tuple(field for field in fields if field != "clip_score")
        self.expected = {self.key(row) for row in expected}
        if len(self.expected) != len(expected):
            raise ValueError("Experiment plan contains duplicate jobs")
        self.rows: dict[tuple, dict] = {}
        if path.is_symlink() or (path.exists() and path.stat().st_nlink > 1):
            raise ValueError(f"Refusing linked CSV: {path}")
        if path.exists():
            # A partial final line must not be mistaken for a completed score.
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    raise ValueError(f"Empty checkpoint: {path}")
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise ValueError(f"Incomplete final CSV line: {path}")
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != list(fields):
                    raise ValueError(f"Unexpected CSV header in {path}")
                for raw_row in reader:
                    row = self.validate(raw_row)
                    key = self.key(row)
                    if key in self.rows:
                        raise ValueError(f"Duplicate checkpoint row: {key}")
                    self.rows[key] = row
        else:
            with path.open("x", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()

    def key(self, row: dict) -> tuple:
        return tuple(str(row[field]) for field in self.key_fields)

    def validate(self, raw_row: dict) -> dict:
        if set(raw_row) != set(self.fields) or any(v is None for v in raw_row.values()):
            raise ValueError(f"Malformed row in {self.path}: {raw_row}")
        row = dict(raw_row)
        for field in ("rank", "seed"):
            if field in row:
                # Strict integer parsing also rejects truncated or float keys.
                row[field] = int(str(row[field]))
        row["clip_score"] = float(row["clip_score"])
        if not math.isfinite(row["clip_score"]):
            raise ValueError(f"Nonfinite CLIP score in {self.path}")
        if self.key(row) not in self.expected:
            raise ValueError(f"CSV row is outside the recorded protocol: {row}")
        return row

    def get(self, job: dict) -> dict | None:
        return self.rows.get(self.key(job))

    def append(self, raw_row: dict) -> None:
        row = self.validate(raw_row)
        key = self.key(row)
        if key in self.rows:
            raise ValueError(f"Already completed: {key}")
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[key] = row
