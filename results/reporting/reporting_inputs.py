"""Portable verification of the explicit frozen reporting input set.

Manifest hashes describe committed Git content, not historical working-tree
snapshots. Only CRLF/LF checkout conversion is tolerated for declared text;
binary inputs and generated figures always retain exact byte comparisons.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def text_bytes(content: bytes) -> bytes:
    """Return the LF representation without changing any non-EOL byte."""
    return content.replace(b"\r\n", b"\n")


def load_manifest(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "results/reporting/input_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty reporting input manifest")
    for relative, entry in manifest["files"].items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
            raise ValueError(f"Invalid reporting input path: {relative}")
        if entry.get("content") not in ("text_lf", "binary"):
            raise ValueError(f"Unknown content policy: {relative}")
    return manifest


def verify_inputs(root: Path = ROOT) -> dict:
    """Reject missing/altered required inputs, independently of extra files.

The expected SHA-256 values were derived from Git blobs at source_commit.
Verification needs no Git executable or ignored artifacts at runtime.
"""
    root = root.resolve()
    files = load_manifest(root)["files"]
    for relative, entry in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Reporting input resolves outside the repository: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Required committed reporting input is missing: {relative}")
        content = path.read_bytes()
        if entry["content"] == "text_lf":
            content = text_bytes(content)
        if hashlib.sha256(content).hexdigest() != entry["committed_sha256"]:
            raise ValueError(f"Required committed reporting input changed: {relative}")
    return files


def inventory_digest(entry: dict) -> str:
    """Preserve historical inventory metadata, never use it for verification."""
    return entry.get("historical_inventory_sha256", entry["committed_sha256"])
