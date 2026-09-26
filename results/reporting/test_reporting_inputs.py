"""Offline provenance/EOL checks; no scientific data is edited."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from generate_report import write
from reporting_inputs import verify_inputs


class ReportingInputChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".input-checks-", dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        reporting = self.root / "results/reporting"
        reporting.mkdir(parents=True)
        self.text = self.root / "results/source.csv"
        self.binary = self.root / "results/subspace.pt"
        self.text.write_bytes(b"name,value\ncar,1\n")
        self.binary.write_bytes(b"\x00tensor\r\ndata\xff")
        files = {}
        for path, mode in ((self.text, "text_lf"), (self.binary, "binary")):
            files[path.relative_to(self.root).as_posix()] = {
                "content": mode, "committed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (reporting / "input_manifest.json").write_text(
            json.dumps({"format_version": 1, "files": files}), encoding="utf-8")

    def test_crlf_checkout_is_accepted_without_rewriting(self):
        converted = self.text.read_bytes().replace(b"\n", b"\r\n")
        self.text.write_bytes(converted)
        self.assertEqual(len(verify_inputs(self.root)), 2)
        self.assertEqual(self.text.read_bytes(), converted)

    def test_missing_required_file_is_rejected(self):
        self.text.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "results/source.csv"):
            verify_inputs(self.root)

    def test_non_eol_change_is_rejected(self):
        self.text.write_bytes(b"name,value\ncar,2\n")
        with self.assertRaisesRegex(ValueError, "results/source.csv"):
            verify_inputs(self.root)

    def test_binary_is_never_eol_normalized(self):
        self.binary.write_bytes(self.binary.read_bytes().replace(b"\r\n", b"\n"))
        with self.assertRaisesRegex(ValueError, "results/subspace.pt"):
            verify_inputs(self.root)

    def test_unlisted_results_and_absent_archive_do_not_affect_inputs(self):
        before = verify_inputs(self.root)
        (self.root / "results/future-audit.md").write_text("unrelated", encoding="utf-8")
        runs = self.root / "results/runs/future"
        runs.mkdir(parents=True)
        (runs / "scores.csv").write_text("not frozen data", encoding="utf-8")
        self.assertEqual(verify_inputs(self.root), before)
        self.assertFalse((self.root / "results/reporting/source_snapshot.json").exists())

    def test_existing_crlf_artifact_is_not_rewritten(self):
        output = self.root / "results/reporting/table.md"
        output.write_bytes(b"# Table\r\n")
        write(output, "# Table\n")
        self.assertEqual(output.read_bytes(), b"# Table\r\n")

    def test_changed_artifact_is_not_overwritten(self):
        output = self.root / "results/reporting/table.md"
        output.write_bytes(b"# Changed\n")
        with self.assertRaises(FileExistsError):
            write(output, "# Table\n")
        self.assertEqual(output.read_bytes(), b"# Changed\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
