#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sources = load_module("build_c8_v3_sources", "build_c8_v3_sources.py")
c3 = load_module("freeze_c3_v12", "freeze_c3_v12.py")


class C8V3SourceTests(unittest.TestCase):
    def test_source_selection_is_unique_and_refuses_thin_strata(self):
        rows = [
            {"source": "a", "text": "abc"},
            {"source": "duplicate", "text": "abc"},
            {"source": "b", "text": "defg"},
        ]
        selected = sources.select_without_replacement(
            rows, seed=7, domain="fixture", byte_target=4
        )
        self.assertEqual(
            len({sources.sha256_bytes(row["text"].encode()) for row in selected}),
            len(selected),
        )
        with self.assertRaisesRegex(ValueError, "below its 100-byte pool target"):
            sources.select_without_replacement(
                rows, seed=7, domain="fixture", byte_target=100
            )

    def test_structured_rendering_separates_call_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "species": "s3",
                        "text": "interpolate <call> 1 2 3 = 4 </call>",
                        "meta": {"cell": "lerp"},
                    }
                )
                + "\n"
            )
            rendered = list(sources.structured_rows(path, "fixture"))
            record = json.loads(rendered[0]["text"])
            self.assertEqual(record["instruction"], "interpolate")
            self.assertEqual(record["call"], "1 2 3 = 4")
            self.assertIsNone(record["trace"])

    def test_c3_freeze_obeys_predeclared_byte_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            with source.open("w") as handle:
                for index in range(10):
                    handle.write(
                        json.dumps(
                            {"source": f"row:{index}", "text": f"x{index}"}
                        )
                        + "\n"
                    )
            spec = {
                "schema_version": 1,
                "slice_id": "fixture",
                "frozen_date": "2026-07-25",
                "seed": 11,
                "items_per_domain": 3,
                "domains": {
                    "code": {
                        "path": "source.jsonl",
                        "sha256": c3.sha256_file(source),
                        "maximum_text_bytes": 2,
                        "maximum_heldout_bytes": 6,
                    }
                },
            }
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            manifest = c3.freeze(
                spec_path, root / "heldout.jsonl", root / "manifest.json"
            )
            self.assertEqual(manifest["items_total"], 3)
            self.assertLessEqual(
                manifest["source_evidence"]["code"]["selected_text_bytes"], 6
            )

    def test_tracked_freeze_artifacts_agree(self):
        c8_spec = json.loads((HERE / "c8_v3_spec.json").read_text())
        source_spec = json.loads((HERE / "c8_v3_source_spec.json").read_text())
        c3_manifest = json.loads((HERE / "c3_v12_manifest.json").read_text())
        c8_manifest = json.loads((HERE / "c8_manifest_v3.json").read_text())
        self.assertEqual(c8_spec["status"], "frozen")
        self.assertEqual(source_spec["status"], "frozen")
        self.assertEqual(c3_manifest["items_total"], 200)
        self.assertEqual(c8_manifest["c3_overlap_text_hashes"], 0)
        self.assertEqual(
            c8_spec["expected_output"]["sha256"], c8_manifest["output_sha256"]
        )
        self.assertEqual(c8_manifest["total_bytes_actual"], 80_000_000)


if __name__ == "__main__":
    unittest.main()
