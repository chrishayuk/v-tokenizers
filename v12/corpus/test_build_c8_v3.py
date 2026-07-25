#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_c8_v3.py")
SPEC = importlib.util.spec_from_file_location("build_c8_v3", MODULE_PATH)
assert SPEC and SPEC.loader
c8 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = c8
SPEC.loader.exec_module(c8)


class C8V3Tests(unittest.TestCase):
    def test_largest_remainder_is_exact_and_stable(self):
        targets = c8.allocate_byte_targets(11, {"prose": 0.45, "code": 0.2, "math": 0.15, "json": 0.15, "noise": 0.05})
        self.assertEqual(sum(targets.values()), 11)
        self.assertEqual(targets, {"prose": 5, "code": 2, "math": 2, "json": 2, "noise": 0})

    def test_utf8_prefix_never_splits_a_code_point(self):
        self.assertEqual(c8.utf8_prefix("a🫠b", 1), "a")
        self.assertEqual(c8.utf8_prefix("a🫠b", 4), "a")
        self.assertEqual(c8.utf8_prefix("a🫠b", 5), "a🫠")

    def test_build_is_reproducible_and_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            domains = ["natural_prose", "code", "math_reasoning", "json_tool_cell", "noisy_unicode_mixed"]
            source = root / "source.jsonl"
            with source.open("w") as handle:
                for domain in domains:
                    for index in range(20):
                        handle.write(json.dumps({"domain": domain, "source": f"{domain}:{index}", "text": f"{domain}-{index}-abcdefghij"}) + "\n")
            digest = c8.sha256_file(source)
            excluded_text = "code-0-abcdefghij"
            c3_manifest = root / "c3.json"
            c3_manifest.write_text(
                json.dumps(
                    {
                        "slice_id": "fixture-c3",
                        "items_total": 1,
                        "heldout_jsonl_sha256": "fixture",
                        "items": [
                            {
                                "domain": "code",
                                "text_sha256": c8.sha256_bytes(
                                    excluded_text.encode()
                                ),
                            }
                        ],
                    }
                )
            )
            shares = [0.45, 0.2, 0.15, 0.15, 0.05]
            spec = {
                "schema_version": 1,
                "corpus_id": "fixture",
                "status": "frozen",
                "seed": 7,
                "total_bytes": 200,
                "c3_exclusion": {
                    "path": "c3.json",
                    "sha256": c8.sha256_file(c3_manifest),
                },
                "domains": {
                    domain: {
                        "share": share,
                        "sources": [{"path": "source.jsonl", "sha256": digest, "where": {"field": "domain", "equals": domain}}],
                    }
                    for domain, share in zip(domains, shares)
                },
            }
            spec_path = root / "spec.json"
            spec["status"] = "draft"
            spec_path.write_text(json.dumps(spec))
            output_a, output_b = root / "a.jsonl", root / "b.jsonl"
            manifest_a, manifest_b = root / "a.json", root / "b.json"
            first = c8.build(spec_path, output_a, manifest_a)
            second = c8.build(spec_path, output_b, manifest_b)

            self.assertEqual(first["total_bytes_actual"], 200)
            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
            self.assertEqual(first["output_sha256"], second["output_sha256"])
            self.assertEqual(first["c3_overlap_text_hashes"], 0)
            self.assertEqual(first["domains"]["code"]["c3_rows_excluded"], 1)
            self.assertNotIn(excluded_text, output_a.read_text())
            self.assertTrue(all(d["bytes_actual"] == d["bytes_target"] for d in first["domains"].values()))

    def test_frozen_build_requires_expected_output_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(
                json.dumps({"source": "fixture", "text": "abcdefghij"}) + "\n"
            )
            spec = {
                "schema_version": 1,
                "corpus_id": "fixture",
                "status": "frozen",
                "seed": 7,
                "total_bytes": 5,
                "domains": {
                    "only": {
                        "share": 1.0,
                        "sources": [
                            {
                                "path": "source.jsonl",
                                "sha256": c8.sha256_file(source),
                            }
                        ],
                    }
                },
            }
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "must pin expected_output"):
                c8.build(spec_path, root / "out.jsonl", root / "manifest.json")

    def test_thin_domain_fails_instead_of_repeating_rows(self):
        row = c8.SourceRow(text="abc", source="fixture", text_sha256=c8.sha256_bytes(b"abc"))
        with self.assertRaisesRegex(ValueError, "add source data rather than repeating"):
            c8.select_to_byte_budget([row], 4)


if __name__ == "__main__":
    unittest.main()
