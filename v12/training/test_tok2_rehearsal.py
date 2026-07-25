#!/usr/bin/env python3

import json
import hashlib
import unittest
from collections import Counter
from pathlib import Path

from v12.training.rehearse_tok2_matrix import canary_ffn_width, token_chunks


HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tok2RehearsalTests(unittest.TestCase):
    def test_chunking_preserves_every_next_token_target(self):
        tokens = list(range(600))
        chunks = token_chunks(tokens, 256)
        self.assertEqual(sum(len(chunk) - 1 for chunk in chunks), len(tokens) - 1)
        self.assertEqual(chunks[0][-1], chunks[1][0])
        self.assertEqual(chunks[1][-1], chunks[2][0])

    def test_canary_widths_preserve_control_routing(self):
        spec = json.loads((HERE / "tok2_rehearsal_spec.json").read_text())
        self.assertEqual(canary_ffn_width(2048, spec), 64)
        self.assertEqual(canary_ffn_width(2968, spec), 96)
        self.assertEqual(canary_ffn_width(3232, spec), 104)

    def test_frozen_arm_manifest_is_complete(self):
        manifest = json.loads((HERE / "tok2_tokenizer_arms.json").read_text())
        arms = manifest["arms"]
        self.assertEqual(len(arms), 8)
        self.assertEqual(len({arm["id"] for arm in arms}), 8)
        self.assertEqual(
            Counter(arm["family"] for arm in arms),
            {"v11": 1, "U16": 3, "B16": 3, "byte": 1},
        )
        self.assertEqual(
            {arm["vocab_rows"] for arm in arms if arm["family"] in {"U16", "B16"}},
            {16000},
        )

    def test_tracked_rehearsal_covers_the_full_matrix(self):
        result = json.loads((HERE / "tok2_rehearsal_results.json").read_text())
        rows = result["results"]
        self.assertEqual(result["status"], "pass-canary-only-not-a-model-result")
        self.assertEqual(result["matrix"]["cells_completed"], 48)
        self.assertEqual(
            result["spec_sha256"], sha256_file(HERE / "tok2_rehearsal_spec.json")
        )
        self.assertEqual(
            result["arms_manifest_sha256"],
            sha256_file(HERE / "tok2_tokenizer_arms.json"),
        )
        self.assertEqual(len(rows), 48)
        self.assertEqual(
            Counter(row["control"] for row in rows),
            {"fixed_trunk": 24, "fixed_total": 24},
        )
        self.assertEqual(Counter(row["run_label"] for row in rows), {0: 16, 1: 16, 2: 16})
        self.assertTrue(
            all(
                row["phase1_completed"]
                and row["phase3_completed"]
                and row["evaluation_completed"]
                and not row["phase1_eliminated"]
                for row in rows
            )
        )
        self.assertEqual(
            len(
                {
                    tuple(sorted(row["streams"].items()))
                    for row in rows
                }
            ),
            1,
        )
        required_domains = {
            "natural_prose",
            "code",
            "math_reasoning",
            "json_tool_cell",
            "noisy_unicode_mixed",
        }
        for phase in ("phase1", "phase3", "evaluation"):
            self.assertEqual(set(result["streams"][phase]["domains"]), required_domains)
        required_accounting = {
            "raw_bytes_consumed",
            "documents_consumed",
            "input_tokens_consumed",
            "target_tokens_predicted",
            "optimizer_steps",
            "estimated_production_training_flops",
            "wall_clock_seconds",
            "peak_host_memory_bytes_process",
        }
        for row in rows:
            self.assertTrue(required_accounting <= set(row["phase1"]))
            self.assertTrue(required_accounting <= set(row["phase3"]))

    def test_actual_production_module_counts_were_checked(self):
        result = json.loads((HERE / "tok2_rehearsal_results.json").read_text())
        configurations = result["production_module_validation"]["configurations"]
        self.assertEqual(
            configurations["vocab71260_ffn2048"]["nominal_parameters"],
            115149312,
        )
        self.assertEqual(
            configurations["vocab16000_ffn2968"]["nominal_parameters"],
            115118592,
        )
        self.assertEqual(
            configurations["vocab260_ffn3232"]["nominal_parameters"],
            115169792,
        )


if __name__ == "__main__":
    unittest.main()
