from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from analyze_tok2_production import (
    interpolate,
    paired_document_bootstrap,
    training_interval,
)
from freeze_tok2_production import build_manifest
from train_tok2_production import (
    Accounting,
    Document,
    RuntimeTokenizer,
    evaluate,
    lr_at_progress,
    train_documents,
)


HERE = Path(__file__).resolve().parent


class MiniLM(nn.Module):
    def __init__(self, vocab_size: int, dimension: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dimension)
        self.projection = nn.Linear(dimension, vocab_size, bias=False)
        self.projection.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embed(input_ids))


class Tok2ProductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec_path = HERE / "tok2_production_spec.json"
        cls.manifest_path = HERE / "tok2_production_manifest.json"
        cls.spec = json.loads(cls.spec_path.read_text())
        cls.manifest = json.loads(cls.manifest_path.read_text())

    def test_frozen_manifest_is_reproducible_and_complete(self):
        self.assertEqual(build_manifest(self.spec_path), self.manifest)
        self.assertEqual(len(self.manifest["runs"]), 48)
        self.assertEqual(
            {row["status"] for row in self.manifest["runs"]},
            {"registered-not-started"},
        )
        self.assertEqual(
            {
                (
                    row["control"],
                    row["arm_id"],
                    row["paired_run_label"],
                )
                for row in self.manifest["runs"]
            }.__len__(),
            48,
        )

    def test_every_phase_has_four_exact_document_boundaries(self):
        for phase in ("phase1", "phase3"):
            stream = self.manifest["streams"][phase]
            boundaries = stream["checkpoint_boundaries"]
            self.assertEqual(
                [row["fraction"] for row in boundaries],
                [0.25, 0.5, 0.75, 1.0],
            )
            self.assertEqual(boundaries[-1]["documents"], stream["documents"])
            self.assertEqual(boundaries[-1]["raw_bytes"], stream["actual_raw_bytes"])

    def test_archive_hashes_match_runtime_hashes(self):
        archive = json.loads(
            (HERE / "tok2_tokenizer_archive_manifest.json").read_text()
        )
        archived = {row["arm_id"]: row for row in archive["artifacts"]}
        for arm_id, row in archived.items():
            self.assertEqual(
                row["sha256"], self.manifest["tokenizers"][arm_id]["sha256"]
            )
            self.assertEqual(
                row["artifact_id"],
                self.manifest["tokenizers"][arm_id]["archive"]["artifact_id"],
            )

    def test_byte_scheduled_training_accounts_complete_documents(self):
        arm = {
            "id": "pure_byte",
            "kind": "pure_byte",
            "vocab_rows": 260,
        }
        tokenizer = RuntimeTokenizer(
            arm, HERE / "candidates/pure_byte_v0/pure_byte_v0.vocab.json"
        )
        texts = [" alpha", "\tbeta\n", "👩🏽‍💻 gamma"]
        documents = [
            Document(
                domain="probe",
                source="unit",
                text=text,
                text_sha256="unused",
                raw_bytes=len(text.encode()),
                line_number=index,
            )
            for index, text in enumerate(texts, 1)
        ]
        model = MiniLM(tokenizer.vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        accounting = Accounting()
        frequency = {}
        counter = __import__("collections").Counter(frequency)
        spec = copy.deepcopy(self.spec)
        spec["optimization"]["maximum_sequence_length"] = 8
        spec["optimization"]["batch_size_sequences"] = 2
        trainable = sum(parameter.numel() for parameter in model.parameters())
        train_documents(
            model=model,
            tokenizer=tokenizer,
            documents=documents,
            optimizer=optimizer,
            accounting=accounting,
            phase_raw_bytes=sum(document.raw_bytes for document in documents),
            phase_trainable_parameters=trainable,
            peak_lr=1e-3,
            spec=spec,
            device=torch.device("cpu"),
            token_frequency=counter,
        )
        self.assertEqual(accounting.documents_consumed, len(documents))
        self.assertEqual(
            accounting.raw_bytes_consumed,
            sum(document.raw_bytes for document in documents),
        )
        self.assertGreater(accounting.target_tokens_predicted, 0)
        self.assertEqual(
            accounting.estimated_training_flops,
            6 * trainable * accounting.target_tokens_predicted,
        )
        metrics = evaluate(model, tokenizer, documents, 8, torch.device("cpu"))
        self.assertTrue(math.isfinite(metrics["corpus_bpb"]))
        self.assertEqual(metrics["raw_bytes"], accounting.raw_bytes_consumed)

    def test_preregistered_statistics_helpers(self):
        interval = training_interval([-0.1, -0.2, -0.3])
        self.assertAlmostEqual(interval["mean_difference"], -0.2)
        self.assertEqual(interpolate([(1.0, 4.0), (3.0, 2.0)], 2.0), 3.0)
        self.assertIsNone(interpolate([(1.0, 4.0), (3.0, 2.0)], 0.5))
        rows_a = {
            "per_document": [
                {
                    "text_sha256": "a",
                    "raw_bytes": 10,
                    "loss_nats_sum": 4.0,
                },
                {
                    "text_sha256": "b",
                    "raw_bytes": 20,
                    "loss_nats_sum": 7.0,
                },
            ]
        }
        rows_b = {
            "per_document": [
                {
                    "text_sha256": "a",
                    "raw_bytes": 10,
                    "loss_nats_sum": 5.0,
                },
                {
                    "text_sha256": "b",
                    "raw_bytes": 20,
                    "loss_nats_sum": 9.0,
                },
            ]
        }
        bootstrap = paired_document_bootstrap(
            rows_a, rows_b, seed=7, replicates=100, interval=0.9
        )
        self.assertLess(bootstrap["point_difference_corpus_bpb"], 0)
        self.assertEqual(bootstrap["replicates"], 100)

    def test_lr_schedule_is_byte_parameterized(self):
        self.assertAlmostEqual(lr_at_progress(1.0, 0.01, 0.02, 0.1), 0.5)
        self.assertAlmostEqual(lr_at_progress(1.0, 1.0, 0.02, 0.1), 0.1)


if __name__ == "__main__":
    unittest.main()
