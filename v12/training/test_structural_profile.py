#!/usr/bin/env python3

import unittest

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from v12.training.evaluate_structural_profile import contains_subsequence, evaluate


class StructuralProfileTests(unittest.TestCase):
    def test_subsequence(self):
        self.assertTrue(contains_subsequence([1, 2, 3, 4], [2, 3]))
        self.assertFalse(contains_subsequence([1, 2, 3, 4], [2, 4]))

    def test_profile_reports_boundaries_and_spans(self):
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(" ", behavior="isolated")
        tokenizer.decoder = decoders.ByteFallback()
        tokenizer.train_from_iterator(
            ["cat cats sleep sleeping"],
            trainers.BpeTrainer(vocab_size=30, special_tokens=["<unk>"], show_progress=False),
        )
        probes = [
            {
                "id": "morph",
                "category": "morpheme",
                "text": "cats sleeping",
                "boundaries": [3, 4, 5, 10],
                "spans": [
                    {"start": 0, "end": 3, "label": "cat"},
                    {"start": 3, "end": 4, "label": "plural"},
                    {"start": 5, "end": 10, "label": "sleep"},
                    {"start": 10, "end": 13, "label": "progressive"},
                ],
            }
        ]
        result = evaluate(tokenizer, probes)
        category = result["categories"]["morpheme"]
        self.assertEqual(category["probes"], 1)
        self.assertEqual(category["spans"], 4)
        self.assertEqual(category["roundtrip_rate"], 1.0)
        self.assertGreaterEqual(category["boundary_recall"], 0.0)
        self.assertLessEqual(category["boundary_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
