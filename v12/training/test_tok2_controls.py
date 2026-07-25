#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

from v12.training.validate_tok2_controls import parameter_counts, validate


class Tok2ControlTests(unittest.TestCase):
    def test_frozen_manifests_validate(self):
        validate()

    def test_reference_formula(self):
        path = Path(__file__).with_name("tok2_architecture_controls.json")
        config = json.loads(path.read_text())
        counts = parameter_counts(config, 71261, 2048)
        self.assertEqual(counts["nominal_parameters"], 115149824)
        self.assertEqual(counts["embedding_output_parameters"], 36485632)


if __name__ == "__main__":
    unittest.main()
