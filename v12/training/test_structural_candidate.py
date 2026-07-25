#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer

from v12.training.pretokenization import build_pre_tokenizer
from v12.training.train_structural_candidate import train


class StructuralCandidateTests(unittest.TestCase):
    def test_structural_boundaries(self):
        text = "HTTPServer parseJSON foo_bar(x+=42)"
        whitespace = [part for part, _ in build_pre_tokenizer("whitespace_split").pre_tokenize_str(text)]
        digits = [part for part, _ in build_pre_tokenizer("digit_isolating").pre_tokenize_str(text)]
        code = [part for part, _ in build_pre_tokenizer("code_aware").pre_tokenize_str(text)]
        self.assertIn("foo_bar(x+=42)", whitespace)
        self.assertIn("4", digits)
        self.assertIn("2", digits)
        self.assertEqual(code[:2], ["HTTP", "Server"])
        self.assertIn("_", code)
        self.assertIn("+=", code)

    def test_saved_candidates_round_trip_unicode_and_control_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "fixture.jsonl"
            texts = [
                "hello  world\n",
                "HTTPServer parseJSON foo_bar(x+=42)",
                "café cafe\u0301 Ελληνικά 日本語",
            ]
            corpus.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
            probes = texts + ["unseen 🫠\u200d🚀", "a\x00b", "\r\n\tmixed whitespace"]

            for algorithm in ("unigram", "bpe"):
                for pre_tokenization in ("whitespace_split", "digit_isolating", "code_aware"):
                    out = root / f"{algorithm}-{pre_tokenization}"
                    metadata = train(
                        algorithm,
                        300,
                        pre_tokenization,
                        corpus,
                        out,
                        out.name,
                        show_progress=False,
                    )
                    tokenizer = Tokenizer.from_file(str(out / "tokenizer.json"))
                    self.assertEqual(metadata["special_token_ids"], {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3})
                    for probe in probes:
                        encoding = tokenizer.encode(probe)
                        self.assertNotIn(1, encoding.ids)
                        self.assertEqual(tokenizer.decode(encoding.ids), probe)


if __name__ == "__main__":
    unittest.main()
