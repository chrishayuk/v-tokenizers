#!/usr/bin/env python3

import json
import hashlib
import unittest
from pathlib import Path

from tokenizers import Tokenizer

from v12.training.v11_ws_exact import WhitespaceExactV11


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_JSON = REPO_ROOT / "v11" / "artifacts" / "tokenizer.json"
CASES = REPO_ROOT / "bench" / "conformance" / "cases.jsonl"
MANIFEST = Path(__file__).with_name("v11_ws_exact_manifest.json")


class WhitespaceExactV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Tokenizer.from_file(str(TOKENIZER_JSON))
        cls.exact = WhitespaceExactV11(cls.base)

    def test_no_rows_added_and_existing_space_row_used(self):
        self.assertEqual(self.exact.vocab_size, 71_260)
        self.assertEqual(self.exact.space_byte_id, self.base.token_to_id("<0x20>"))

    def test_non_leading_inputs_keep_canonical_ids(self):
        for text in ["", "x", "\tx", "\n x", "a  b", "café", "👩‍💻"]:
            self.assertEqual(self.exact.encode_ids(text), self.base.encode(text).ids, repr(text))

    def test_leading_spaces_use_existing_byte_fallback(self):
        expected = [self.exact.space_byte_id] + self.base.encode("  x").ids
        encoding = self.exact.encode("  x")
        self.assertEqual(encoding.ids, expected)
        self.assertEqual(encoding.offsets[0], (0, 1))
        self.assertEqual(self.exact.decode(encoding.ids), "  x")

    def test_leading_space_keeps_canonical_suffix(self):
        for text in [" x", "  x", "   "]:
            encoding = self.exact.encode(text)
            self.assertEqual(encoding.ids[0], self.exact.space_byte_id)
            self.assertEqual(encoding.ids[1:], self.base.encode(text).ids)
            self.assertEqual(self.exact.decode(encoding.ids), text)

    def test_whole_input_contract_on_conformance_corpus(self):
        for line in CASES.read_text().splitlines():
            case = json.loads(line)
            if not case.get("expect_decode_roundtrip", True):
                continue
            text = case["text"]
            self.assertEqual(self.exact.decode(self.exact.encode_ids(text)), text, case["id"])

    def test_frozen_manifest_pins_artifacts_and_implementations(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["base_artifact"]["model_visible_rows"], 71_260)
        self.assertEqual(manifest["contract"]["rows_added"], 0)
        self.assertEqual(manifest["contract"]["leading_ascii_space_sentinel_id"], 36)
        pinned_files = [
            (
                REPO_ROOT / "v11" / "artifacts" / "tokenizer.json",
                manifest["base_artifact"]["tokenizer_json_sha256"],
            ),
            (
                REPO_ROOT / "v11" / "artifacts" / "v11.vocab.bin",
                manifest["base_artifact"]["vocab_bin_sha256"],
            ),
            (
                REPO_ROOT / "v11" / "artifacts" / "v11.vocab.json",
                manifest["base_artifact"]["vocab_json_sha256"],
            ),
            (
                REPO_ROOT / "v12" / "v11-ws-exact" / "src" / "lib.rs",
                manifest["implementations"]["rust"]["sha256"],
            ),
            (
                Path(__file__).with_name("v11_ws_exact.py"),
                manifest["implementations"]["python_hf_and_transformers"]["sha256"],
            ),
        ]
        for path, expected in pinned_files:
            with self.subTest(path=str(path)):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
