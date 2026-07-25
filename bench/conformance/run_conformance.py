#!/usr/bin/env python3
"""Differential conformance runner for v-tokenizer artifacts.

Always checks Hugging Face `tokenizers`. Optionally compares local
`transformers.AutoTokenizer`, the current Rust CLI, and the installed `v11`
Python binding. The generated lane is deterministic and dependency-free.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from generate_cases import generated_cases


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_ARTIFACTS = REPO_ROOT / "v11" / "artifacts"


def load_cases(path: Path, seed: int, random_count: int) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cases.extend(generated_cases(seed, random_count))
    return cases


def chunk_points(text: str, seed_material: str) -> list[int]:
    if len(text) < 2:
        return []
    candidates = {1, len(text) - 1, len(text) // 2}
    for index, char in enumerate(text):
        if char in "\r\n\t \u200b\u200d" and 0 < index < len(text):
            candidates.add(index)
            if index + 1 < len(text):
                candidates.add(index + 1)
    # Stable extra boundary without relying on process-randomized hash().
    candidates.add(1 + sum(seed_material.encode("utf-8")) % (len(text) - 1))
    return sorted(candidates)


def validate_offsets(text: str, offsets: list[tuple[int, int]]) -> list[str]:
    failures = []
    previous_start = 0
    covered: set[int] = set()
    for index, (start, end) in enumerate(offsets):
        if not (0 <= start <= end <= len(text)):
            failures.append(f"offset[{index}] out of range: {(start, end)} for len={len(text)}")
        if start < previous_start:
            failures.append(f"offset[{index}] moves backwards: {(start, end)} after start={previous_start}")
        previous_start = start
        covered.update(range(start, end))
    if text and covered != set(range(len(text))):
        missing = sorted(set(range(len(text))) - covered)
        failures.append(f"offsets do not cover character positions {missing[:12]}")
    return failures


class RustCli:
    name = "rust-cli"

    def __init__(self, binary: Path, model: Path, use_batch: bool = False):
        self.binary = binary
        self.model = model
        self.use_batch = use_batch
        self.encoded: dict[str, list[int]] = {}
        self.decoded: dict[tuple[int, ...], str] = {}

    def prepare(self, texts: list[str]) -> None:
        if not self.use_batch:
            return
        result = subprocess.run(
            [str(self.binary), "--model", str(self.model), "batch"],
            input=json.dumps(texts, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            check=True,
        )
        for row in json.loads(result.stdout.decode("utf-8")):
            ids = list(row["ids"])
            self.encoded[row["text"]] = ids
            self.decoded[tuple(ids)] = row["decoded"]

    def encode(self, text: str) -> list[int]:
        if text in self.encoded:
            return self.encoded[text]
        base = [str(self.binary), "--model", str(self.model), "encode", "--json"]
        if any(ord(char) < 32 for char in text):
            result = subprocess.run(
                base,
                input=text.encode("utf-8"),
                capture_output=True,
                check=True,
            )
        else:
            result = subprocess.run(
                base + ["--text", text],
                capture_output=True,
                check=True,
            )
        return json.loads(result.stdout.decode("utf-8"))["ids"]

    def decode(self, ids: list[int]) -> str:
        if tuple(ids) in self.decoded:
            return self.decoded[tuple(ids)]
        if not ids:
            return ""
        result = subprocess.run(
            [str(self.binary), "--model", str(self.model), "decode", "--ids", ",".join(map(str, ids))],
            capture_output=True,
            check=True,
        )
        decoded = result.stdout.decode("utf-8")
        return decoded[:-1] if decoded.endswith("\n") else decoded


class PythonBinding:
    name = "python-binding"

    def __init__(self, model: Path):
        module = importlib.import_module("v11")
        self.version = getattr(module, "__version__", "unknown")
        self.tokenizer = module.Tokenizer.from_file(str(model))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


class WhitespaceExactBinding:
    """Apply the experiment adapter contract to a native runtime binding."""

    def __init__(self, binding: Any, space_byte_id: int):
        self.binding = binding
        self.space_byte_id = space_byte_id
        self.name = f"{binding.name}-ws-exact"
        self.version = binding.version

    def encode(self, text: str) -> list[int]:
        ids = self.binding.encode(text)
        return [self.space_byte_id, *ids] if text.startswith(" ") else ids

    def decode(self, ids: list[int]) -> str:
        if ids and ids[0] == self.space_byte_id:
            return " " + self.binding.decode(ids[1:])
        return self.binding.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=HERE / "cases.jsonl")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--random-count", type=int, default=200)
    parser.add_argument("--auto-tokenizer", action="store_true")
    parser.add_argument("--rust-cli", type=Path)
    parser.add_argument("--python-binding", action="store_true")
    parser.add_argument(
        "--ws-exact",
        action="store_true",
        help="test the separately named v11-ws-exact adapter over canonical v11",
    )
    parser.add_argument("--max-failures", type=int, default=50)
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    base_hf = Tokenizer.from_file(str(artifacts / "tokenizer.json"))
    if args.ws_exact:
        sys.path.insert(0, str(REPO_ROOT))
        from v12.training.v11_ws_exact import (
            WhitespaceExactAutoTokenizer,
            WhitespaceExactV11,
        )

        hf = WhitespaceExactV11(base_hf)
    else:
        hf = base_hf
    cases = load_cases(args.cases, args.seed, args.random_count)
    adapters = []
    metadata: dict[str, Any] = {
        "seed": args.seed,
        "static_cases": len(cases) - args.random_count,
        "random_cases": args.random_count,
        "tokenizer_json": str(artifacts / "tokenizer.json"),
        "adapter": "v11-ws-exact" if args.ws_exact else None,
    }

    auto = None
    if args.auto_tokenizer:
        if args.ws_exact:
            auto = WhitespaceExactAutoTokenizer.from_pretrained(artifacts)
            metadata["transformers_class"] = type(auto.tokenizer).__name__
        else:
            from transformers import AutoTokenizer

            auto = AutoTokenizer.from_pretrained(artifacts, local_files_only=True)
            metadata["transformers_class"] = type(auto).__name__
    if args.rust_cli:
        adapters.append(
            RustCli(
                args.rust_cli.resolve(),
                artifacts / "v11.vocab.bin",
                use_batch=args.ws_exact,
            )
        )
    if args.python_binding:
        binding = PythonBinding(artifacts / "v11.vocab.bin")
        if args.ws_exact:
            binding = WhitespaceExactBinding(binding, hf.space_byte_id)
        adapters.append(binding)
        metadata["python_binding_version"] = binding.version

    for adapter in adapters:
        prepare = getattr(adapter, "prepare", None)
        if prepare is not None:
            prepare([case["text"] for case in cases])

    failures: list[dict[str, Any]] = []
    chunk_failures: list[dict[str, Any]] = []
    chunk_id_mismatches: list[dict[str, Any]] = []
    category_counts = Counter(case["category"] for case in cases)
    checks = 0

    def fail(case, check, detail):
        if len(failures) < args.max_failures:
            failures.append({"case": case["id"], "category": case["category"], "check": check, "detail": detail})

    for case in cases:
        text = case["text"]
        expect_roundtrip = case.get("expect_decode_roundtrip", True)
        expect_id_parity = case.get("expect_id_parity", True)
        encoding = hf.encode(text)
        ids = encoding.ids
        checks += 1
        if hf.decode(ids, skip_special_tokens=False) != text and expect_roundtrip:
            fail(case, "hf-roundtrip", repr(hf.decode(ids, skip_special_tokens=False)))

        for detail in validate_offsets(text, encoding.offsets):
            fail(case, "hf-offsets", detail)
        checks += 1

        for point in chunk_points(text, case["id"]):
            reconstructed = ""
            chunk_ids_combined: list[int] = []
            for chunk in (text[:point], text[point:]):
                chunk_ids = hf.encode(chunk).ids
                chunk_ids_combined.extend(chunk_ids)
                reconstructed += hf.decode(chunk_ids, skip_special_tokens=False)
            checks += 1
            if chunk_ids_combined != ids and len(chunk_id_mismatches) < args.max_failures:
                chunk_id_mismatches.append(
                    {
                        "case": case["id"],
                        "category": case["category"],
                        "split": point,
                        "whole_ids": ids,
                        "chunk_ids": chunk_ids_combined,
                    }
                )
            if reconstructed != text and expect_roundtrip:
                if len(chunk_failures) < args.max_failures:
                    chunk_failures.append(
                        {
                            "case": case["id"],
                            "category": case["category"],
                            "split": point,
                            "reconstructed": reconstructed,
                        }
                    )

        if auto is not None:
            if args.ws_exact:
                auto_encoding = auto.encode(text)
                auto_ids = auto_encoding.ids
                auto_offsets = auto_encoding.offsets
                decoded = auto.decode(auto_ids, skip_special_tokens=False)
            else:
                auto_encoding = auto(text, add_special_tokens=False, return_offsets_mapping=True)
                auto_ids = list(auto_encoding["input_ids"])
                auto_offsets = [tuple(pair) for pair in auto_encoding["offset_mapping"]]
                decoded = auto.decode(
                    auto_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            checks += 2
            if auto_ids != ids and expect_id_parity:
                fail(case, "auto-ids", f"hf={ids} auto={auto_ids}")
            if decoded != text and expect_roundtrip:
                fail(case, "auto-roundtrip", repr(decoded))
            for detail in validate_offsets(text, auto_offsets):
                fail(case, "auto-offsets", detail)

        for adapter in adapters:
            try:
                adapter_ids = adapter.encode(text)
                checks += 2
                if adapter_ids != ids and expect_id_parity:
                    fail(case, f"{adapter.name}-ids", f"hf={ids} adapter={adapter_ids}")
                decoded = adapter.decode(adapter_ids)
                if decoded != text and expect_roundtrip:
                    fail(case, f"{adapter.name}-roundtrip", repr(decoded))
            except Exception as exc:
                fail(case, f"{adapter.name}-exception", repr(exc))

    result = {
        **metadata,
        "cases": len(cases),
        "categories": dict(sorted(category_counts.items())),
        "checks": checks,
        "failures_returned": len(failures),
        "independent_chunk_roundtrip_failures_returned": len(chunk_failures),
        "independent_chunk_id_mismatches_returned": len(chunk_id_mismatches),
        "independent_chunk_roundtrip_note": (
            "diagnostic only: a streaming tokenizer must carry boundary state; "
            "round-trip failures and token-ID mismatches do not change pass"
        ),
        "pass": not failures,
        "failures": failures,
        "independent_chunk_roundtrip_failures": chunk_failures,
        "independent_chunk_id_mismatches": chunk_id_mismatches,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
