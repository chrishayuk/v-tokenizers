#!/usr/bin/env python3
"""Generate deterministic Unicode fuzz cases for tokenizer conformance."""

from __future__ import annotations

import argparse
import json
import random
import unicodedata


POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    " \t\n\r",
    ".,;:!?_+-=*/\\|&<>[]{}()\"'`#@$%^~",
    "áéíóúàèìòùäëïöüßøåæœçñ",
    "αβγδεζηθικλμνξοπρστυφχψω",
    "абвгдежзийклмнопрстуфхцчшщ",
    "日本語漢字かなカナ한국어",
    "العربيةעבריתहिन्दी",
    "🫠🚀🧪🔥✨👩🏽‍💻👨‍👩‍👧‍👦",
    "\u0301\u0308\u0327\u200b\u200c\u200d\u2060\ufe0e\ufe0f",
]


def _random_scalar(rng: random.Random) -> str:
    while True:
        codepoint = rng.randrange(0x110000)
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        char = chr(codepoint)
        if unicodedata.category(char) != "Cn":
            return char


def generated_cases(seed: int, count: int):
    rng = random.Random(seed)
    for index in range(count):
        parts = []
        for _ in range(rng.randint(1, 24)):
            if rng.random() < 0.04:
                parts.append(_random_scalar(rng))
            else:
                pool = rng.choice(POOLS)
                parts.append(rng.choice(pool))
        text = "".join(parts)
        yield {
            "id": f"random-{seed}-{index:04d}",
            "category": "random-unicode",
            "text": text,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    for case in generated_cases(args.seed, args.count):
        print(json.dumps(case, ensure_ascii=False))


if __name__ == "__main__":
    main()
