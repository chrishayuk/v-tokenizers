#!/usr/bin/env python3
"""Builds the real T-core target set from v11's own config.json -- the same
real data v11's vocab was built to guarantee single-token coverage for, not
a fabricated or hand-picked list.

Replaces the 31-row illustrative stub. Pulls three real categories:
  language     v11/config.json's `language_extras` -- real per-language
               priority builtins/types for the 10 priority_languages
               (python, rust, c, cpp, javascript, typescript, java, go,
               ruby, swift). 379 items.
  symbol       v11/config.json's `infra` math/greek/legal/currency
               categories (excludes plain ASCII/digits/punct, which are
               infra not "priority concepts"). 231 items.
  acronym      v11/config.json's `acronyms` list. 46 items.

function_words is deliberately excluded -- generic high-frequency words
are single-token for nearly any subword tokenizer and don't discriminate
between candidates the way priority/technical concepts do (matches the
design doc's own framing of the priority-target suite: "keywords,
technical acronyms, Greek, scientific terms", not "the/a/an").

Run: python3 v12/targets/build_t_core.py
"""
import json
from pathlib import Path

TARGETS_DIR = Path(__file__).resolve().parent
V11_CONFIG = TARGETS_DIR.parent.parent / "v11" / "config.json"
OUT = TARGETS_DIR / "t_core.jsonl"

SYMBOL_CATEGORIES = [
    "greek_lower", "greek_upper",
    "math_operators", "math_relations", "math_logic", "math_sets",
    "math_arrows", "math_fractions", "super", "legal", "currency",
]


def main():
    cfg = json.loads(V11_CONFIG.read_text())
    # text -> set of sources (a token like "int" is priority across several
    # languages -- keep every source for provenance, but the target SET
    # should count each concept once. t_core_fertility measures whether
    # *concepts* get single-token treatment; deduping keeps the measurement
    # about concept diversity, not amplified by how many languages happen
    # to share a keyword like "int"/"void"/"float".
    by_text = {}

    for lang, items in sorted(cfg["language_extras"].items()):
        for tok in items:
            by_text.setdefault(tok, {"domain": "core", "sources": []})["sources"].append(f"language_extras.{lang}")

    infra = cfg["infra"]
    for cat in SYMBOL_CATEGORIES:
        for sym in infra[cat]:
            by_text.setdefault(sym, {"domain": "core", "sources": []})["sources"].append(f"infra.{cat}")

    for acro in cfg["acronyms"]:
        by_text.setdefault(acro, {"domain": "core", "sources": []})["sources"].append("acronyms")

    rows = [
        {"text": text, "domain": v["domain"], "source": "v11/config.json " + "+".join(v["sources"])}
        for text, v in sorted(by_text.items())
    ]

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    category_counts = {}
    for v in by_text.values():
        for s in v["sources"]:
            cat = s.split(".")[0]
            category_counts[cat] = category_counts.get(cat, 0) + 1
    print(json.dumps({
        "total_unique_rows": len(rows),
        "raw_occurrences_before_dedup": sum(len(v["sources"]) for v in by_text.values()),
        "category_occurrence_counts": category_counts,
        "note": "real data from v11/config.json (language_extras/infra/acronyms), "
                "not fabricated -- replaces the 31-row illustrative stub. Deduped "
                "by text -- a token shared across languages (e.g. 'int') appears "
                "once, with all its real sources recorded.",
    }, indent=2))


if __name__ == "__main__":
    main()
