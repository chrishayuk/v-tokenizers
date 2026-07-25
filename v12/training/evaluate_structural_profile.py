#!/usr/bin/env python3
"""Evaluate a tokenizer against offset-annotated structural probes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


V12_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROBES = V12_ROOT / "targets" / "structural_profile.jsonl"


def contains_subsequence(sequence: list[int], candidate: list[int]) -> bool:
    if not candidate:
        return True
    width = len(candidate)
    return any(sequence[index : index + width] == candidate for index in range(len(sequence) - width + 1))


def load_probes(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(tokenizer: Tokenizer, probes: list[dict[str, Any]]) -> dict[str, Any]:
    totals = defaultdict(
        lambda: {
            "probes": 0,
            "roundtrip_passes": 0,
            "gold_boundaries": 0,
            "gold_boundaries_hit": 0,
            "token_boundaries": 0,
            "token_boundaries_correct": 0,
            "spans": 0,
            "atomic_spans": 0,
            "span_token_count": 0,
            "context_sequences_preserved": 0,
        }
    )

    for probe in probes:
        text = probe["text"]
        category = probe["category"]
        encoding = tokenizer.encode(text)
        offsets = list(encoding.offsets)
        token_boundaries = {
            boundary
            for start, end in offsets
            for boundary in (start, end)
            if 0 < boundary < len(text)
        }
        gold_boundaries = set(probe.get("boundaries", []))
        bucket = totals[category]
        bucket["probes"] += 1
        bucket["roundtrip_passes"] += tokenizer.decode(encoding.ids, skip_special_tokens=False) == text
        bucket["gold_boundaries"] += len(gold_boundaries)
        bucket["gold_boundaries_hit"] += len(gold_boundaries & token_boundaries)
        bucket["token_boundaries"] += len(token_boundaries)
        bucket["token_boundaries_correct"] += len(gold_boundaries & token_boundaries)

        for span in probe.get("spans", []):
            start, end = span["start"], span["end"]
            if not (0 <= start < end <= len(text)):
                raise ValueError(f"{probe['id']} has invalid span {span}")
            covering = [
                token_id
                for token_id, (token_start, token_end) in zip(encoding.ids, offsets)
                if token_end > start and token_start < end
            ]
            bucket["spans"] += 1
            bucket["span_token_count"] += len(covering)
            bucket["atomic_spans"] += (
                len(covering) == 1
                and any(token_start == start and token_end == end for token_start, token_end in offsets)
            )
            standalone_ids = tokenizer.encode(text[start:end]).ids
            bucket["context_sequences_preserved"] += contains_subsequence(encoding.ids, standalone_ids)

    def finalize(raw: dict[str, int]) -> dict[str, Any]:
        gold = raw["gold_boundaries"]
        predicted = raw["token_boundaries"]
        spans = raw["spans"]
        probes_count = raw["probes"]
        return {
            **raw,
            "roundtrip_rate": raw["roundtrip_passes"] / probes_count if probes_count else None,
            "boundary_recall": raw["gold_boundaries_hit"] / gold if gold else None,
            "boundary_precision": raw["token_boundaries_correct"] / predicted if predicted else None,
            "atomic_span_rate": raw["atomic_spans"] / spans if spans else None,
            "mean_identifier_or_unit_fragmentation": raw["span_token_count"] / spans if spans else None,
            "context_sequence_preservation_rate": (
                raw["context_sequences_preserved"] / spans if spans else None
            ),
        }

    aggregate = {
        key: sum(bucket[key] for bucket in totals.values())
        for key in next(iter(totals.values())).keys()
    } if totals else {}
    return {
        "schema_version": 1,
        "offset_unit": "Unicode code points",
        "categories": {category: finalize(dict(raw)) for category, raw in sorted(totals.items())},
        "overall": finalize(aggregate) if aggregate else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    result = evaluate(tokenizer, load_probes(args.probes))
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
