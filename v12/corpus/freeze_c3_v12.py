#!/usr/bin/env python3
"""Draw the fresh, domain-balanced C3 v12 exclusion slice.

The selection is content-hash based and is run before TOK-2 model results.
The full held-out text remains a generated artifact; the tracked manifest
contains the exact text hashes needed to prove and enforce exclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(source_path: Path, domain: str) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            text = raw.get("text")
            if not isinstance(text, str) or not text:
                continue
            text_sha = sha256_bytes(text.encode("utf-8"))
            rows.setdefault(
                text_sha,
                {
                    "domain": domain,
                    "source": str(raw.get("source") or f"{source_path.name}:{line_number}"),
                    "text": text,
                    "text_sha256": text_sha,
                },
            )
    return list(rows.values())


def freeze(
    spec_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION}")
    seed = int(spec["seed"])
    items_per_domain = int(spec["items_per_domain"])
    selected: list[dict[str, str]] = []
    source_evidence: dict[str, Any] = {}

    for domain, configured in spec["domains"].items():
        source_path = (spec_path.parent / configured["path"]).resolve()
        actual_sha = sha256_file(source_path)
        if actual_sha != configured["sha256"]:
            raise ValueError(
                f"C3 source digest mismatch for {domain}: "
                f"expected {configured['sha256']}, got {actual_sha}"
            )
        rows = load_rows(source_path, domain)
        maximum_text_bytes = configured.get("maximum_text_bytes")
        if maximum_text_bytes is not None:
            rows = [
                row
                for row in rows
                if len(row["text"].encode("utf-8")) <= int(maximum_text_bytes)
            ]
        rows.sort(
            key=lambda row: (
                sha256_bytes(
                    f"{seed}\0{domain}\0{row['text_sha256']}".encode()
                ),
                row["text_sha256"],
            )
        )
        if len(rows) < items_per_domain:
            raise ValueError(f"{domain} has only {len(rows)} unique rows")
        domain_selected = rows[:items_per_domain]
        selected_bytes = sum(
            len(row["text"].encode("utf-8")) for row in domain_selected
        )
        maximum_heldout_bytes = configured.get("maximum_heldout_bytes")
        if (
            maximum_heldout_bytes is not None
            and selected_bytes > int(maximum_heldout_bytes)
        ):
            raise ValueError(
                f"{domain} heldout uses {selected_bytes} bytes, above "
                f"the predeclared {maximum_heldout_bytes}-byte bound"
            )
        selected.extend(domain_selected)
        source_evidence[domain] = {
            "path": configured["path"],
            "sha256": actual_sha,
            "unique_rows": len(rows),
            "maximum_text_bytes": maximum_text_bytes,
            "selected_text_bytes": selected_bytes,
            "maximum_heldout_bytes": maximum_heldout_bytes,
            "constraint_reason": configured.get("constraint_reason"),
        }

    selected.sort(key=lambda row: (row["domain"], row["text_sha256"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(
                json.dumps(
                    {
                        "domain": row["domain"],
                        "source": row["source"],
                        "text": row["text"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    output_sha = sha256_file(output_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "slice_id": spec["slice_id"],
        "status": "frozen-before-tok2-results",
        "frozen_date": spec["frozen_date"],
        "seed": seed,
        "selection": "lowest sha256(seed, domain, text_sha256), without replacement",
        "items_per_domain": items_per_domain,
        "items_total": len(selected),
        "source_evidence": source_evidence,
        "heldout_jsonl": output_path.name,
        "heldout_jsonl_sha256": output_sha,
        "items": [
            {
                "domain": row["domain"],
                "source": row["source"],
                "text_sha256": row["text_sha256"],
                "text_bytes": len(row["text"].encode("utf-8")),
            }
            for row in selected
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=Path(__file__).with_name("c3_v12_spec.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("c3_v12_heldout.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("c3_v12_manifest.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.spec.resolve(),
                args.output.resolve(),
                args.manifest.resolve(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
