#!/usr/bin/env python3
"""Build a deterministic C8 v3 corpus with byte-exact domain budgets.

The builder is deliberately source-agnostic. A JSON specification supplies
one or more JSONL inputs for each domain, optional row filters, and pinned
SHA-256 digests. Domain shares are converted to integer byte targets with the
largest-remainder method. Rows are deduplicated, deterministically ordered,
and the final row is UTF-8-safely truncated to hit each target exactly.

This script never repeats rows to fill a thin domain. A build fails instead,
making corpus coverage gaps visible before tokenizer training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceRow:
    text: str
    source: str
    text_sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utf8_prefix(text: str, byte_budget: int) -> str:
    """Return the longest code-point-safe prefix no larger than byte_budget."""
    if byte_budget < 0:
        raise ValueError("byte_budget must be non-negative")
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_budget:
        return text
    return encoded[:byte_budget].decode("utf-8", errors="ignore")


def allocate_byte_targets(total_bytes: int, shares: dict[str, Any]) -> dict[str, int]:
    """Allocate total_bytes exactly using stable largest remainders."""
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    decimals = {name: Decimal(str(value)) for name, value in shares.items()}
    if not decimals or any(value <= 0 for value in decimals.values()):
        raise ValueError("every domain share must be positive")
    if abs(sum(decimals.values()) - Decimal("1")) > Decimal("0.000000001"):
        raise ValueError(f"domain shares must sum to 1.0, got {sum(decimals.values())}")

    exact = {name: Decimal(total_bytes) * share for name, share in decimals.items()}
    targets = {name: math.floor(value) for name, value in exact.items()}
    remainder = total_bytes - sum(targets.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - targets[name]), name))
    for name in order[:remainder]:
        targets[name] += 1
    return targets


def _matches(row: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    field = where["field"]
    value = row.get(field)
    if "equals" in where:
        return value == where["equals"]
    if "contains" in where:
        return isinstance(value, str) and where["contains"] in value
    if "not_contains" in where:
        return not isinstance(value, str) or where["not_contains"] not in value
    raise ValueError(f"unsupported row filter: {where}")


def _resolve(spec_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else (spec_path.parent / path).resolve()


def load_source_rows(
    spec_path: Path,
    source_spec: dict[str, Any],
    require_pins: bool,
) -> tuple[list[SourceRow], dict[str, Any]]:
    path = _resolve(spec_path, source_spec["path"])
    actual_sha256 = sha256_file(path)
    expected_sha256 = source_spec.get("sha256")
    if require_pins and not expected_sha256:
        raise ValueError(f"frozen build requires sha256 for {path}")
    if expected_sha256 and expected_sha256 != actual_sha256:
        raise ValueError(
            f"source digest mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )

    text_field = source_spec.get("text_field", "text")
    source_field = source_spec.get("source_field", "source")
    rows: list[SourceRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not _matches(raw, source_spec.get("where")):
                continue
            text = raw.get(text_field)
            if not isinstance(text, str) or not text:
                continue
            source = raw.get(source_field) or f"{path.name}:{line_number}"
            text_digest = sha256_bytes(text.encode("utf-8"))
            rows.append(SourceRow(text=text, source=str(source), text_sha256=text_digest))

    provenance = {
        "path": str(path),
        "sha256": actual_sha256,
        "rows_eligible": len(rows),
        "bytes_eligible_before_dedup": sum(len(row.text.encode("utf-8")) for row in rows),
        "where": source_spec.get("where"),
        "text_field": text_field,
    }
    return rows, provenance


def load_excluded_hashes(
    spec_path: Path,
    exclusion_spec: dict[str, Any] | None,
    require_pins: bool,
) -> tuple[set[str], dict[str, Any] | None]:
    if not exclusion_spec:
        return set(), None
    path = _resolve(spec_path, exclusion_spec["path"])
    actual_sha256 = sha256_file(path)
    expected_sha256 = exclusion_spec.get("sha256")
    if require_pins and not expected_sha256:
        raise ValueError(f"frozen build requires a C3 manifest sha256 for {path}")
    if expected_sha256 and expected_sha256 != actual_sha256:
        raise ValueError(
            f"C3 manifest digest mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    manifest = json.loads(path.read_text())
    hashes = {
        item["text_sha256"]
        for item in manifest.get("items", [])
        if isinstance(item.get("text_sha256"), str)
    }
    if require_pins and len(hashes) != manifest.get("items_total"):
        raise ValueError(
            f"C3 manifest item/hash count mismatch: "
            f"{len(hashes)} unique hashes for {manifest.get('items_total')} items"
        )
    return hashes, {
        "path": exclusion_spec["path"],
        "sha256": actual_sha256,
        "slice_id": manifest.get("slice_id"),
        "items_total": manifest.get("items_total"),
        "unique_text_hashes": len(hashes),
        "heldout_jsonl_sha256": manifest.get("heldout_jsonl_sha256"),
    }


def deterministic_rows(rows: Iterable[SourceRow], seed: int, domain: str) -> list[SourceRow]:
    unique: dict[str, SourceRow] = {}
    for row in rows:
        unique.setdefault(row.text_sha256, row)

    def order_key(row: SourceRow) -> tuple[str, str]:
        material = f"{seed}\0{domain}\0{row.text_sha256}".encode()
        return sha256_bytes(material), row.text_sha256

    return sorted(unique.values(), key=order_key)


def select_to_byte_budget(rows: list[SourceRow], byte_budget: int) -> list[SourceRow]:
    available = sum(len(row.text.encode("utf-8")) for row in rows)
    if available < byte_budget:
        raise ValueError(
            f"domain has {available:,} unique bytes but needs {byte_budget:,}; "
            "add source data rather than repeating rows"
        )

    chosen: list[SourceRow] = []
    used = 0
    for row in rows:
        remaining = byte_budget - used
        if remaining == 0:
            break
        row_bytes = len(row.text.encode("utf-8"))
        if row_bytes <= remaining:
            chosen.append(row)
            used += row_bytes
            continue
        prefix = utf8_prefix(row.text, remaining)
        if prefix:
            chosen.append(
                SourceRow(
                    text=prefix,
                    source=f"{row.source}#utf8-prefix",
                    text_sha256=sha256_bytes(prefix.encode("utf-8")),
                )
            )
            used += len(prefix.encode("utf-8"))

    if used != byte_budget:
        # A target ending inside a multi-byte code point can leave at most
        # three bytes. Exact shares matter more than preserving row order, so
        # fill that tiny remainder with a prefix from an unused ASCII-bearing
        # row. In normal text corpora this path is rarely reached.
        remaining = byte_budget - used
        for row in rows[len(chosen) :]:
            ascii_prefix = "".join(ch for ch in row.text if ord(ch) < 128)[:remaining]
            if len(ascii_prefix.encode("utf-8")) == remaining:
                chosen.append(
                    SourceRow(
                        text=ascii_prefix,
                        source=f"{row.source}#ascii-byte-fill",
                        text_sha256=sha256_bytes(ascii_prefix.encode("utf-8")),
                    )
                )
                used += remaining
                break
    if used != byte_budget:
        raise ValueError(f"could not fill byte target exactly: selected {used}, target {byte_budget}")
    return chosen


def build(spec_path: Path, output_path: Path, manifest_path: Path, total_bytes: int | None = None) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION}")

    build_total = total_bytes or int(spec["total_bytes"])
    domains = spec["domains"]
    shares = {name: cfg["share"] for name, cfg in domains.items()}
    targets = allocate_byte_targets(build_total, shares)
    require_pins = spec.get("status") == "frozen"
    seed = int(spec["seed"])
    excluded_hashes, exclusion_provenance = load_excluded_hashes(
        spec_path, spec.get("c3_exclusion"), require_pins
    )

    selected_by_domain: dict[str, list[SourceRow]] = {}
    domain_manifest: dict[str, Any] = {}
    global_hashes: set[str] = set()

    for domain, domain_spec in domains.items():
        loaded: list[SourceRow] = []
        provenances = []
        for source_spec in domain_spec["sources"]:
            source_rows, provenance = load_source_rows(spec_path, source_spec, require_pins)
            loaded.extend(source_rows)
            provenances.append(provenance)

        ordered = deterministic_rows(loaded, seed, domain)
        c3_duplicates = sum(row.text_sha256 in excluded_hashes for row in ordered)
        ordered = [row for row in ordered if row.text_sha256 not in excluded_hashes]
        cross_domain_duplicates = sum(row.text_sha256 in global_hashes for row in ordered)
        ordered = [row for row in ordered if row.text_sha256 not in global_hashes]
        selected = select_to_byte_budget(ordered, targets[domain])
        selected_by_domain[domain] = selected
        global_hashes.update(row.text_sha256 for row in selected)
        actual_bytes = sum(len(row.text.encode("utf-8")) for row in selected)
        domain_manifest[domain] = {
            "share_target": float(Decimal(str(domain_spec["share"]))),
            "bytes_target": targets[domain],
            "bytes_actual": actual_bytes,
            "rows_selected": len(selected),
            "eligible_rows_after_domain_dedup": len(ordered),
            "c3_rows_excluded": c3_duplicates,
            "cross_domain_duplicates_removed": cross_domain_duplicates,
            "sources": provenances,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = [
        (sha256_bytes(f"{seed}\0{domain}\0{row.text_sha256}".encode()), domain, row)
        for domain, rows in selected_by_domain.items()
        for row in rows
    ]
    interleaved.sort(key=lambda item: item[0])
    with output_path.open("w", encoding="utf-8") as handle:
        for _, domain, row in interleaved:
            handle.write(
                json.dumps(
                    {"domain": domain, "source": row.source, "text": row.text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    output_sha256 = sha256_file(output_path)
    expected_output = spec.get("expected_output")
    if require_pins:
        if not expected_output:
            raise ValueError("frozen C8 spec must pin expected_output")
        expected_sha256 = expected_output.get("sha256")
        if output_sha256 != expected_sha256:
            raise ValueError(
                f"C8 output digest mismatch: expected {expected_sha256}, "
                f"got {output_sha256}"
            )
        expected_rows = expected_output.get("rows")
        if expected_rows is not None and len(interleaved) != int(expected_rows):
            raise ValueError(
                f"C8 output row-count mismatch: expected {expected_rows}, "
                f"got {len(interleaved)}"
            )
    selected_hashes = {
        sha256_bytes(row.text.encode("utf-8"))
        for rows in selected_by_domain.values()
        for row in rows
    }
    c3_overlap = selected_hashes & excluded_hashes
    if c3_overlap:
        raise ValueError(f"final C8 output overlaps C3 by {len(c3_overlap)} text hashes")
    actual_total = sum(item["bytes_actual"] for item in domain_manifest.values())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": spec["corpus_id"],
        "spec_status": spec.get("status", "draft"),
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "seed": seed,
        "total_bytes_target": build_total,
        "total_bytes_actual": actual_total,
        "domain_shares_constant_across_scales": True,
        "selection": "sha256(seed, domain, text_sha256), without replacement",
        "deduplication": "exact UTF-8 text SHA-256 within and across domains",
        "normalization": "identity; source Unicode and line endings are preserved",
        "c3_exclusion": exclusion_provenance,
        "c3_overlap_text_hashes": len(c3_overlap),
        "domains": domain_manifest,
        "output_jsonl": str(output_path),
        "output_sha256": output_sha256,
        "rows": len(interleaved),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path(__file__).with_name("c8_v3_spec.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("c8_corpus_v3.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("c8_manifest_v3.json"))
    parser.add_argument(
        "--total-bytes",
        type=int,
        help="scale override; domain proportions remain those pinned in the spec",
    )
    args = parser.parse_args()
    manifest = build(args.spec.resolve(), args.output.resolve(), args.manifest.resolve(), args.total_bytes)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
