#!/usr/bin/env python3
"""Freeze the byte streams and 48 production cells for decisive TOK-2.

This command is intentionally read-only with respect to model training.  It
hash-checks all immutable inputs, derives complete-document phase boundaries,
and writes one compact manifest from which every production run is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(base: Path, value: str) -> Path:
    configured = Path(value)
    return configured if configured.is_absolute() else (base / configured).resolve()


@dataclass(frozen=True)
class DocumentIndex:
    domain: str
    source: str
    text_sha256: str
    raw_bytes: int
    line_number: int


def load_index(path: Path) -> list[DocumentIndex]:
    documents: list[DocumentIndex] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            text = row["text"]
            raw = text.encode("utf-8")
            text_sha = sha256_bytes(raw)
            if text_sha in seen:
                raise AssertionError(f"duplicate C8 text at line {line_number}: {text_sha}")
            seen.add(text_sha)
            documents.append(
                DocumentIndex(
                    domain=row["domain"],
                    source=row["source"],
                    text_sha256=text_sha,
                    raw_bytes=len(raw),
                    line_number=line_number,
                )
            )
    return documents


def order_key(seed: int, document: DocumentIndex) -> str:
    return sha256_bytes(
        f"{seed}\0{document.domain}\0{document.text_sha256}".encode("utf-8")
    )


def stream_evidence(
    documents: list[DocumentIndex],
    *,
    seed: int,
    nominal_budget: int | None,
    checkpoint_fractions: list[float] | None = None,
) -> dict[str, Any]:
    ordered = sorted(documents, key=lambda item: order_key(seed, item))
    if nominal_budget is not None:
        selected: list[DocumentIndex] = []
        used = 0
        for document in ordered:
            if used + document.raw_bytes > nominal_budget:
                break
            selected.append(document)
            used += document.raw_bytes
    else:
        selected = ordered

    material = "".join(
        f"{document.line_number}\0{document.domain}\0{document.source}\0"
        f"{document.raw_bytes}\0{document.text_sha256}\n"
        for document in selected
    ).encode("utf-8")
    domain_bytes = Counter()
    domain_documents = Counter()
    for document in selected:
        domain_bytes[document.domain] += document.raw_bytes
        domain_documents[document.domain] += 1
    actual_bytes = sum(document.raw_bytes for document in selected)
    evidence = {
        "order_seed": seed,
        "nominal_raw_byte_budget": nominal_budget,
        "actual_raw_bytes": actual_bytes,
        "unused_budget_bytes": (
            None if nominal_budget is None else nominal_budget - actual_bytes
        ),
        "documents": len(selected),
        "first_line_number": selected[0].line_number if selected else None,
        "last_line_number": selected[-1].line_number if selected else None,
        "ordered_stream_sha256": sha256_bytes(material),
        "domain_raw_bytes": dict(sorted(domain_bytes.items())),
        "domain_documents": dict(sorted(domain_documents.items())),
    }
    if checkpoint_fractions:
        boundaries: list[dict[str, Any]] = []
        cumulative = 0
        next_fraction = 0
        for document_index, document in enumerate(selected, 1):
            cumulative += document.raw_bytes
            while (
                next_fraction < len(checkpoint_fractions)
                and cumulative
                >= actual_bytes * float(checkpoint_fractions[next_fraction])
            ):
                fraction = float(checkpoint_fractions[next_fraction])
                boundaries.append(
                    {
                        "fraction": fraction,
                        "documents": document_index,
                        "raw_bytes": cumulative,
                        "terminal_text_sha256": document.text_sha256,
                    }
                )
                next_fraction += 1
        if len(boundaries) != len(checkpoint_fractions):
            raise AssertionError("failed to derive every checkpoint boundary")
        evidence["checkpoint_boundaries"] = boundaries
    return evidence


def verify_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def archive_by_arm(archive: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["arm_id"]: row for row in archive["artifacts"]}


def build_manifest(spec_path: Path) -> dict[str, Any]:
    spec = load_json(spec_path)
    base = spec_path.parent
    inputs = spec["inputs"]

    input_paths = {
        key: resolve(base, inputs[key])
        for key in (
            "c8_jsonl",
            "c3_jsonl",
            "tokenizer_arms",
            "tokenizer_archive",
            "architecture_controls",
            "paired_seeds",
            "tiny_model_source",
        )
    }
    verify_hash(input_paths["c8_jsonl"], inputs["c8_sha256"], "C8")
    verify_hash(input_paths["c3_jsonl"], inputs["c3_sha256"], "C3")
    verify_hash(
        input_paths["tiny_model_source"],
        inputs["tiny_model_source_sha256"],
        "TinyModel source",
    )

    arms_doc = load_json(input_paths["tokenizer_arms"])
    archive_doc = load_json(input_paths["tokenizer_archive"])
    architecture = load_json(input_paths["architecture_controls"])
    seeds = load_json(input_paths["paired_seeds"])
    archived = archive_by_arm(archive_doc)

    artifacts: dict[str, Any] = {}
    for arm in arms_doc["arms"]:
        artifact_path = resolve(input_paths["tokenizer_arms"].parent, arm["artifact"])
        verify_hash(artifact_path, arm["artifact_sha256"], f"{arm['id']} tokenizer")
        evidence: dict[str, Any] = {
            "path": str(artifact_path.relative_to(HERE.parent.parent)),
            "sha256": arm["artifact_sha256"],
            "vocab_rows": arm["vocab_rows"],
            "family": arm["family"],
            "kind": arm["kind"],
        }
        if arm["id"] in archived:
            row = archived[arm["id"]]
            if row["sha256"] != arm["artifact_sha256"]:
                raise AssertionError(f"archive hash mismatch for {arm['id']}")
            evidence["archive"] = {
                "artifact_id": row["artifact_id"],
                "uri": row["uri"],
                "producing_run_id": archive_doc["registry"]["producing_run_id"],
            }
        artifacts[arm["id"]] = evidence

    c8_documents = load_index(input_paths["c8_jsonl"])
    c3_documents = load_index(input_paths["c3_jsonl"])
    data_order = seeds["data_order"]
    schedule = spec["stream_schedule"]
    streams = {
        "phase1": stream_evidence(
            c8_documents,
            seed=int(data_order[schedule["phase1"]["document_order_seed_field"]]),
            nominal_budget=int(schedule["phase1"]["nominal_raw_byte_budget"]),
            checkpoint_fractions=spec["checkpoints"]["phase_byte_fractions"],
        ),
        "phase3": stream_evidence(
            c8_documents,
            seed=int(data_order[schedule["phase3"]["document_order_seed_field"]]),
            nominal_budget=int(schedule["phase3"]["nominal_raw_byte_budget"]),
            checkpoint_fractions=spec["checkpoints"]["phase_byte_fractions"],
        ),
        "evaluation": stream_evidence(
            c3_documents,
            seed=int(data_order[schedule["evaluation"]["document_order_seed_field"]]),
            nominal_budget=None,
        ),
    }

    runs: list[dict[str, Any]] = []
    controls = ("fixed_trunk", "fixed_total")
    for control in controls:
        for arm in arms_doc["arms"]:
            family = arm["family"]
            parameters = architecture[control][family]
            for paired in seeds["paired_runs"]:
                label = int(paired["label"])
                run_id = f"tok2-{control}-{arm['id']}-seed{label}"
                runs.append(
                    {
                        "run_id": run_id,
                        "control": control,
                        "arm_id": arm["id"],
                        "family": family,
                        "paired_run_label": label,
                        "vocab_rows": arm["vocab_rows"],
                        "ffn_width": parameters["ffn_width"],
                        "parameters": {
                            key: parameters[key]
                            for key in (
                                "nominal_parameters",
                                "phase1_trainable_parameters",
                                "phase3_trainable_parameters",
                            )
                        },
                        "seeds": {
                            key: value
                            for key, value in paired.items()
                            if key != "label"
                        },
                        "status": "registered-not-started",
                    }
                )

    if len(runs) != int(spec["execution"]["matrix_cells"]):
        raise AssertionError(f"expected 48 cells, built {len(runs)}")
    if len({row["run_id"] for row in runs}) != len(runs):
        raise AssertionError("production run IDs are not unique")

    source_hashes = {
        key: sha256_file(path)
        for key, path in input_paths.items()
        if key not in {"c8_jsonl", "c3_jsonl"}
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen-validated-not-started",
        "production_spec": spec_path.name,
        "production_spec_sha256": sha256_file(spec_path),
        "inputs": {
            "c8_sha256": inputs["c8_sha256"],
            "c8_rows": len(c8_documents),
            "c3_sha256": inputs["c3_sha256"],
            "c3_rows": len(c3_documents),
            "tiny_model_repository_commit": inputs["tiny_model_repository_commit"],
            "source_manifest_sha256": source_hashes,
        },
        "tokenizers": artifacts,
        "streams": streams,
        "runs": runs,
        "launch_policy": spec["execution"]["launch_policy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=HERE / "tok2_production_spec.json"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "tok2_production_manifest.json"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the existing manifest instead of writing it",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.spec.resolve())
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not args.output.exists():
            raise SystemExit(f"missing frozen manifest: {args.output}")
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("frozen production manifest differs from regenerated value")
        print("TOK-2 production manifest is reproducible and valid (48 cells)")
        return
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output} ({len(manifest['runs'])} cells)")


if __name__ == "__main__":
    main()
