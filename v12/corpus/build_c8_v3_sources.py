#!/usr/bin/env python3
"""Materialize the non-code C8 v3 source pools from pinned raw artifacts.

The output pools are generated artifacts.  The tracked source specification
pins every raw file, the extraction rule, and (once frozen) every output hash.
Rows are selected by a stable hash, not source order, and no row is repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(spec_path: Path, source: dict[str, Any]) -> Path:
    candidates = [source["path"], *source.get("local_candidates", [])]
    resolved = [
        Path(candidate)
        if Path(candidate).is_absolute()
        else (spec_path.parent / candidate).resolve()
        for candidate in candidates
    ]
    for candidate in resolved:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"none of the pinned source candidates exists for {source['id']}: "
        + ", ".join(str(path) for path in resolved)
    )


def verify_source(spec_path: Path, source: dict[str, Any]) -> Path:
    path = resolve_source(spec_path, source)
    actual = sha256_file(path)
    if actual != source["sha256"]:
        raise ValueError(
            f"raw source digest mismatch for {source['id']}: "
            f"expected {source['sha256']}, got {actual}"
        )
    return path


def arrow_rows(path: Path, source_id: str) -> Iterator[dict[str, Any]]:
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError("build_c8_v3_sources.py requires the 'datasets' package") from error

    dataset = Dataset.from_file(str(path))
    for index, row in enumerate(dataset):
        yield {"row_id": f"{source_id}:{index}", **row}


def jsonl_rows(path: Path, source_id: str) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield {"row_id": f"{source_id}:{line_number}", **json.loads(line)}


def _stable_key(seed: int, domain: str, source: str, text: str) -> tuple[str, str]:
    text_sha = sha256_bytes(text.encode("utf-8"))
    material = f"{seed}\0{domain}\0{source}\0{text_sha}".encode()
    return sha256_bytes(material), text_sha


def select_without_replacement(
    rows: Iterable[dict[str, str]],
    *,
    seed: int,
    domain: str,
    byte_target: int,
) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        text = row["text"]
        if not text:
            continue
        text_sha = sha256_bytes(text.encode("utf-8"))
        unique.setdefault(text_sha, row)

    ordered = sorted(
        unique.values(),
        key=lambda row: _stable_key(seed, domain, row["source"], row["text"]),
    )
    selected: list[dict[str, str]] = []
    selected_bytes = 0
    for row in ordered:
        selected.append(row)
        selected_bytes += len(row["text"].encode("utf-8"))
        if selected_bytes >= byte_target:
            break
    if selected_bytes < byte_target:
        raise ValueError(
            f"{domain} source stratum has {selected_bytes:,} unique bytes, "
            f"below its {byte_target:,}-byte pool target"
        )
    return selected


def prose_rows(path: Path, source_id: str) -> Iterator[dict[str, str]]:
    for row in arrow_rows(path, source_id):
        text = row.get("text")
        if isinstance(text, str) and text:
            yield {"source": row["row_id"], "text": text}


def math_rows(path: Path, source_id: str) -> Iterator[dict[str, str]]:
    for row in jsonl_rows(path, source_id):
        text = row.get("text")
        if isinstance(text, str) and text:
            yield {"source": row["row_id"], "text": text}


def structured_rows(path: Path, source_id: str) -> Iterator[dict[str, str]]:
    for row in jsonl_rows(path, source_id):
        original = row.get("text")
        if not isinstance(original, str) or not original:
            continue
        before_call, marker, after_call = original.partition("<call>")
        call_body = None
        if marker:
            call_body = after_call.partition("</call>")[0].strip()
        record = {
            "kind": "cell80_record",
            "dataset": source_id,
            "species": row.get("species"),
            "instruction": before_call.strip() if marker else None,
            "call": call_body,
            "trace": None if marker else original,
            "meta": row.get("meta"),
        }
        text = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        yield {"source": row["row_id"], "text": text}


def noisy_unicode_rows(path: Path, source_id: str) -> Iterator[dict[str, str]]:
    for row in arrow_rows(path, source_id):
        text = row.get("text")
        if not isinstance(text, str) or not text:
            continue
        # Identity-only extraction.  The lane captures naturally occurring
        # non-ASCII, multiline web text rather than synthetic corruption.
        if any(ord(character) > 127 for character in text) and (
            "\n" in text or "\r" in text or "\t" in text
        ):
            yield {"source": row["row_id"], "text": text}


def write_pool(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "text_bytes": sum(len(row["text"].encode("utf-8")) for row in rows),
        "unique_text_rows": len(
            {sha256_bytes(row["text"].encode("utf-8")) for row in rows}
        ),
    }


def build(spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION}")
    seed = int(spec["seed"])
    sources = {source["id"]: source for source in spec["raw_sources"]}
    verified = {
        source_id: verify_source(spec_path, source)
        for source_id, source in sources.items()
    }

    outputs: dict[str, Any] = {}
    for domain, domain_spec in spec["domains"].items():
        candidates: list[dict[str, str]] = []
        for stratum in domain_spec["strata"]:
            source_id = stratum["source_id"]
            path = verified[source_id]
            extractor = stratum["extractor"]
            if extractor == "tinystories_text":
                rows = prose_rows(path, source_id)
            elif extractor == "cell80_math_text":
                rows = math_rows(path, source_id)
            elif extractor == "cell80_structured_json":
                rows = structured_rows(path, source_id)
            elif extractor == "fineweb_natural_unicode":
                rows = noisy_unicode_rows(path, source_id)
            else:
                raise ValueError(f"unknown extractor: {extractor}")
            candidates.extend(
                select_without_replacement(
                    rows,
                    seed=seed,
                    domain=f"{domain}:{source_id}",
                    byte_target=int(stratum["pool_bytes"]),
                )
            )

        selected = select_without_replacement(
            candidates,
            seed=seed,
            domain=domain,
            byte_target=int(domain_spec["pool_bytes"]),
        )
        output_path = (
            spec_path.parent / domain_spec["output_path"]
        ).resolve()
        output = write_pool(output_path, selected)
        expected = domain_spec.get("expected_output")
        if spec.get("status") == "frozen":
            if not expected:
                raise ValueError(f"frozen source spec lacks expected_output for {domain}")
            for field in ("sha256", "rows", "text_bytes", "unique_text_rows"):
                if output[field] != expected[field]:
                    raise ValueError(
                        f"{domain} output {field} mismatch: "
                        f"expected {expected[field]}, got {output[field]}"
                    )
        outputs[domain] = output
    return {"schema_version": SCHEMA_VERSION, "source_pool_id": spec["source_pool_id"], "outputs": outputs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).with_name("c8_v3_source_spec.json"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.spec.resolve()), indent=2))


if __name__ == "__main__":
    main()
