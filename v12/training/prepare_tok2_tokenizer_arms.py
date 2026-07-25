#!/usr/bin/env python3
"""Materialize and verify the eight frozen TOK-2 tokenizer arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

try:
    from .rehearse_tok2_matrix import sha256_file
    from .train_structural_candidate import train
except ImportError:
    from rehearse_tok2_matrix import sha256_file
    from train_structural_candidate import train


HERE = Path(__file__).resolve().parent


def resolve(base: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else (base / path).resolve()


def runtime_rows(arm: dict[str, Any], artifact: Path) -> int:
    if arm["kind"] in {"v11_ws_exact", "hf_tokenizer"}:
        return Tokenizer.from_file(str(artifact)).get_vocab_size()
    raw = json.loads(artifact.read_text())
    return len(raw["pieces"])


def prepare(
    manifest_path: Path,
    corpus_path: Path,
    *,
    audit_retrain_root: Path | None = None,
    audit_families: set[str] | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(corpus_path) != manifest["corpus_sha256"]:
        raise ValueError("C8 corpus hash does not match the frozen arm manifest")
    evidence = []
    for arm in manifest["arms"]:
        artifact = resolve(manifest_path.parent, arm["artifact"])
        if not artifact.exists():
            raise FileNotFoundError(
                f"missing immutable {arm['kind']} artifact for {arm['id']}: {artifact}"
            )
        digest = sha256_file(artifact)
        if digest != arm["artifact_sha256"]:
            raise ValueError(
                f"{arm['id']} artifact hash mismatch: "
                f"expected {arm['artifact_sha256']}, got {digest}"
            )
        rows = runtime_rows(arm, artifact)
        if rows != arm["vocab_rows"]:
            raise ValueError(
                f"{arm['id']} runtime row mismatch: expected {arm['vocab_rows']}, got {rows}"
            )
        evidence.append(
            {
                "arm": arm["id"],
                "artifact": arm["artifact"],
                "sha256": digest,
                "runtime_vocab_rows": rows,
            }
        )
    retraining_audit = []
    if audit_retrain_root is not None:
        audit_retrain_root.mkdir(parents=True, exist_ok=True)
        for arm in manifest["arms"]:
            if arm["kind"] != "hf_tokenizer":
                continue
            if audit_families and arm["family"] not in audit_families:
                continue
            algorithm = "unigram" if arm["family"] == "U16" else "bpe"
            audit_dir = audit_retrain_root / f"{arm['id']}_audit"
            train(
                algorithm,
                16000,
                arm["pre_tokenization"],
                corpus_path,
                audit_dir,
                arm["id"],
                show_progress=False,
                require_exact_vocab=True,
            )
            audit_artifact = audit_dir / "tokenizer.json"
            audit_sha = sha256_file(audit_artifact)
            retraining_audit.append(
                {
                    "arm": arm["id"],
                    "family": arm["family"],
                    "frozen_sha256": arm["artifact_sha256"],
                    "retrained_sha256": audit_sha,
                    "byte_identical_to_frozen": audit_sha
                    == arm["artifact_sha256"],
                    "runtime_vocab_rows": runtime_rows(arm, audit_artifact),
                }
            )
    return {
        "status": "pass",
        "manifest_sha256": sha256_file(manifest_path),
        "corpus_sha256": sha256_file(corpus_path),
        "arms_verified": len(evidence),
        "arms": evidence,
        "retraining_audit": retraining_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=HERE / "tok2_tokenizer_arms.json"
    )
    parser.add_argument(
        "--corpus", type=Path, default=HERE.parent / "corpus" / "c8_corpus_v3.jsonl"
    )
    parser.add_argument(
        "--audit-retrain-root",
        type=Path,
        help="safely retrain selected structural arms outside the frozen artifact directories",
    )
    parser.add_argument(
        "--audit-families",
        nargs="+",
        choices=["U16", "B16"],
        help="limit --audit-retrain-root to selected families",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.manifest.resolve(),
                args.corpus.resolve(),
                audit_retrain_root=(
                    args.audit_retrain_root.resolve()
                    if args.audit_retrain_root
                    else None
                ),
                audit_families=set(args.audit_families or []),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
