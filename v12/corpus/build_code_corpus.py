#!/usr/bin/env python3
"""Builds the C8 code-domain corpus from this repo's own real source files.

No code corpus existed anywhere on this machine when this funnel started
(checked cell80 + tiny-model's datasets/ dirs — those hold tree-sitter
*grammar metadata*, not source text). Rather than pull an external dataset,
this harvests real, already-reviewed Rust/Python source already in this
repo: the v11/v12 tokenizer crates, the common bench harness, and the
small multi-language priority-target samples under v11/corpus/code.

Deliberately small (a few hundred KB) relative to the prose/math domains —
a v0 seed, not a comprehensive code corpus. Every file's content and sha256
are recorded so the corpus is exactly reproducible and auditable.

NOTE: originally harvested from tiny-model's broader source tree
(model/v11-train, knowledge-extractor/, benchmarks/) when this lived at
tiny-model/tokenizer/v12/. Now that v-tokenizers is a standalone repo,
SOURCE_DIRS is scoped to what actually exists here -- rerunning this
script now produces a smaller, v-tokenizers-only corpus than the one
this repo was seeded with (the copied c8_code_corpus.jsonl on disk still
reflects the old tiny-model-wide harvest until this is rerun).

Run: python3 v12/corpus/build_code_corpus.py
"""
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # v-tokenizers/

EXTENSIONS = {".py": "python", ".rs": "rust", ".c": "c", ".go": "go",
              ".java": "java", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript"}

SOURCE_DIRS = [
    "v11/core/src",
    "v11/builder/src",
    "v11/cli/src",
    "v12/msi/src",
    "bench",
    "v11/corpus/code",  # small multi-language priority-target samples
]

OUT_JSONL = Path(__file__).resolve().parent / "c8_code_corpus.jsonl"
OUT_MANIFEST = Path(__file__).resolve().parent / "c8_code_corpus_manifest.json"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    rows = []
    for rel_dir in SOURCE_DIRS:
        d = REPO_ROOT / rel_dir
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix not in EXTENSIONS:
                continue
            data = p.read_bytes()
            rel_path = str(p.relative_to(REPO_ROOT))
            rows.append({
                "path": rel_path,
                "language": EXTENSIONS[p.suffix],
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "text": data.decode("utf-8", errors="replace"),
            })

    with OUT_JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    per_language = {}
    for r in rows:
        agg = per_language.setdefault(r["language"], {"files": 0, "bytes": 0})
        agg["files"] += 1
        agg["bytes"] += r["bytes"]

    combined_sha = sha256_bytes("".join(r["sha256"] for r in rows).encode())
    manifest = {
        "domain": "code",
        "role": "C8 code-domain slice (v0 seed, this repo's own source)",
        "total_files": len(rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "per_language": per_language,
        "source_dirs": SOURCE_DIRS,
        "combined_sha256": combined_sha,
        "output": str(OUT_JSONL.relative_to(REPO_ROOT)),
        "file_list": [{"path": r["path"], "language": r["language"], "bytes": r["bytes"], "sha256": r["sha256"]} for r in rows],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(json.dumps({
        "total_files": manifest["total_files"],
        "total_bytes": manifest["total_bytes"],
        "per_language": per_language,
        "combined_sha256": combined_sha,
    }, indent=2))


if __name__ == "__main__":
    main()
