#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.24", "tokenizers>=0.20", "transformers>=4.44", "pyyaml>=6"]
# ///
"""Publish one tokenizer to the Hub as an immutable, AutoTokenizer-loadable
model repo -- with identity enforced by the machine, not by a README convention.

  publish_tokenizer.py --tokenizer-json v11/artifacts/tokenizer.json \\
                       --repo-id chrishayuk/v11-tokenizer \\
                       --status adopted --dry-run

Acceptance criterion: a clean environment can do
`AutoTokenizer.from_pretrained(repo_id)` with no GitHub clone, no training code
and no trust_remote_code. This script proves that by loading the *downloaded*
artifact, not the local one, before it reports success.

FOUR THINGS THIS DOES DIFFERENTLY from the obvious version, each because the
obvious version is wrong for this project:

1. IDENTITY IS THE CONTENT HASH, NOT THE HUB REVISION.
   A Hub commit oid is a property of the upload: re-pushing identical bytes
   mints a new oid, and a README typo produces a new oid over an unchanged
   tokenizer. So the anchor is sha256(tokenizer.json) -- the same value that
   joins against a checkpoint's `tokenizer_hash` and against `tokenizer_sha`
   in the chuk-datasets catalog. The Hub revision is recorded too, as a
   retrieval coordinate, but it is never the identity and never the gate.

2. IMMUTABILITY IS ENFORCED, NOT DOCUMENTED.
   If the repo already exists, its tokenizer.json is fetched and hashed first.
   A mismatch REFUSES the push and tells you to publish under a new name. There
   is no --force: silently replacing a published tokenizer is how every
   downstream `tokenizer_hash` join quietly starts lying.

3. VERIFICATION IS A GOLDEN-VECTOR ROUND-TRIP, NOT A LENGTH CHECK.
   `len(loaded) == len(local)` plus special_tokens_map equality passes happily
   through a changed normalizer, pre-tokenizer, post-processor, or added-token
   ordering. Instead every frame in bench/msi/frame_battery.jsonl is encoded by
   the downloaded tokenizer and compared id-for-id against the local one. Those
   frames are the MSI-sensitive surfaces (call-operand, post-delimiter,
   mid-string punctuation), so a tokenizer that loads cleanly but re-segments a
   call delimiter fails here rather than in a training run.

4. PROVENANCE IS A POINTER, NOT A COPY.
   chuk-datasets owns corpus identity and chuk-experiments owns the results
   ledger. Embedding corpus_manifest.json / evaluation_results.json in the Hub
   repo forks both into a second source of truth that nothing keeps in sync.
   provenance.json carries dataset name + version + content_sha and an
   experiment run id; the catalog resolves them.

Also deliberately absent: `model_max_length`. That is a property of a model, not
of a tokenizer, and baking 512 into the artifact makes a claim the tokenizer
cannot honour. Set it in the model repo.

WHEN TO PUBLISH AT ALL. An artifact goes to the Hub when it acquires *external*
identity -- referenced by a model repo, a paper, or a reviewer. EVO-TOK will
generate populations; publishing every candidate fills the namespace with island
members. Everything else stays content-addressed in the experiment server.
Anything published before TOK-4 adjudicates must carry
`--status candidate` (the default), which stamps "candidate, not adopted" into
the card so a phase-2 read cannot quietly acquire authority it has not earned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAME_BATTERY = REPO_ROOT / "bench" / "msi" / "frame_battery.jsonl"
DORMANT_MAP = REPO_ROOT / "v12" / "dormant_block_map.yaml"

STATUSES = {
    "candidate": "**candidate — NOT adopted.** Published for external reference "
                 "only. TOK-4 has not adjudicated; do not treat any number here "
                 "as a promotion decision.",
    "adopted": "**adopted.** This is the current recommended tokenizer for this line.",
    "historical": "**historical baseline.** Superseded; kept for replication of "
                  "earlier results.",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frames() -> list[dict]:
    """The MSI frame battery: each probe wraps an operand in the surfaces where
    segmentation actually goes wrong -- call operands, post-delimiter, and
    mid-string punctuation -- rather than in 'representative strings'."""
    if not FRAME_BATTERY.is_file():
        sys.exit(f"missing frame battery: {FRAME_BATTERY}")
    return [json.loads(line) for line in FRAME_BATTERY.read_text().splitlines() if line.strip()]


def golden_encodings(tok, frames: list[dict]) -> dict:
    """{probe_id/frame_name: [ids]} for every frame. This is the artifact the
    published repo carries AND the vector set the post-upload check replays."""
    out = {}
    for probe in frames:
        for frame_name, text in probe["frames"].items():
            out[f"{probe['probe_id']}/{frame_name}"] = {
                "text": text,
                "ids": tok.encode(text).ids,
            }
    return out


def dormant_blocks_doc() -> dict | None:
    """Machine-readable dormant-block declaration (C7).

    A tokenizer published through save_pretrained exposes every id to anyone who
    loads it, including ids the model physically masks out of the projection
    matmul and optimizer state at 115M. A consumer who sizes an embedding table
    from len(tokenizer) gets a table that disagrees with the model's active
    slice -- and no vocab-length check catches it, because both numbers are
    'correct'. So the masking policy ships as data, not as a prose limitation."""
    if not DORMANT_MAP.is_file():
        return None
    import yaml
    m = yaml.safe_load(DORMANT_MAP.read_text())
    blocks = []
    for b in m.get("layout_order", []):
        tokens = b.get("tokens")
        blocks.append({
            "block": b.get("block"),
            "category": b.get("category"),
            "active_at_115M": b.get("active_at_115M"),
            "active_at_1B": b.get("active_at_1B"),
            "merge_opaque": b.get("merge_opaque"),
            "token_count": len(tokens) if isinstance(tokens, list) else None,
            "tokens_frozen": bool(tokens) if tokens is not None else None,
        })
    return {
        "schema": "v-tokenizers-dormant-blocks-1",
        "source": "v12/dormant_block_map.yaml (C7)",
        "masking_policy": m.get("masking_policy", {}),
        "layout_order": blocks,
        "note": (
            "Dormant ids are present in the tokenizer's vocabulary but are excluded "
            "from the projection matmul and optimizer state at 115M. Size an active "
            "embedding slice from the active blocks, NOT from len(tokenizer). Where "
            "token_count is 0 the block is schema-pinned but not yet frozen (filled "
            "at TOK-5 once the winning vocab layout is known)."
        ),
    }


def build_release(tokenizer_json: Path, out_dir: Path, args, frames) -> tuple[dict, str]:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer.from_file(str(tokenizer_json))
    local_sha = sha256_file(tokenizer_json)

    # No model_max_length: see the module docstring.
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=args.unk_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        pad_token=args.pad_token,
        clean_up_tokenization_spaces=False,
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    fast.save_pretrained(str(out_dir))

    # save_pretrained re-serializes tokenizer.json through transformers. The
    # identity anchor must stay the file we were given, so overwrite it back and
    # re-check -- otherwise the published sha is transformers' round-trip, not
    # the artifact anyone else hashed.
    shutil.copyfile(tokenizer_json, out_dir / "tokenizer.json")
    if sha256_file(out_dir / "tokenizer.json") != local_sha:
        sys.exit("internal error: tokenizer.json changed while staging")

    normalize_config(out_dir, fast)

    # crates.io ships code, not data. A viewer who runs `cargo install v11-cli`
    # has a working `v11` binary and no vocabulary for it to open, which makes
    # every CLI example in the docs unrunnable. Shipping the vocab here is what
    # closes that -- `v11 --model <hub download> vocab --blocks` then works from
    # a clean machine with no clone.
    if args.vocab_bin:
        if not args.vocab_bin.is_file():
            sys.exit(f"missing vocab: {args.vocab_bin}")
        shutil.copyfile(args.vocab_bin, out_dir / args.vocab_bin.name)

    golden = golden_encodings(backend, frames)
    (out_dir / "golden_encodings.json").write_text(json.dumps(golden, indent=2))

    dormant = dormant_blocks_doc()
    if dormant:
        (out_dir / "dormant_blocks.json").write_text(json.dumps(dormant, indent=2))

    provenance = {
        "schema": "v-tokenizers-provenance-1",
        "tokenizer_sha256": local_sha,
        "vocab_bin_sha256": sha256_file(args.vocab_bin) if args.vocab_bin else None,
        "vocab_size": backend.get_vocab_size(),
        "status": args.status,
        "source_repo": "https://github.com/chrishayuk/v-tokenizers",
        "source_commit": args.source_commit or None,
        # Pointers, not copies -- the catalog and the experiment server own these.
        "corpus": {
            "catalog": "chuk-datasets",
            "dataset": args.dataset_name or None,
            "version_id": args.dataset_version or None,
            "content_sha": args.dataset_sha or None,
            "resolve": "https://chuk-datasets.fly.dev/v1/datasets",
        },
        "evaluation": {
            "ledger": "chuk-experiments",
            "run_id": args.experiment_run or None,
            "note": "Headline numbers belong in the card; the ledger is authoritative.",
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    lines = [f"{sha256_file(p)}  {p.name}"
             for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "checksums.sha256"]
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")

    (out_dir / "README.md").write_text(render_card(args, backend, local_sha, dormant, len(golden)))
    return provenance, local_sha


def normalize_config(out_dir: Path, fast) -> None:
    """Make the staged config loadable by BOTH transformers 4.x and 5.x.

    Measured, not assumed: transformers 5.14 writes
    `tokenizer_class: "TokenizersBackend"` and a `backend` key, and 4.46 then
    refuses the repo outright -- "Tokenizer class TokenizersBackend does not
    exist or is not currently imported". That silently breaks the acceptance
    criterion for every consumer on 4.x, and a golden-vector replay cannot
    catch it because the replay runs on whatever version staged the files.

    `PreTrainedTokenizerFast` resolves on both. Also drops the
    `model_max_length` sentinel transformers writes when it is unset
    (10^30-ish): a length belongs to a model, not a tokenizer, and an absent
    key reads as unset far better than a nonsense number does."""
    cfg_path = out_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.pop("model_max_length", None)
    cfg.pop("backend", None)
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    # 5.x folds specials into tokenizer_config; 4.x tooling and the Hub UI both
    # still expect this file, and the existing published repos carry it.
    specials = {k: v for k, v in (
        ("unk_token", cfg.get("unk_token")), ("bos_token", cfg.get("bos_token")),
        ("eos_token", cfg.get("eos_token")), ("pad_token", cfg.get("pad_token")),
    ) if v}
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(specials, indent=2, sort_keys=True) + "\n")


def render_card(args, backend, sha, dormant, n_golden) -> str:
    cli_section = ""
    if args.vocab_bin:
        name = args.vocab_bin.name
        cli_section = f"""
## Using the Rust CLI

`{name}` is the native vocabulary, shipped here because `cargo install` delivers
the binary and not the data:

```bash
cargo install v11-cli
huggingface-cli download {args.repo_id} {name} --local-dir .
v11 --model {name} vocab --blocks
v11 --model {name} encode --text "Once upon a time" --show-pieces
```
"""

    vocab = backend.get_vocab_size()
    specials = [(role, tok) for role, tok in (
        ("Unknown", args.unk_token), ("Beginning", args.bos_token),
        ("End", args.eos_token), ("Padding", args.pad_token)) if tok]
    rows = "\n".join(
        f"| {role} | `{tok}` | {backend.token_to_id(tok)} |" for role, tok in specials)

    dormant_section = ""
    if dormant:
        frozen = [b for b in dormant["layout_order"] if b.get("active_at_115M") is False]
        if frozen:
            dormant_section = f"""
## Dormant blocks — read before sizing an embedding table

This vocabulary declares blocks that are **present as ids but masked out of the
projection matmul and optimizer state** at 115M parameters. Sizing an embedding
table from `len(tokenizer)` will disagree with the model's active slice, and no
vocabulary-length check will catch it.

See `dormant_blocks.json` for the machine-readable declaration.

| Block | Active @115M | Active @1B | Tokens frozen |
|---|---|---|---|
""" + "\n".join(
                f"| {b['block']} | {b['active_at_115M']} | {b['active_at_1B']} | "
                f"{'yes' if b['tokens_frozen'] else 'no (schema pinned, filled at TOK-5)'} |"
                for b in frozen) + "\n"

    return f"""---
language:
  - en
library_name: transformers
tags:
  - tokenizer
  - language-modeling
license: apache-2.0
---

# {args.repo_id.split('/')[-1]}

{STATUSES[args.status]}

## Loading

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{args.repo_id}")
```

No `trust_remote_code`, no clone, no training code. That is verified against the
downloaded artifact at publish time, not asserted.

## Identity

**Identity is the content hash, not the Hub revision.** Re-pushing identical
bytes mints a new commit oid; a README edit does too. Join on this instead — it
is the same value a checkpoint records as `tokenizer_hash`:

| | |
|---|---|
| `tokenizer.json` sha256 | `{sha}` |
| Vocabulary size | {vocab:,} |
| Source | [chrishayuk/v-tokenizers](https://github.com/chrishayuk/v-tokenizers) |
| Source commit | `{args.source_commit or 'unrecorded'}` |

This repo is immutable: a fix ships under a new name, never as a replacement.

## Special tokens

| Role | Token | ID |
|---|---|---|
{rows}
{cli_section}{dormant_section}
## Golden encodings

`golden_encodings.json` carries {n_golden} frames from the MSI frame battery —
call operands, post-delimiter and mid-string-punctuation surfaces, not
"representative strings", because those are where segmentation actually breaks.
Every one is replayed against the downloaded artifact at publish time.

```python
import json, urllib.request
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("{args.repo_id}")
golden = json.load(urllib.request.urlopen(
    "https://huggingface.co/{args.repo_id}/resolve/main/golden_encodings.json"))
for name, case in golden.items():
    assert tok(case["text"], add_special_tokens=False)["input_ids"] == case["ids"], name
```

## Provenance

`provenance.json` points at the systems that own these facts rather than copying
them: the corpus lives in the **chuk-datasets** catalog (dataset name, version,
content sha), and results live in the **chuk-experiments** ledger (run id). This
repo is a publication location, not a second source of truth.

## Intended use

Research and training of compact language models.

## Limitations

Corpus coverage, language coverage and normalisation behaviour are described in
the source repository. {"Dormant-block ids are exposed by this artifact but inactive at 115M — see above." if dormant_section else ""}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer-json", type=Path, required=True)
    ap.add_argument("--vocab-bin", type=Path, default=None,
                    help="v11.vocab.bin — the native Rust vocabulary. Ship it: "
                         "`cargo install v11-cli` delivers the binary but not the "
                         "data, so without this the published CLI has nothing to load.")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--status", choices=sorted(STATUSES), default="candidate")
    ap.add_argument("--unk-token", default="<unk>")
    ap.add_argument("--bos-token", default="<s>")
    ap.add_argument("--eos-token", default="</s>")
    ap.add_argument("--pad-token", default="<pad>")
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dataset-name", default="")
    ap.add_argument("--dataset-version", default="")
    ap.add_argument("--dataset-sha", default="")
    ap.add_argument("--experiment-run", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage and verify locally; contact the Hub read-only, push nothing")
    args = ap.parse_args()

    if not args.tokenizer_json.is_file():
        sys.exit(f"missing tokenizer: {args.tokenizer_json}")

    frames = load_frames()
    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="tok-release-"))
    provenance, local_sha = build_release(args.tokenizer_json, out_dir, args, frames)

    print(f"staged {args.repo_id}")
    print(f"  vocab            {provenance['vocab_size']:,}")
    print(f"  tokenizer sha256 {local_sha}")
    print(f"  status           {args.status}")
    print(f"  golden frames    {len(frames)} probes")
    print(f"  files            {sorted(p.name for p in out_dir.iterdir())}")

    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    api = HfApi()

    # --- immutability precondition, before anything is uploaded --------------
    try:
        existing = api.hf_hub_download(repo_id=args.repo_id, filename="tokenizer.json",
                                       repo_type="model")
        published_sha = sha256_file(Path(existing))
        if published_sha != local_sha:
            sys.exit(
                f"\nREFUSING TO PUSH -- {args.repo_id} already holds a different tokenizer.\n"
                f"  published {published_sha}\n  local     {local_sha}\n\n"
                f"Published tokenizers are immutable: every downstream tokenizer_hash\n"
                f"join is only as trustworthy as that promise. Publish the fix under a\n"
                f"new repo id (…-v2, …-16k-v2) instead of replacing this one.\n")
        print(f"  precondition     repo exists, identical bytes — re-push is a no-op")
    except (RepositoryNotFoundError, EntryNotFoundError):
        print(f"  precondition     repo is new")

    if args.dry_run:
        print(f"\ndry run — nothing pushed. Staged at {out_dir}")
        return

    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo_id, repo_type="model",
                               folder_path=str(out_dir),
                               commit_message=f"Publish tokenizer {local_sha[:16]} ({args.status})")
    revision = getattr(commit, "oid", None) or "main"
    print(f"  pushed           revision {revision}")

    # --- verify the DOWNLOADED artifact, not the local one -------------------
    verify_published(args.repo_id, revision, local_sha, out_dir)

    print(f"\npublished {args.repo_id}")
    print(f"  identity (join on this): {local_sha}")
    print(f"  hub revision (retrieval coordinate only): {revision}")


def verify_published(repo_id: str, revision: str, local_sha: str, staged: Path) -> None:
    """Load what the Hub actually serves and replay the golden vectors through
    it. A length check would pass through a changed normalizer or
    post-processor; an id-for-id replay will not."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    remote_json = hf_hub_download(repo_id=repo_id, filename="tokenizer.json",
                                  repo_type="model", revision=revision)
    remote_sha = sha256_file(Path(remote_json))
    if remote_sha != local_sha:
        sys.exit(f"VERIFY FAILED: hub serves {remote_sha}, expected {local_sha}")

    tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True, revision=revision)
    golden = json.loads((staged / "golden_encodings.json").read_text())
    bad = []
    for name, case in golden.items():
        got = tok(case["text"], add_special_tokens=False)["input_ids"]
        if got != case["ids"]:
            bad.append((name, case["ids"], got))
    if bad:
        print(f"\nVERIFY FAILED: {len(bad)} golden vector(s) differ through AutoTokenizer:")
        for name, want, got in bad[:5]:
            print(f"  {name}\n    expected {want}\n    got      {got}")
        sys.exit(1)
    print(f"  verified         sha match + {len(golden)} golden vectors replay identically")


if __name__ == "__main__":
    main()
