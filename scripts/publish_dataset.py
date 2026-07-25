#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.24"]
# ///
"""Publish one dataset to the Hub as a dataset repo, and prove what landed.

  publish_dataset.py --root v11/corpus --repo-id chrishayuk/v11-corpus \\
                     --card scripts/cards/v11-corpus.md --dry-run

  publish_dataset.py --root v11/corpus --repo-id chrishayuk/v11-corpus \\
                     --verify-only

This exists because `api.upload_folder(...)` followed by `print("pushed")` is
not evidence. It reports success on the strength of the request having been
accepted, which is a different claim from "the bytes a reader will download are
the bytes I meant to publish". Two of this project's three publishing surfaces
already enforce their own outcome -- the crates job asserts every version is
really on the sparse index, and publish_tokenizer.py replays golden vectors
against the *downloaded* tokenizer -- and the dataset push was the last one
that simply trusted the run. The pipeline has already produced two releases
where every job went green while something had silently not happened; this
closes the remaining place that could happen again.

WHAT IT ENFORCES, beyond uploading:

1. VERIFICATION READS BACK, IT DOES NOT ASSUME.
   Every local file is downloaded from the Hub at the resulting revision and
   compared by sha256 against the local copy. Not a file count, not a size
   check -- both of those pass through a truncated or re-encoded upload. The
   corpus is small enough (tens of KB) that hashing all of it is free, and the
   whole point of publishing it is that someone can re-run a gate against it.

2. MISSING AND EXTRA FILES ARE BOTH ERRORS.
   Missing means the upload was partial. Extra means the repo carries files
   this push did not put there -- a stale artifact from an earlier layout that
   a reader would reasonably assume is current. `--allow-extra` opts out for
   the single-file case, where the repo legitimately holds more than one thing.

3. --verify-only IS A FIRST-CLASS MODE.
   The pipeline has no dry-run against a real registry, so a published dataset
   can otherwise only be checked by hand. This mode uploads nothing and exits
   non-zero if the live repo has drifted from the working tree, which makes it
   usable as a standalone audit.

The card is a file rather than a heredoc in the workflow, so it can be
reviewed, diffed, and rendered without reading YAML. {n_files} is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_manifest(root: Path) -> dict[str, str]:
    """Map of repo-relative path -> sha256, for everything being published."""
    if root.is_file():
        return {root.name: sha256_file(root)}
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def verify(api: HfApi, repo_id: str, revision: str, manifest: dict[str, str],
           allow_extra: bool) -> None:
    """Download every published file and compare sha256 with the local copy."""
    remote = {
        f for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        # .gitattributes is created by the Hub itself, never by this push.
        if f != ".gitattributes"
    }

    missing = sorted(set(manifest) - remote)
    if missing:
        raise SystemExit(
            f"verification failed: {len(missing)} file(s) absent from {repo_id}\n  "
            + "\n  ".join(missing[:20])
        )

    # A card is written by this script, not part of the source tree; ignore it
    # when judging what is "extra".
    extra = sorted(remote - set(manifest) - {"README.md"})
    if extra and not allow_extra:
        raise SystemExit(
            f"verification failed: {repo_id} carries {len(extra)} file(s) this push "
            f"did not publish -- a reader cannot tell those are stale\n  "
            + "\n  ".join(extra[:20])
            + "\n(pass --allow-extra if the repo is meant to hold more than this)"
        )

    mismatched = []
    for rel, want in sorted(manifest.items()):
        got = sha256_file(Path(hf_hub_download(
            repo_id, rel, repo_type="dataset", revision=revision,
        )))
        if got != want:
            mismatched.append(f"{rel}\n      local  {want}\n      remote {got}")

    if mismatched:
        raise SystemExit(
            f"verification failed: {len(mismatched)} file(s) differ from the local copy\n  "
            + "\n  ".join(mismatched[:10])
        )

    print(f"  verified         {len(manifest)} files, sha256 match on every one")
    if extra:
        print(f"  note             {len(extra)} additional file(s) in the repo, allowed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True,
                    help="Directory or single file to publish")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--card", type=Path, default=None,
                    help="Markdown card; {n_files} is substituted")
    ap.add_argument("--commit-message", default="")
    ap.add_argument("--allow-extra", action="store_true",
                    help="Tolerate files in the repo this push did not publish")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be pushed; contact the Hub not at all")
    ap.add_argument("--verify-only", action="store_true",
                    help="Upload nothing; check the live repo matches the working tree")
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"--root does not exist: {args.root}")

    manifest = local_manifest(args.root)
    if not manifest:
        raise SystemExit(f"--root contains no files: {args.root}")

    total = sum((args.root / r if args.root.is_dir() else args.root).stat().st_size
                for r in manifest)
    print(f"staged {args.repo_id}")
    print(f"  source           {args.root}")
    print(f"  files            {len(manifest)}")
    print(f"  bytes            {total:,}")

    if args.dry_run:
        for rel in sorted(manifest):
            print(f"    {manifest[rel][:12]}  {rel}")
        print("  dry run          nothing uploaded")
        return

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    if args.verify_only:
        info = api.repo_info(args.repo_id, repo_type="dataset")
        print(f"  revision         {info.sha}")
        verify(api, args.repo_id, info.sha, manifest, args.allow_extra)
        print(f"\nverified {args.repo_id} (nothing uploaded)")
        return

    if not token:
        raise SystemExit("HF_TOKEN is not set -- refusing to attempt an upload")

    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    msg = args.commit_message or f"publish {len(manifest)} files"
    if args.root.is_dir():
        # delete_patterns so a file removed locally is removed on the Hub too;
        # without it the repo silently accumulates whatever it used to hold,
        # and the extra-files check below would then fail on every later run.
        #
        # This has to run BEFORE the card: "**" matches README.md, which is not
        # in the source tree, so uploading the card first would delete it here.
        api.upload_folder(
            folder_path=str(args.root), path_in_repo="", repo_id=args.repo_id,
            repo_type="dataset", delete_patterns=["**"], commit_message=msg,
        )
    else:
        api.upload_file(
            path_or_fileobj=str(args.root), path_in_repo=args.root.name,
            repo_id=args.repo_id, repo_type="dataset", commit_message=msg,
        )

    if args.card:
        card = args.card.read_text().replace("{n_files}", str(len(manifest)))
        api.upload_file(
            path_or_fileobj=card.encode(), path_in_repo="README.md",
            repo_id=args.repo_id, repo_type="dataset",
            commit_message="card",
        )

    revision = api.repo_info(args.repo_id, repo_type="dataset").sha
    print(f"  pushed           revision {revision}")
    verify(api, args.repo_id, revision, manifest, args.allow_extra)
    print(f"\npublished {args.repo_id}")
    print(f"  revision (retrieval coordinate only): {revision}")


if __name__ == "__main__":
    main()
