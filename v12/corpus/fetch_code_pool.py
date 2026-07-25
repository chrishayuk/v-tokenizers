#!/usr/bin/env python3
"""Populate canonical local caches for the frozen C8 v3 code repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def verify(repo: Path, configured: dict[str, Any]) -> dict[str, Any]:
    commit = configured["commit"]
    resolved = git(repo, "rev-parse", f"{commit}^{{commit}}")
    if resolved != commit:
        raise ValueError(f"{configured['id']} resolved {commit} to {resolved}")
    license_bytes = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "cat-file",
            "blob",
            f"{commit}:{configured['license_evidence_path']}",
        ]
    )
    license_sha256 = hashlib.sha256(license_bytes).hexdigest()
    if license_sha256 != configured["license_evidence_sha256"]:
        raise ValueError(
            f"{configured['id']} licence evidence mismatch: "
            f"{license_sha256} != {configured['license_evidence_sha256']}"
        )
    return {
        "id": configured["id"],
        "path": str(repo),
        "commit": commit,
        "license_evidence_sha256": license_sha256,
    }


def fetch(spec_path: Path, reuse_local_candidates: bool = False) -> list[dict[str, Any]]:
    spec = json.loads(spec_path.read_text())
    results = []
    for configured in spec["repositories"]:
        primary = Path(configured["path"])
        if not primary.is_absolute():
            primary = (spec_path.parent / primary).resolve()
        if (primary / ".git").exists():
            results.append(verify(primary, configured))
            continue

        if reuse_local_candidates:
            for candidate_value in configured.get("local_candidates", []):
                candidate = Path(candidate_value)
                if not candidate.is_absolute():
                    candidate = (spec_path.parent / candidate).resolve()
                if (candidate / ".git").exists():
                    results.append(verify(candidate, configured))
                    break
            else:
                raise FileNotFoundError(
                    f"{configured['id']} has no canonical cache or usable local candidate"
                )
            continue

        primary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", configured["url"], str(primary)],
            check=True,
        )
        try:
            git(primary, "cat-file", "-e", f"{configured['commit']}^{{commit}}")
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "-C", str(primary), "fetch", "origin", configured["commit"]],
                check=True,
            )
        results.append(verify(primary, configured))
    return results


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=here / "code_pool_spec.json")
    parser.add_argument(
        "--reuse-local-candidates",
        action="store_true",
        help="verify existing sibling clones instead of populating canonical caches",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            fetch(args.spec.resolve(), args.reuse_local_candidates),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
