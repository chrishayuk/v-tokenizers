#!/usr/bin/env python3
"""Apply the preregistered TOK-2 paired analysis to 48 completed results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .freeze_tok2_production import sha256_file
except ImportError:
    from freeze_tok2_production import sha256_file


HERE = Path(__file__).resolve().parent
T90_DF2 = 2.919985580353724


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_index(
    output_root: Path, manifest: dict[str, Any], manifest_sha: str
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[str]]:
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    missing: list[str] = []
    for run in manifest["runs"]:
        path = output_root / run["run_id"] / "training_results.json"
        if not path.exists():
            missing.append(run["run_id"])
            continue
        result = load_json(path)
        if result.get("status") != "complete":
            missing.append(run["run_id"])
            continue
        if result["production_manifest_sha256"] != manifest_sha:
            raise AssertionError(f"{run['run_id']} used a different frozen manifest")
        result_run = result["run"]
        key = (
            result_run["control"],
            result_run["arm_id"],
            int(result_run["paired_run_label"]),
        )
        if key in indexed:
            raise AssertionError(f"duplicate result key: {key}")
        indexed[key] = result
    return indexed, missing


def training_interval(values: list[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise AssertionError("paired training interval requires exactly three runs")
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    margin = T90_DF2 * standard_deviation / math.sqrt(3)
    return {
        "paired_differences": values,
        "mean_difference": mean,
        "sample_standard_deviation": standard_deviation,
        "two_sided_90_percent_student_t_interval_df2": [
            mean - margin,
            mean + margin,
        ],
        "caveat": "three paired training runs weakly characterize training variance",
    }


def stable_seed(base: int, *parts: object) -> int:
    material = "\0".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def paired_document_bootstrap(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    seed: int,
    replicates: int,
    interval: float,
) -> dict[str, Any]:
    candidate_rows = {
        row["text_sha256"]: row for row in candidate["per_document"]
    }
    incumbent_rows = {
        row["text_sha256"]: row for row in incumbent["per_document"]
    }
    if set(candidate_rows) != set(incumbent_rows):
        raise AssertionError("candidate/incumbent evaluation documents differ")
    keys = sorted(candidate_rows)
    candidate_loss = np.asarray(
        [candidate_rows[key]["loss_nats_sum"] for key in keys], dtype=np.float64
    )
    incumbent_loss = np.asarray(
        [incumbent_rows[key]["loss_nats_sum"] for key in keys], dtype=np.float64
    )
    raw_bytes = np.asarray(
        [candidate_rows[key]["raw_bytes"] for key in keys], dtype=np.float64
    )
    if any(
        candidate_rows[key]["raw_bytes"] != incumbent_rows[key]["raw_bytes"]
        for key in keys
    ):
        raise AssertionError("paired evaluation raw-byte denominators differ")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), size=(replicates, len(keys)))
    denominators = raw_bytes[indices].sum(axis=1) * math.log(2)
    differences = (
        candidate_loss[indices].sum(axis=1)
        - incumbent_loss[indices].sum(axis=1)
    ) / denominators
    tail = (1.0 - interval) / 2.0
    return {
        "documents": len(keys),
        "replicates": replicates,
        "seed": seed,
        "point_difference_corpus_bpb": (
            candidate_loss.sum() - incumbent_loss.sum()
        )
        / (raw_bytes.sum() * math.log(2)),
        "interval": [
            float(np.quantile(differences, tail)),
            float(np.quantile(differences, 1.0 - tail)),
        ],
        "interval_mass": interval,
        "label": "evaluation-set uncertainty only; not training-run uncertainty",
    }


def endpoint(result: dict[str, Any], phase: str) -> dict[str, Any]:
    return result[phase]["checkpoints"][-1]


def interpolate(points: list[tuple[float, float]], coordinate: float) -> float | None:
    ordered = sorted(points)
    if coordinate < ordered[0][0] or coordinate > ordered[-1][0]:
        return None
    for left, right in zip(ordered, ordered[1:]):
        if left[0] <= coordinate <= right[0]:
            if right[0] == left[0]:
                return right[1]
            fraction = (coordinate - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
    return ordered[-1][1]


def resource_views(
    indexed: dict[tuple[str, str, int], dict[str, Any]],
    controls: list[str],
    arms: list[str],
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for control in controls:
        for phase in ("phase1", "phase3"):
            for label in (0, 1, 2):
                results = [indexed[(control, arm, label)] for arm in arms]
                for coordinate_name in (
                    "estimated_training_flops",
                    "wall_clock_seconds",
                ):
                    common_budget = min(
                        endpoint(result, phase)["accounting"][coordinate_name]
                        for result in results
                    )
                    values: dict[str, float | None] = {}
                    for arm, result in zip(arms, results):
                        points = [
                            (
                                checkpoint["accounting"][coordinate_name],
                                checkpoint["evaluation"]["corpus_bpb"],
                            )
                            for checkpoint in result[phase]["checkpoints"]
                        ]
                        values[arm] = interpolate(points, common_budget)
                    views.append(
                        {
                            "control": control,
                            "phase": phase,
                            "paired_run_label": label,
                            "coordinate": coordinate_name,
                            "common_budget": common_budget,
                            "linear_interpolation_from_frozen_byte_checkpoints": values,
                        }
                    )
    return views


def validate_initialization_pairing(
    indexed: dict[tuple[str, str, int], dict[str, Any]],
    controls: list[str],
    arms: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for label in (0, 1, 2):
        fixed_trunk_hashes = {
            indexed[("fixed_trunk", arm, label)]["initialization_evidence"][
                "shared_trunk_sha256"
            ]
            for arm in arms
        }
        if len(fixed_trunk_hashes) != 1:
            raise AssertionError(
                f"fixed-trunk initialization is not paired for seed {label}"
            )
        checks.append(
            {
                "scope": "fixed_trunk_shared_trunk",
                "paired_run_label": label,
                "sha256": next(iter(fixed_trunk_hashes)),
            }
        )
        for control in controls:
            initial_attention = {
                indexed[(control, arm, label)]["initialization_evidence"][
                    "attention_sha256"
                ]
                for arm in arms
            }
            phase3_attention = {
                indexed[(control, arm, label)][
                    "phase3_attention_reinitialization_sha256"
                ]
                for arm in arms
            }
            if len(initial_attention) != 1 or len(phase3_attention) != 1:
                raise AssertionError(
                    f"attention initialization is not paired for {control} seed {label}"
                )
            checks.extend(
                [
                    {
                        "scope": f"{control}_phase1_attention",
                        "paired_run_label": label,
                        "sha256": next(iter(initial_attention)),
                    },
                    {
                        "scope": f"{control}_phase3_attention",
                        "paired_run_label": label,
                        "sha256": next(iter(phase3_attention)),
                    },
                ]
            )
    return {"status": "passed", "checks": checks}


def analyze(
    manifest: dict[str, Any],
    spec: dict[str, Any],
    indexed: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    arms = list(manifest["tokenizers"])
    controls = ["fixed_trunk", "fixed_total"]
    incumbent = spec["statistics"]["incumbent_arm"]
    comparisons: list[dict[str, Any]] = []
    endpoint_results: list[dict[str, Any]] = []
    for control in controls:
        for phase in ("phase1", "phase3"):
            for arm in arms:
                values = [
                    endpoint(indexed[(control, arm, label)], phase)["evaluation"][
                        "corpus_bpb"
                    ]
                    for label in (0, 1, 2)
                ]
                endpoint_results.append(
                    {
                        "control": control,
                        "phase": phase,
                        "arm_id": arm,
                        "bpb_by_paired_run_label": values,
                        "mean_bpb": statistics.mean(values),
                        "sample_standard_deviation": statistics.stdev(values),
                    }
                )
            for arm in arms:
                if arm == incumbent:
                    continue
                differences = [
                    endpoint(indexed[(control, arm, label)], phase)["evaluation"][
                        "corpus_bpb"
                    ]
                    - endpoint(
                        indexed[(control, incumbent, label)], phase
                    )["evaluation"]["corpus_bpb"]
                    for label in (0, 1, 2)
                ]
                comparison = {
                    "control": control,
                    "phase": phase,
                    "candidate_arm": arm,
                    "incumbent_arm": incumbent,
                    **training_interval(differences),
                }
                interval = comparison[
                    "two_sided_90_percent_student_t_interval_df2"
                ]
                comparison["decision"] = (
                    "unresolved"
                    if len({difference > 0 for difference in differences}) > 1
                    or interval[0] <= 0 <= interval[1]
                    else (
                        "candidate_lower_bpb"
                        if comparison["mean_difference"] < 0
                        else "incumbent_lower_bpb"
                    )
                )
                comparisons.append(comparison)

    bootstrap_spec = spec["statistics"]["document_bootstrap"]
    primary_bootstraps: list[dict[str, Any]] = []
    for arm in arms:
        if arm == incumbent:
            continue
        for label in (0, 1, 2):
            candidate = endpoint(
                indexed[("fixed_total", arm, label)], "phase3"
            )["evaluation"]
            reference = endpoint(
                indexed[("fixed_total", incumbent, label)], "phase3"
            )["evaluation"]
            primary_bootstraps.append(
                {
                    "candidate_arm": arm,
                    "incumbent_arm": incumbent,
                    "paired_run_label": label,
                    **paired_document_bootstrap(
                        candidate,
                        reference,
                        seed=stable_seed(
                            int(bootstrap_spec["seed"]),
                            "fixed_total",
                            "phase3",
                            arm,
                            label,
                        ),
                        replicates=int(bootstrap_spec["replicates"]),
                        interval=float(bootstrap_spec["interval"]),
                    ),
                }
            )

    primary = [
        row
        for row in comparisons
        if row["control"] == "fixed_total" and row["phase"] == "phase3"
    ]
    secondary = [
        row
        for row in comparisons
        if row["control"] == "fixed_trunk" and row["phase"] == "phase3"
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "primary_endpoint": spec["statistics"]["primary_endpoint"],
        "secondary_endpoint": spec["statistics"]["secondary_endpoint"],
        "initialization_pairing": validate_initialization_pairing(
            indexed, controls, arms
        ),
        "endpoint_results": endpoint_results,
        "primary_comparisons": primary,
        "secondary_comparisons": secondary,
        "all_endpoint_comparisons": comparisons,
        "primary_document_bootstraps": primary_bootstraps,
        "compute_normalized_views": resource_views(indexed, controls, arms),
        "interpretation_rule": spec["statistics"]["unresolved_rule"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=HERE / "tok2_production_manifest.json"
    )
    parser.add_argument(
        "--spec", type=Path, default=HERE / "tok2_production_spec.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=HERE / "tok2_production_runs"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    spec = load_json(args.spec)
    manifest_sha = sha256_file(args.manifest)
    indexed, missing = result_index(args.output_root, manifest, manifest_sha)
    if missing:
        message = f"{len(missing)} of 48 frozen runs are incomplete"
        if args.allow_incomplete:
            print(message)
            for run_id in missing:
                print(run_id)
            return
        raise SystemExit(message)
    analysis = analyze(manifest, spec, indexed)
    output = args.output or args.output_root / "tok2_decisive_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"wrote preregistered analysis: {output}")


if __name__ == "__main__":
    main()
