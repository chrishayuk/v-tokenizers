#!/usr/bin/env python3
"""Validate the frozen TOK-2 architecture and seed-control manifests."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def parameter_counts(config: dict, vocab_rows: int, ffn_width: int) -> dict[str, int]:
    invariants = config["invariants"]
    dim = invariants["d_model"]
    layers = invariants["layers"]
    kv_width = invariants["kv_heads"] * invariants["head_dimension"]
    embedding = vocab_rows * dim
    attention_per_layer = dim * dim + 2 * dim * kv_width + dim * dim
    ffn_per_layer = 3 * dim * ffn_width
    layer_norms = 2 * dim
    trunk = layers * (attention_per_layer + ffn_per_layer + layer_norms) + dim
    nominal = embedding + trunk
    phase3_trainable = nominal - layers * (ffn_per_layer + dim)
    return {
        "embedding_output_parameters": embedding,
        "trunk_parameters": trunk,
        "nominal_parameters": nominal,
        "phase1_trainable_parameters": nominal,
        "phase3_trainable_parameters": phase3_trainable,
    }


def validate(
    architecture_path: Path = HERE / "tok2_architecture_controls.json",
    seeds_path: Path = HERE / "tok2_paired_seeds.json",
) -> None:
    architecture = json.loads(architecture_path.read_text())
    target = architecture["reference"]["nominal_parameter_target"]
    tolerance = architecture["reference"]["maximum_absolute_mismatch_fraction"]
    width_multiple = architecture["invariants"]["ffn_width_hardware_multiple"]

    for control_name in ("fixed_trunk", "fixed_total"):
        for arm, pinned in architecture[control_name].items():
            computed = parameter_counts(architecture, pinned["vocab_rows"], pinned["ffn_width"])
            for field, value in computed.items():
                if pinned[field] != value:
                    raise AssertionError(
                        f"{control_name}.{arm}.{field}: pinned={pinned[field]} computed={value}"
                    )
            if pinned["ffn_width"] % width_multiple:
                raise AssertionError(f"{control_name}.{arm}: FFN width is not a multiple of {width_multiple}")
            if control_name == "fixed_total":
                mismatch = (computed["nominal_parameters"] - target) / target
                if abs(mismatch) > tolerance:
                    raise AssertionError(f"{arm}: fixed-total mismatch {mismatch} exceeds {tolerance}")
                if abs(mismatch - pinned["target_mismatch_fraction"]) > 1e-15:
                    raise AssertionError(f"{arm}: pinned mismatch does not match parameter count")

    seeds = json.loads(seeds_path.read_text())
    labels = [run["label"] for run in seeds["paired_runs"]]
    if labels != [0, 1, 2]:
        raise AssertionError(f"paired run labels must be [0, 1, 2], got {labels}")
    seed_fields = [field for field in seeds["paired_runs"][0] if field.endswith("_seed")]
    for field in seed_fields:
        values = [run[field] for run in seeds["paired_runs"]]
        if len(values) != len(set(values)):
            raise AssertionError(f"{field} is not unique across paired runs")


if __name__ == "__main__":
    validate()
    print("TOK-2 architecture and paired-seed controls are valid")
