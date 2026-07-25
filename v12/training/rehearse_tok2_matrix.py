#!/usr/bin/env python3
"""Run the complete TOK-2 matrix through a tiny, non-ranking canary.

This is an orchestration rehearsal, not a language-model experiment.  It
loads the real tokenizer artifacts and frozen documents, validates the real
115M architecture on PyTorch's meta device, then runs a small tied-embedding
model through phase 1, phase 3, and held-out evaluation for all 48 cells.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import resource
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from .v11_ws_exact import WhitespaceExactV11
    from .validate_tok2_controls import validate as validate_controls
except ImportError:
    from v11_ws_exact import WhitespaceExactV11
    from validate_tok2_controls import validate as validate_controls


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(base: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else (base / path).resolve()


def tensor_set_sha256(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Document:
    domain: str
    source: str
    text: str
    text_sha256: str
    raw_bytes: int


class RuntimeTokenizer:
    def __init__(self, arm: dict[str, Any], artifact_path: Path):
        self.arm = arm
        self.kind = arm["kind"]
        self.vocab_size = int(arm["vocab_rows"])
        if self.kind == "v11_ws_exact":
            self.runtime = WhitespaceExactV11.from_file(artifact_path)
            self.bos_id = self.runtime.tokenizer.token_to_id("<s>")
            self.unk_id = self.runtime.tokenizer.token_to_id("<unk>")
        elif self.kind == "hf_tokenizer":
            self.runtime = Tokenizer.from_file(str(artifact_path))
            self.bos_id = self.runtime.token_to_id("<s>")
            self.unk_id = self.runtime.token_to_id("<unk>")
        elif self.kind == "pure_byte":
            json.loads(artifact_path.read_text())
            self.runtime = None
            self.bos_id = 2
            self.unk_id = 1
        else:
            raise ValueError(f"unsupported tokenizer kind: {self.kind}")
        if self.runtime_vocab_size() != self.vocab_size:
            raise AssertionError(
                f"{arm['id']} runtime has {self.runtime_vocab_size()} rows, "
                f"expected {self.vocab_size}"
            )

    def runtime_vocab_size(self) -> int:
        if self.kind == "v11_ws_exact":
            return self.runtime.vocab_size
        if self.kind == "hf_tokenizer":
            return self.runtime.get_vocab_size()
        return 260

    def encode_content(self, text: str) -> list[int]:
        if self.kind == "v11_ws_exact":
            return self.runtime.encode_ids(text)
        if self.kind == "hf_tokenizer":
            return list(self.runtime.encode(text).ids)
        return [4 + byte for byte in text.encode("utf-8")]

    def decode_content(self, ids: list[int]) -> str:
        if self.kind == "v11_ws_exact":
            return self.runtime.decode(ids)
        if self.kind == "hf_tokenizer":
            return self.runtime.decode(ids)
        return bytes(token - 4 for token in ids).decode("utf-8")

    def validate_and_encode(self, document: Document) -> list[int]:
        ids = self.encode_content(document.text)
        if self.unk_id in ids:
            raise AssertionError(f"{self.arm['id']} produced UNK for {document.text_sha256}")
        decoded = self.decode_content(ids)
        if decoded != document.text:
            raise AssertionError(
                f"{self.arm['id']} whole-input round trip failed for "
                f"{document.text_sha256}: {decoded!r} != {document.text!r}"
            )
        return [self.bos_id, *ids]


class CanaryLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, heads: int, ffn_width: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.attn_norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn = nn.MultiheadAttention(dim, heads, bias=False, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.gate = nn.Linear(dim, ffn_width, bias=False)
        self.up = nn.Linear(dim, ffn_width, bias=False)
        self.down = nn.Linear(ffn_width, dim, bias=False)
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        sequence = token_ids.shape[1]
        mask = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool, device=token_ids.device),
            diagonal=1,
        )
        normalized = self.attn_norm(x)
        attention, _ = self.attn(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        x = x + attention
        normalized = self.ffn_norm(x)
        x = x + self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        return F.linear(self.final_norm(x), self.embed.weight)


def initialize_canary(
    model: CanaryLM,
    trunk_seed: int,
    embedding_seed: int,
) -> None:
    trunk_generator = torch.Generator(device="cpu").manual_seed(trunk_seed)
    embedding_generator = torch.Generator(device="cpu").manual_seed(embedding_seed)
    for name, parameter in model.named_parameters():
        if name == "embed.weight":
            nn.init.normal_(parameter, mean=0.0, std=0.02, generator=embedding_generator)
        elif parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter, generator=trunk_generator)
        elif name.endswith("weight"):
            nn.init.ones_(parameter)
        else:
            nn.init.zeros_(parameter)


def reinitialize_attention(model: CanaryLM, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in model.attn.parameters():
        nn.init.xavier_uniform_(parameter, generator=generator)


def freeze_phase3_ffn(model: CanaryLM) -> None:
    for module in (model.ffn_norm, model.gate, model.up, model.down):
        for parameter in module.parameters():
            parameter.requires_grad = False


def canary_ffn_width(production_width: int, spec: dict[str, Any]) -> int:
    canary = spec["canary_model"]
    divisor = int(canary["production_ffn_scale_divisor"])
    multiple = int(canary["ffn_width_multiple"])
    return math.ceil((production_width / divisor) / multiple) * multiple


def load_c8_candidates(
    path: Path,
    minimum_bytes: int,
    maximum_bytes: int,
    data_order: dict[str, Any],
    documents_per_domain: int,
) -> dict[str, list[Document]]:
    phase_seeds = {
        phase: int(data_order[f"{phase}_document_order_seed"])
        for phase in ("phase1", "phase3")
    }
    retained: dict[str, dict[str, list[tuple[str, Document]]]] = {
        phase: defaultdict(list) for phase in phase_seeds
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            text = raw["text"]
            byte_count = len(text.encode("utf-8"))
            if not minimum_bytes <= byte_count <= maximum_bytes:
                continue
            digest = sha256_bytes(text.encode("utf-8"))
            document = Document(
                domain=raw["domain"],
                source=str(raw["source"]),
                text=text,
                text_sha256=digest,
                raw_bytes=byte_count,
            )
            for phase, seed in phase_seeds.items():
                key = sha256_bytes(
                    f"{seed}\0{document.domain}\0{document.text_sha256}".encode()
                )
                candidates = retained[phase][document.domain]
                candidates.append((key, document))
                candidates.sort(key=lambda item: (item[0], item[1].text_sha256))
                # Phase 3 may need to skip every phase-1 choice.  Keeping
                # twice the requested count is sufficient while remaining
                # bounded independently of corpus size.
                limit = documents_per_domain * (2 if phase == "phase3" else 1)
                del candidates[limit:]

    by_domain: dict[str, dict[str, Document]] = defaultdict(dict)
    for phase_candidates in retained.values():
        for domain, candidates in phase_candidates.items():
            for _, document in candidates:
                by_domain[domain][document.text_sha256] = document
    return {
        domain: list(documents.values())
        for domain, documents in by_domain.items()
    }


def ordered_training_streams(
    candidates: dict[str, list[Document]],
    data_order: dict[str, Any],
    documents_per_domain: int,
) -> tuple[list[Document], list[Document]]:
    selected: dict[str, list[Document]] = {}
    used: set[str] = set()
    for phase in ("phase1", "phase3"):
        seed = int(data_order[f"{phase}_document_order_seed"])
        phase_documents: list[Document] = []
        for domain in sorted(candidates):
            ordered = sorted(
                (document for document in candidates[domain] if document.text_sha256 not in used),
                key=lambda document: (
                    sha256_bytes(
                        f"{seed}\0{domain}\0{document.text_sha256}".encode()
                    ),
                    document.text_sha256,
                ),
            )
            if len(ordered) < documents_per_domain:
                raise ValueError(f"{domain} lacks rehearsal documents for {phase}")
            chosen = ordered[:documents_per_domain]
            phase_documents.extend(chosen)
            used.update(document.text_sha256 for document in chosen)
        phase_documents.sort(
            key=lambda document: sha256_bytes(
                f"{seed}\0{document.domain}\0{document.text_sha256}".encode()
            )
        )
        selected[phase] = phase_documents
    return selected["phase1"], selected["phase3"]


def load_evaluation_stream(
    heldout_path: Path,
    seed: int,
    documents_per_domain: int,
) -> list[Document]:
    by_domain: dict[str, list[Document]] = defaultdict(list)
    with heldout_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            text = raw["text"]
            by_domain[raw["domain"]].append(
                Document(
                    domain=raw["domain"],
                    source=str(raw["source"]),
                    text=text,
                    text_sha256=sha256_bytes(text.encode("utf-8")),
                    raw_bytes=len(text.encode("utf-8")),
                )
            )
    selected: list[Document] = []
    for domain in sorted(by_domain):
        ordered = sorted(
            by_domain[domain],
            key=lambda document: (
                sha256_bytes(
                    f"{seed}\0{domain}\0{document.text_sha256}".encode()
                ),
                document.text_sha256,
            ),
        )
        selected.extend(ordered[:documents_per_domain])
    selected.sort(
        key=lambda document: sha256_bytes(
            f"{seed}\0{document.domain}\0{document.text_sha256}".encode()
        )
    )
    return selected


def stream_evidence(documents: list[Document]) -> dict[str, Any]:
    ordered_hashes = [document.text_sha256 for document in documents]
    material = "\n".join(
        f"{document.domain}\0{document.raw_bytes}\0{document.text_sha256}"
        for document in documents
    ).encode()
    return {
        "documents": len(documents),
        "raw_bytes": sum(document.raw_bytes for document in documents),
        "ordered_text_sha256": ordered_hashes,
        "ordered_stream_sha256": sha256_bytes(material),
        "domains": {
            domain: sum(document.domain == domain for document in documents)
            for domain in sorted({document.domain for document in documents})
        },
    }


def token_chunks(token_ids: list[int], maximum_sequence_length: int) -> list[list[int]]:
    if len(token_ids) < 2:
        return []
    chunks = []
    for start in range(0, len(token_ids) - 1, maximum_sequence_length - 1):
        content = token_ids[start : start + maximum_sequence_length]
        if len(content) >= 2:
            chunks.append(content)
    return chunks


def phase_pass(
    model: CanaryLM,
    runtime: RuntimeTokenizer,
    documents: list[Document],
    *,
    learning_rate: float,
    stochasticity_seed: int,
    maximum_sequence_length: int,
    production_trainable_parameters: int,
) -> dict[str, Any]:
    torch.manual_seed(stochasticity_seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )
    start = time.perf_counter()
    input_tokens = 0
    predicted_tokens = 0
    optimizer_steps = 0
    loss_sum = 0.0
    input_rows: set[int] = set()
    target_rows: set[int] = set()
    for document in documents:
        encoded = runtime.validate_and_encode(document)
        input_tokens += len(encoded)
        input_rows.update(encoded)
        for chunk in token_chunks(encoded, maximum_sequence_length):
            batch = torch.tensor([chunk], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            targets = batch[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, runtime.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            count = targets.numel()
            predicted_tokens += count
            target_rows.update(targets.reshape(-1).tolist())
            optimizer_steps += 1
            loss_sum += float(loss.detach()) * count
    wall = time.perf_counter() - start
    canary_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "raw_bytes_consumed": sum(document.raw_bytes for document in documents),
        "documents_consumed": len(documents),
        "input_tokens_consumed": input_tokens,
        "target_tokens_predicted": predicted_tokens,
        "optimizer_steps": optimizer_steps,
        "mean_loss_nats_per_token": loss_sum / max(predicted_tokens, 1),
        "estimated_production_training_flops": 6
        * production_trainable_parameters
        * predicted_tokens,
        "wall_clock_seconds": wall,
        "peak_host_memory_bytes_process": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin"
            else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "canary_trainable_parameters": canary_trainable,
        "input_rows_observed": len(input_rows),
        "target_rows_observed": len(target_rows),
        "output_rows_exercised_by_full_softmax": runtime.vocab_size,
    }


@torch.no_grad()
def evaluate(
    model: CanaryLM,
    runtime: RuntimeTokenizer,
    documents: list[Document],
    maximum_sequence_length: int,
) -> dict[str, Any]:
    model.eval()
    start = time.perf_counter()
    loss_sum = 0.0
    predicted_tokens = 0
    input_tokens = 0
    per_domain: dict[str, dict[str, float]] = defaultdict(
        lambda: {"loss_sum": 0.0, "predicted_tokens": 0, "raw_bytes": 0}
    )
    for document in documents:
        encoded = runtime.validate_and_encode(document)
        input_tokens += len(encoded)
        per_domain[document.domain]["raw_bytes"] += document.raw_bytes
        for chunk in token_chunks(encoded, maximum_sequence_length):
            batch = torch.tensor([chunk], dtype=torch.long)
            logits = model(batch)
            targets = batch[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, runtime.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
            count = targets.numel()
            value = float(loss)
            loss_sum += value
            predicted_tokens += count
            per_domain[document.domain]["loss_sum"] += value
            per_domain[document.domain]["predicted_tokens"] += count
    raw_bytes = sum(document.raw_bytes for document in documents)
    return {
        "raw_bytes": raw_bytes,
        "documents": len(documents),
        "input_tokens": input_tokens,
        "predicted_tokens": predicted_tokens,
        "loss_sum_nats": loss_sum,
        "canary_bpb": loss_sum / (math.log(2) * raw_bytes),
        "wall_clock_seconds": time.perf_counter() - start,
        "by_domain_canary_bpb": {
            domain: values["loss_sum"] / (math.log(2) * values["raw_bytes"])
            for domain, values in sorted(per_domain.items())
        },
    }


def load_tiny_model_class(model_path: Path):
    spec = importlib.util.spec_from_file_location("tok2_pinned_tiny_model", model_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TinyModel


def validate_production_modules(
    architecture: dict[str, Any],
    architecture_path: Path,
) -> dict[str, Any]:
    reference = architecture["reference"]
    tiny_model_root = HERE.parents[2] / "tiny-model"
    model_path = tiny_model_root / reference["model_source"]
    config_path = tiny_model_root / reference["config_source"]
    actual_source_sha = sha256_file(model_path)
    if actual_source_sha != reference["model_source_sha256"]:
        raise AssertionError(
            f"TinyModel source mismatch: expected {reference['model_source_sha256']}, "
            f"got {actual_source_sha}"
        )
    actual_config_sha = sha256_file(config_path)
    if actual_config_sha != reference["config_source_sha256"]:
        raise AssertionError(
            f"TinyModel config mismatch: expected {reference['config_source_sha256']}, "
            f"got {actual_config_sha}"
        )
    TinyModel = load_tiny_model_class(model_path)
    invariants = architecture["invariants"]
    evidence: dict[str, Any] = {
        "model_source": str(model_path),
        "model_source_sha256": actual_source_sha,
        "config_source": str(config_path),
        "config_source_sha256": actual_config_sha,
        "configurations": {},
    }
    checked: set[tuple[int, int]] = set()
    for control in ("fixed_trunk", "fixed_total"):
        for family, pinned in architecture[control].items():
            key = (int(pinned["vocab_rows"]), int(pinned["ffn_width"]))
            if key in checked:
                continue
            checked.add(key)
            with torch.device("meta"):
                model = TinyModel(
                    vocab_size=key[0],
                    dim=invariants["d_model"],
                    n_layers=invariants["layers"],
                    ffn_dim=key[1],
                    n_heads=invariants["attention_heads"],
                    n_kv_heads=invariants["kv_heads"],
                    max_seq=invariants["maximum_sequence_length"],
                )
            nominal = sum(parameter.numel() for parameter in model.parameters())
            if nominal != pinned["nominal_parameters"]:
                raise AssertionError(
                    f"actual TinyModel count mismatch for {key}: "
                    f"{nominal} != {pinned['nominal_parameters']}"
                )
            for layer in model.layers:
                for parameter in layer.ffn.parameters():
                    parameter.requires_grad = False
                for parameter in layer.ffn_norm.parameters():
                    parameter.requires_grad = False
            phase3 = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            if phase3 != pinned["phase3_trainable_parameters"]:
                raise AssertionError(
                    f"actual TinyModel phase3 count mismatch for {key}: "
                    f"{phase3} != {pinned['phase3_trainable_parameters']}"
                )
            evidence["configurations"][f"vocab{key[0]}_ffn{key[1]}"] = {
                "nominal_parameters": nominal,
                "phase3_trainable_parameters": phase3,
            }
            del model
    evidence["architecture_spec_sha256"] = sha256_file(architecture_path)
    return evidence


def verify_input(path: Path, expected_sha256: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise AssertionError(
            f"{label} hash mismatch: expected {expected_sha256}, got {actual}"
        )


def run(spec_path: Path, output_path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = json.loads(spec_path.read_text())
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION}")
    inputs = spec["inputs"]
    arms_path = resolve(spec_path.parent, inputs["arms"])
    architecture_path = resolve(spec_path.parent, inputs["architecture_controls"])
    seeds_path = resolve(spec_path.parent, inputs["paired_seeds"])
    c8_path = resolve(spec_path.parent, inputs["c8_corpus"])
    c3_manifest_path = resolve(spec_path.parent, inputs["c3_manifest"])
    c3_heldout_path = resolve(spec_path.parent, inputs["c3_heldout"])
    verify_input(c8_path, inputs["c8_corpus_sha256"], "C8")
    verify_input(c3_manifest_path, inputs["c3_manifest_sha256"], "C3 manifest")
    verify_input(c3_heldout_path, inputs["c3_heldout_sha256"], "C3 heldout")
    validate_controls(architecture_path, seeds_path)

    arm_manifest = json.loads(arms_path.read_text())
    architecture = json.loads(architecture_path.read_text())
    seeds = json.loads(seeds_path.read_text())
    arms = arm_manifest["arms"]
    expected_arms = int(spec["matrix"]["expected_arms"])
    if len(arms) != expected_arms:
        raise AssertionError(f"expected {expected_arms} arms, got {len(arms)}")
    production_modules = validate_production_modules(architecture, architecture_path)

    runtimes: dict[str, RuntimeTokenizer] = {}
    artifact_evidence: dict[str, Any] = {}
    for arm in arms:
        artifact_path = resolve(arms_path.parent, arm["artifact"])
        verify_input(artifact_path, arm["artifact_sha256"], arm["id"])
        runtime = RuntimeTokenizer(arm, artifact_path)
        runtimes[arm["id"]] = runtime
        artifact_evidence[arm["id"]] = {
            "artifact": arm["artifact"],
            "sha256": arm["artifact_sha256"],
            "runtime_vocab_rows": runtime.runtime_vocab_size(),
        }

    data_spec = spec["data_rehearsal"]
    candidates = load_c8_candidates(
        c8_path,
        int(data_spec["training_document_min_bytes"]),
        int(data_spec["training_document_max_bytes"]),
        seeds["data_order"],
        int(data_spec["training_documents_per_domain_per_phase"]),
    )
    required_domains = set(data_spec["required_domains"])
    if set(candidates) != required_domains:
        raise AssertionError(
            f"training rehearsal domain coverage mismatch: "
            f"expected {sorted(required_domains)}, got {sorted(candidates)}"
        )
    phase1_documents, phase3_documents = ordered_training_streams(
        candidates,
        seeds["data_order"],
        int(data_spec["training_documents_per_domain_per_phase"]),
    )
    evaluation_documents = load_evaluation_stream(
        c3_heldout_path,
        int(seeds["data_order"]["evaluation_order_seed"]),
        int(data_spec["evaluation_documents_per_domain"]),
    )
    evaluation_domains = {document.domain for document in evaluation_documents}
    if evaluation_domains != required_domains:
        raise AssertionError(
            f"evaluation rehearsal domain coverage mismatch: "
            f"expected {sorted(required_domains)}, got {sorted(evaluation_domains)}"
        )
    streams = {
        "phase1": stream_evidence(phase1_documents),
        "phase3": stream_evidence(phase3_documents),
        "evaluation": stream_evidence(evaluation_documents),
    }

    canary = spec["canary_model"]
    maximum_sequence_length = int(canary["maximum_sequence_length"])
    results = []
    initialization_evidence: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for control in spec["matrix"]["controls"]:
        for run_seed in seeds["paired_runs"]:
            label = int(run_seed["label"])
            for arm in arms:
                runtime = runtimes[arm["id"]]
                pinned = architecture[control][arm["family"]]
                ffn_width = canary_ffn_width(int(pinned["ffn_width"]), spec)
                model = CanaryLM(
                    runtime.vocab_size,
                    int(canary["dimension"]),
                    int(canary["attention_heads"]),
                    ffn_width,
                )
                initialize_canary(
                    model,
                    int(run_seed["trunk_initialization_seed"]),
                    int(run_seed["embedding_output_initialization_seed"]),
                )
                trunk_hash = tensor_set_sha256(
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if name != "embed.weight"
                )
                attention_hash = tensor_set_sha256(model.attn.named_parameters())
                embedding_hash = tensor_set_sha256(
                    [("embed.weight", model.embed.weight)]
                )
                initialization_evidence[
                    f"{control}:run{label}:attention"
                ][attention_hash].append(arm["id"])
                if control == "fixed_trunk":
                    initialization_evidence[
                        f"{control}:run{label}:trunk"
                    ][trunk_hash].append(arm["id"])

                phase1 = phase_pass(
                    model,
                    runtime,
                    phase1_documents,
                    learning_rate=float(canary["phase1_learning_rate"]),
                    stochasticity_seed=int(run_seed["phase1_stochasticity_seed"]),
                    maximum_sequence_length=maximum_sequence_length,
                    production_trainable_parameters=int(
                        pinned["phase1_trainable_parameters"]
                    ),
                )
                reinitialize_attention(
                    model, int(run_seed["phase3_attention_reinitialization_seed"])
                )
                phase3_attention_hash = tensor_set_sha256(
                    model.attn.named_parameters()
                )
                initialization_evidence[
                    f"phase3:run{label}:attention"
                ][phase3_attention_hash].append(f"{control}:{arm['id']}")
                freeze_phase3_ffn(model)
                phase3 = phase_pass(
                    model,
                    runtime,
                    phase3_documents,
                    learning_rate=float(canary["phase3_learning_rate"]),
                    stochasticity_seed=int(run_seed["phase3_stochasticity_seed"]),
                    maximum_sequence_length=maximum_sequence_length,
                    production_trainable_parameters=int(
                        pinned["phase3_trainable_parameters"]
                    ),
                )
                evaluation = evaluate(
                    model, runtime, evaluation_documents, maximum_sequence_length
                )
                results.append(
                    {
                        "control": control,
                        "run_label": label,
                        "arm": arm["id"],
                        "family": arm["family"],
                        "status": "rehearsal-pass-not-a-model-result",
                        "phase1_completed": True,
                        "phase3_completed": True,
                        "evaluation_completed": True,
                        "phase1_eliminated": False,
                        "production_parameters": pinned,
                        "canary": {
                            "dimension": int(canary["dimension"]),
                            "layers": int(canary["layers"]),
                            "ffn_width": ffn_width,
                            "nominal_parameters": sum(
                                parameter.numel() for parameter in model.parameters()
                            ),
                        },
                        "initialization": {
                            "trunk_sha256": trunk_hash,
                            "attention_sha256": attention_hash,
                            "embedding_output_sha256": embedding_hash,
                            "phase3_attention_sha256": phase3_attention_hash,
                        },
                        "streams": {
                            phase: evidence["ordered_stream_sha256"]
                            for phase, evidence in streams.items()
                        },
                        "phase1": phase1,
                        "phase3": phase3,
                        "evaluation": evaluation,
                    }
                )
                del model
                gc.collect()

    expected_cells = int(spec["matrix"]["expected_cells"])
    if len(results) != expected_cells:
        raise AssertionError(f"expected {expected_cells} cells, got {len(results)}")
    for key, hashes in initialization_evidence.items():
        if len(hashes) != 1:
            raise AssertionError(f"{key} diverged across arms: {dict(hashes)}")
    for result in results:
        for phase in ("phase1", "phase3"):
            if result["streams"][phase] != streams[phase]["ordered_stream_sha256"]:
                raise AssertionError(f"{result['arm']} saw a divergent {phase} stream")
        if result["phase1_eliminated"] or not result["phase3_completed"]:
            raise AssertionError(f"no-phase-one-elimination invariant failed: {result}")

    result = {
        "schema_version": SCHEMA_VERSION,
        "rehearsal_id": spec["rehearsal_id"],
        "status": "pass-canary-only-not-a-model-result",
        "spec_sha256": sha256_file(spec_path),
        "arms_manifest_sha256": sha256_file(arms_path),
        "architecture_controls_sha256": sha256_file(architecture_path),
        "paired_seeds_sha256": sha256_file(seeds_path),
        "matrix": {
            "arms": len(arms),
            "controls": len(spec["matrix"]["controls"]),
            "paired_runs": len(seeds["paired_runs"]),
            "cells_expected": expected_cells,
            "cells_completed": len(results),
            "phase1_eliminations": 0,
        },
        "streams": streams,
        "tokenizer_artifacts": artifact_evidence,
        "production_module_validation": production_modules,
        "initialization_invariants": {
            key: {
                "sha256": next(iter(hashes)),
                "arms_or_cells": next(iter(hashes.values())),
            }
            for key, hashes in sorted(initialization_evidence.items())
        },
        "results": results,
        "interpretation": (
            "This proves the frozen artifacts and orchestration can traverse "
            "all cells. Canary losses, BPB, FLOPs, memory, and timings cannot "
            "rank tokenizers or predict the decisive full runs."
        ),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=HERE / "tok2_rehearsal_spec.json"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "tok2_rehearsal_results.json"
    )
    args = parser.parse_args()
    result = run(args.spec.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "matrix": result["matrix"],
                "streams": result["streams"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
