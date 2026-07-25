#!/usr/bin/env python3
"""Run one frozen cell of the decisive, byte-scheduled TOK-2 matrix.

Expensive training requires both ``--run-id`` and ``--execute``.  The default
``--preflight`` path validates all 48 cells, immutable inputs, tokenizer
contracts, complete-document byte boundaries, and real TinyModel parameter
counts without allocating model weights or taking an optimizer step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from .freeze_tok2_production import (
        build_manifest,
        order_key,
        resolve,
        sha256_bytes,
        sha256_file,
    )
    from .v11_ws_exact import WhitespaceExactV11
except ImportError:
    from freeze_tok2_production import (
        build_manifest,
        order_key,
        resolve,
        sha256_bytes,
        sha256_file,
    )
    from v11_ws_exact import WhitespaceExactV11


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Document:
    domain: str
    source: str
    text: str
    text_sha256: str
    raw_bytes: int
    line_number: int


@dataclass
class Accounting:
    raw_bytes_consumed: int = 0
    documents_consumed: int = 0
    input_tokens_consumed: int = 0
    target_tokens_predicted: int = 0
    optimizer_steps: int = 0
    loss_nats_sum: float = 0.0
    estimated_training_flops: int = 0
    wall_clock_seconds: float = 0.0
    peak_device_memory_bytes: int = 0
    peak_host_memory_bytes: int = 0


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
            json.loads(artifact_path.read_text(encoding="utf-8"))
            self.runtime = None
            self.bos_id = 2
            self.unk_id = 1
        else:
            raise ValueError(f"unsupported tokenizer kind: {self.kind}")
        if self.bos_id is None or self.unk_id is None:
            raise AssertionError(f"{arm['id']} is missing BOS or UNK")
        if self.runtime_vocab_size() != self.vocab_size:
            raise AssertionError(
                f"{arm['id']} runtime rows {self.runtime_vocab_size()} "
                f"!= frozen {self.vocab_size}"
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
        return [4 + value for value in text.encode("utf-8")]

    def decode_content(self, ids: list[int]) -> str:
        if self.kind == "v11_ws_exact":
            return self.runtime.decode(ids)
        if self.kind == "hf_tokenizer":
            return self.runtime.decode(ids)
        return bytes(value - 4 for value in ids).decode("utf-8")

    def encode_document(self, document: Document, verify: bool = True) -> list[int]:
        ids = self.encode_content(document.text)
        if self.unk_id in ids:
            raise AssertionError(
                f"{self.arm['id']} emitted UNK for {document.text_sha256}"
            )
        if verify and self.decode_content(ids) != document.text:
            raise AssertionError(
                f"{self.arm['id']} whole-input round trip failed for "
                f"{document.text_sha256}"
            )
        return [int(self.bos_id), *ids]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            text = row["text"]
            raw = text.encode("utf-8")
            documents.append(
                Document(
                    domain=row["domain"],
                    source=row["source"],
                    text=text,
                    text_sha256=sha256_bytes(raw),
                    raw_bytes=len(raw),
                    line_number=line_number,
                )
            )
    return documents


def ordered_stream(
    path: Path, evidence: dict[str, Any]
) -> list[Document]:
    documents = load_documents(path)
    seed = int(evidence["order_seed"])
    documents.sort(key=lambda item: order_key(seed, item))
    selected = documents[: int(evidence["documents"])]
    if sum(item.raw_bytes for item in selected) != int(evidence["actual_raw_bytes"]):
        raise AssertionError("production stream raw-byte evidence mismatch")
    material = "".join(
        f"{item.line_number}\0{item.domain}\0{item.source}\0{item.raw_bytes}\0"
        f"{item.text_sha256}\n"
        for item in selected
    ).encode("utf-8")
    if sha256_bytes(material) != evidence["ordered_stream_sha256"]:
        raise AssertionError("production ordered-stream SHA-256 mismatch")
    return selected


def model_module(spec: dict[str, Any], spec_path: Path):
    source = resolve(spec_path.parent, spec["inputs"]["tiny_model_source"])
    expected = spec["inputs"]["tiny_model_source_sha256"]
    if sha256_file(source) != expected:
        raise AssertionError("TinyModel source no longer matches frozen SHA-256")
    repository = source.parents[4]
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = completed.stdout.strip()
    expected_commit = spec["inputs"]["tiny_model_repository_commit"]
    if actual_commit != expected_commit:
        raise AssertionError(
            f"TinyModel repository commit {actual_commit} != {expected_commit}"
        )
    module_spec = importlib.util.spec_from_file_location("tok2_frozen_tiny_model", source)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot import TinyModel from {source}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def make_model(TinyModel, run: dict[str, Any], architecture: dict[str, Any]):
    invariants = architecture["invariants"]
    return TinyModel(
        vocab_size=int(run["vocab_rows"]),
        dim=int(invariants["d_model"]),
        n_layers=int(invariants["layers"]),
        ffn_dim=int(run["ffn_width"]),
        n_heads=int(invariants["attention_heads"]),
        n_kv_heads=int(invariants["kv_heads"]),
        max_seq=int(invariants["maximum_sequence_length"]),
    )


def tensor_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def initialize_matrix(parameter: torch.Tensor, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.nn.init.xavier_uniform_(parameter, generator=generator)


def initialize_model(model: torch.nn.Module, run: dict[str, Any]) -> None:
    trunk_seed = int(run["seeds"]["trunk_initialization_seed"])
    embedding_seed = int(run["seeds"]["embedding_output_initialization_seed"])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dim() > 1:
                base = embedding_seed if name == "embed.weight" else trunk_seed
                initialize_matrix(parameter, tensor_seed(base, name))
            elif name.endswith("weight"):
                parameter.fill_(1.0)
            else:
                parameter.zero_()


def reinitialize_attention(model: torch.nn.Module, seed: int) -> None:
    with torch.no_grad():
        for layer_index, layer in enumerate(model.layers):
            for name, parameter in layer.attn.named_parameters():
                canonical = f"layers.{layer_index}.attn.{name}"
                cpu = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
                initialize_matrix(cpu, tensor_seed(seed, canonical))
                parameter.copy_(cpu.to(parameter.device))


def configure_phase3(model: torch.nn.Module, seed: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    reinitialize_attention(model, seed)
    for layer in model.layers:
        for parameter in layer.ffn.parameters():
            parameter.requires_grad = False
        for parameter in layer.ffn_norm.parameters():
            parameter.requires_grad = False


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    nominal = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return nominal, trainable


def tensor_set_sha256(
    named_tensors: Iterable[tuple[str, torch.Tensor]]
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_stochasticity(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def lr_at_progress(
    peak: float,
    progress: float,
    warmup_fraction: float,
    minimum_fraction: float,
) -> float:
    progress = min(max(progress, 0.0), 1.0)
    if progress < warmup_fraction:
        return peak * max(progress / warmup_fraction, 1e-6)
    cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
    multiplier = minimum_fraction + (1.0 - minimum_fraction) * (
        0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    )
    return peak * multiplier


def packed_sequences(
    documents: Iterable[Document],
    tokenizer: RuntimeTokenizer,
    maximum_sequence_length: int,
    token_frequency: Counter[int],
) -> Iterator[tuple[list[int], int, int]]:
    buffer: list[int] = []
    completions: deque[tuple[int, int]] = deque()
    for document in documents:
        ids = tokenizer.encode_document(document)
        token_frequency.update(ids)
        buffer.extend(ids)
        completions.append((len(buffer), document.raw_bytes))
        while len(buffer) >= maximum_sequence_length:
            complete_documents = 0
            complete_bytes = 0
            while completions and completions[0][0] <= maximum_sequence_length:
                _, raw_bytes = completions.popleft()
                complete_documents += 1
                complete_bytes += raw_bytes
            yield buffer[:maximum_sequence_length], complete_documents, complete_bytes
            del buffer[:maximum_sequence_length]
            completions = deque(
                (position - maximum_sequence_length, raw_bytes)
                for position, raw_bytes in completions
            )
    if buffer:
        yield (
            buffer,
            len(completions),
            sum(raw_bytes for _, raw_bytes in completions),
        )


def batches(
    sequences: Iterable[tuple[list[int], int, int]], batch_size: int
) -> Iterator[list[tuple[list[int], int, int]]]:
    pending: list[tuple[list[int], int, int]] = []
    for sequence in sequences:
        pending.append(sequence)
        if len(pending) == batch_size:
            yield pending
            pending = []
    if pending:
        yield pending


def training_batch(
    batch: list[tuple[list[int], int, int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    maximum = max(len(row[0]) for row in batch)
    if maximum < 2:
        maximum = 2
    input_ids = torch.zeros((len(batch), maximum), dtype=torch.long)
    target_mask = torch.zeros((len(batch), maximum - 1), dtype=torch.bool)
    input_tokens = 0
    predicted_tokens = 0
    documents = 0
    raw_bytes = 0
    for index, (ids, completed_documents, completed_bytes) in enumerate(batch):
        input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        if len(ids) > 1:
            target_mask[index, : len(ids) - 1] = True
        input_tokens += len(ids)
        predicted_tokens += max(0, len(ids) - 1)
        documents += completed_documents
        raw_bytes += completed_bytes
    return (
        input_ids.to(device),
        target_mask.to(device),
        input_tokens,
        predicted_tokens,
        documents,
        raw_bytes,
    )


def host_peak_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def device_peak_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    return 0


def reset_device_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def update_memory(accounting: Accounting, device: torch.device) -> None:
    accounting.peak_host_memory_bytes = max(
        accounting.peak_host_memory_bytes, host_peak_bytes()
    )
    accounting.peak_device_memory_bytes = max(
        accounting.peak_device_memory_bytes, device_peak_bytes(device)
    )


def optimizer_for(
    model: torch.nn.Module, spec: dict[str, Any], peak_lr: float
) -> torch.optim.Optimizer:
    optimization = spec["optimization"]
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=peak_lr,
        betas=tuple(float(value) for value in optimization["adam_betas"]),
        eps=float(optimization["adam_epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )


def train_documents(
    *,
    model: torch.nn.Module,
    tokenizer: RuntimeTokenizer,
    documents: list[Document],
    optimizer: torch.optim.Optimizer,
    accounting: Accounting,
    phase_raw_bytes: int,
    phase_trainable_parameters: int,
    peak_lr: float,
    spec: dict[str, Any],
    device: torch.device,
    token_frequency: Counter[int],
) -> None:
    optimization = spec["optimization"]
    start = time.monotonic()
    model.train()
    stream = packed_sequences(
        documents,
        tokenizer,
        int(optimization["maximum_sequence_length"]),
        token_frequency,
    )
    for packed in batches(stream, int(optimization["batch_size_sequences"])):
        (
            input_ids,
            target_mask,
            input_tokens,
            predicted_tokens,
            completed_documents,
            completed_bytes,
        ) = training_batch(packed, device)
        if predicted_tokens == 0:
            accounting.documents_consumed += completed_documents
            accounting.raw_bytes_consumed += completed_bytes
            accounting.input_tokens_consumed += input_tokens
            continue
        progress = (accounting.raw_bytes_consumed + completed_bytes) / phase_raw_bytes
        learning_rate = lr_at_progress(
            peak_lr,
            progress,
            float(optimization["warmup_fraction"]),
            float(optimization["minimum_lr_fraction"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        flat_losses = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
            input_ids[:, 1:].reshape(-1),
            reduction="none",
        )
        mask = target_mask.reshape(-1)
        loss_sum = flat_losses[mask].sum()
        loss = loss_sum / mask.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss is a catastrophic stop gate")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(optimization["gradient_clip_norm"])
        )
        optimizer.step()
        accounting.raw_bytes_consumed += completed_bytes
        accounting.documents_consumed += completed_documents
        accounting.input_tokens_consumed += input_tokens
        accounting.target_tokens_predicted += predicted_tokens
        accounting.optimizer_steps += 1
        accounting.loss_nats_sum += float(loss_sum.detach().cpu())
        accounting.estimated_training_flops = (
            6 * phase_trainable_parameters * accounting.target_tokens_predicted
        )
    accounting.wall_clock_seconds += time.monotonic() - start
    update_memory(accounting, device)


def evaluation_windows(ids: list[int], maximum: int) -> Iterator[list[int]]:
    if len(ids) <= maximum:
        yield ids
        return
    start = 0
    while start < len(ids) - 1:
        window = ids[start : start + maximum]
        if len(window) >= 2:
            yield window
        start += maximum - 1


def evaluate(
    model: torch.nn.Module,
    tokenizer: RuntimeTokenizer,
    documents: list[Document],
    maximum_sequence_length: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_document: list[dict[str, Any]] = []
    domain_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"loss_nats_sum": 0.0, "raw_bytes": 0.0}
    )
    total_loss = 0.0
    total_bytes = 0
    total_tokens = 0
    total_targets = 0
    with torch.no_grad():
        for document in documents:
            ids = tokenizer.encode_document(document)
            document_loss = 0.0
            document_targets = 0
            for window in evaluation_windows(ids, maximum_sequence_length):
                tensor = torch.tensor(window, dtype=torch.long, device=device)[None, :]
                logits = model(tensor)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
                    tensor[:, 1:].reshape(-1),
                    reduction="sum",
                )
                document_loss += float(loss.detach().cpu())
                document_targets += len(window) - 1
            bpb = document_loss / (math.log(2) * document.raw_bytes)
            per_document.append(
                {
                    "text_sha256": document.text_sha256,
                    "domain": document.domain,
                    "raw_bytes": document.raw_bytes,
                    "content_tokens": len(ids) - 1,
                    "target_tokens_predicted": document_targets,
                    "loss_nats_sum": document_loss,
                    "bpb": bpb,
                }
            )
            total_loss += document_loss
            total_bytes += document.raw_bytes
            total_tokens += len(ids) - 1
            total_targets += document_targets
            domain_totals[document.domain]["loss_nats_sum"] += document_loss
            domain_totals[document.domain]["raw_bytes"] += document.raw_bytes
    return {
        "corpus_bpb": total_loss / (math.log(2) * total_bytes),
        "loss_nats_sum": total_loss,
        "raw_bytes": total_bytes,
        "content_tokens": total_tokens,
        "target_tokens_predicted": total_targets,
        "tokens_per_raw_byte": total_tokens / total_bytes,
        "domain_bpb": {
            domain: values["loss_nats_sum"]
            / (math.log(2) * values["raw_bytes"])
            for domain, values in sorted(domain_totals.items())
        },
        "per_document": per_document,
    }


def rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(
    *,
    path: Path,
    run: dict[str, Any],
    phase: str,
    next_boundary_index: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accounting: Accounting,
    token_frequency: Counter[int],
    results: dict[str, Any],
    device: torch.device,
    production_manifest_sha256: str,
) -> None:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run": run,
        "phase": phase,
        "next_boundary_index": next_boundary_index,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "accounting": asdict(accounting),
        "token_frequency": dict(token_frequency),
        "results": results,
        "rng_state": rng_state(device),
        "production_manifest_sha256": production_manifest_sha256,
    }
    torch.save(payload, path)


def active_parameter_count(
    nominal_parameters: int,
    vocab_rows: int,
    dimension: int,
    token_frequency: Counter[int],
) -> int:
    inactive_rows = vocab_rows - len(token_frequency)
    return nominal_parameters - inactive_rows * dimension


def tokenizer_for_arm(
    arm: dict[str, Any], arms_path: Path
) -> RuntimeTokenizer:
    artifact_path = resolve(arms_path.parent, arm["artifact"])
    if sha256_file(artifact_path) != arm["artifact_sha256"]:
        raise AssertionError(f"{arm['id']} tokenizer artifact SHA-256 mismatch")
    return RuntimeTokenizer(arm, artifact_path)


def compare_regenerated_manifest(
    spec_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = load_json(manifest_path)
    regenerated = build_manifest(spec_path)
    if frozen != regenerated:
        raise AssertionError(
            "production manifest differs from current inputs; rerun the freeze "
            "command only before any production result exists"
        )
    return load_json(spec_path), frozen


def preflight(spec_path: Path, manifest_path: Path) -> None:
    spec, manifest = compare_regenerated_manifest(spec_path, manifest_path)
    architecture_path = resolve(
        spec_path.parent, spec["inputs"]["architecture_controls"]
    )
    arms_path = resolve(spec_path.parent, spec["inputs"]["tokenizer_arms"])
    architecture = load_json(architecture_path)
    arms_doc = load_json(arms_path)
    TinyModel = model_module(spec, spec_path).TinyModel

    by_arm = {row["id"]: row for row in arms_doc["arms"]}
    c3_path = resolve(spec_path.parent, spec["inputs"]["c3_jsonl"])
    c3 = ordered_stream(c3_path, manifest["streams"]["evaluation"])
    adversarial = [
        Document("probe", "preflight", text, sha256_bytes(text.encode()), len(text.encode()), index)
        for index, text in enumerate(
            ("", "x", " x", "  x", "\tx", "\n x", "e\u0301", "é", "👩🏽‍💻", "\u200b"),
            1,
        )
    ]
    for arm in arms_doc["arms"]:
        tokenizer = tokenizer_for_arm(arm, arms_path)
        for document in [*adversarial, *c3]:
            tokenizer.encode_document(document)

    checked_shapes: set[tuple[int, int]] = set()
    for run in manifest["runs"]:
        shape = (int(run["vocab_rows"]), int(run["ffn_width"]))
        if shape in checked_shapes:
            continue
        checked_shapes.add(shape)
        with torch.device("meta"):
            model = make_model(TinyModel, run, architecture)
        nominal, trainable = parameter_counts(model)
        if nominal != int(run["parameters"]["nominal_parameters"]):
            raise AssertionError(f"{run['run_id']} nominal parameter mismatch")
        if trainable != int(run["parameters"]["phase1_trainable_parameters"]):
            raise AssertionError(f"{run['run_id']} phase-1 parameter mismatch")
        del model

    if len(manifest["runs"]) != 48 or set(by_arm) != set(manifest["tokenizers"]):
        raise AssertionError("production matrix cardinality mismatch")
    print(
        "TOK-2 production preflight passed: 48 registered cells, 8 tokenizer "
        f"contracts, {len(checked_shapes)} real model shapes, zero optimizer steps"
    )


def run_cell(
    *,
    run_id: str,
    spec_path: Path,
    manifest_path: Path,
    output_root: Path,
    requested_device: str,
    resume_path: Path | None,
) -> None:
    spec, manifest = compare_regenerated_manifest(spec_path, manifest_path)
    matching = [row for row in manifest["runs"] if row["run_id"] == run_id]
    if len(matching) != 1:
        raise ValueError(f"unknown or ambiguous frozen run ID: {run_id}")
    run = matching[0]
    arms_path = resolve(spec_path.parent, spec["inputs"]["tokenizer_arms"])
    architecture_path = resolve(
        spec_path.parent, spec["inputs"]["architecture_controls"]
    )
    arms_doc = load_json(arms_path)
    architecture = load_json(architecture_path)
    arm = next(row for row in arms_doc["arms"] if row["id"] == run["arm_id"])
    tokenizer = tokenizer_for_arm(arm, arms_path)
    TinyModel = model_module(spec, spec_path).TinyModel
    device = select_device(requested_device)
    reset_device_peak(device)

    c8_path = resolve(spec_path.parent, spec["inputs"]["c8_jsonl"])
    c3_path = resolve(spec_path.parent, spec["inputs"]["c3_jsonl"])
    phase_documents = {
        phase: ordered_stream(c8_path, manifest["streams"][phase])
        for phase in ("phase1", "phase3")
    }
    evaluation_documents = ordered_stream(
        c3_path, manifest["streams"]["evaluation"]
    )
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    production_manifest_sha = sha256_file(manifest_path)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run": run,
                "production_manifest_sha256": production_manifest_sha,
                "device": str(device),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    model = make_model(TinyModel, run, architecture)
    initialize_model(model, run)
    initialization_evidence = {
        "shared_trunk_sha256": tensor_set_sha256(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name != "embed.weight"
        ),
        "attention_sha256": tensor_set_sha256(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if ".attn." in name
        ),
        "embedding_output_sha256": tensor_set_sha256(
            [("embed.weight", model.embed.weight)]
        ),
    }
    model = model.to(device=device, dtype=torch.float32)
    nominal, phase1_trainable = parameter_counts(model)
    if nominal != int(run["parameters"]["nominal_parameters"]):
        raise AssertionError("runtime nominal parameter count mismatch")
    if phase1_trainable != int(run["parameters"]["phase1_trainable_parameters"]):
        raise AssertionError("runtime phase-1 trainable parameter count mismatch")

    results: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "running",
        "run": run,
        "production_manifest_sha256": production_manifest_sha,
        "device": str(device),
        "initialization_evidence": initialization_evidence,
        "phase1": {},
        "phase3": {},
    }
    resume: dict[str, Any] | None = None
    if resume_path is not None:
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume["run"]["run_id"] != run_id:
            raise AssertionError("resume checkpoint belongs to a different run")
        if resume["production_manifest_sha256"] != production_manifest_sha:
            raise AssertionError("resume checkpoint uses a different frozen manifest")

    maximum_sequence_length = int(
        architecture["invariants"]["maximum_sequence_length"]
    )
    dimension = int(architecture["invariants"]["d_model"])
    phases = ("phase1", "phase3")
    for phase in phases:
        phase_index = 1 if phase == "phase1" else 3
        peak_lr = float(spec["optimization"][f"{phase}_peak_learning_rate"])
        stochasticity_seed = int(run["seeds"][f"{phase}_stochasticity_seed"])
        seed_stochasticity(stochasticity_seed, device)
        if phase == "phase3":
            configure_phase3(
                model, int(run["seeds"]["phase3_attention_reinitialization_seed"])
            )
            results["phase3_attention_reinitialization_sha256"] = tensor_set_sha256(
                (name, parameter)
                for name, parameter in model.named_parameters()
                if ".attn." in name
            )
        nominal_now, trainable_now = parameter_counts(model)
        expected_trainable = int(
            run["parameters"][f"{phase}_trainable_parameters"]
        )
        if nominal_now != nominal or trainable_now != expected_trainable:
            raise AssertionError(f"{phase} runtime parameter count mismatch")
        optimizer = optimizer_for(model, spec, peak_lr)
        accounting = Accounting()
        token_frequency: Counter[int] = Counter()
        next_boundary_index = 0

        if resume is not None and resume["phase"] == phase:
            model.load_state_dict(resume["model"])
            optimizer.load_state_dict(resume["optimizer"])
            accounting = Accounting(**resume["accounting"])
            token_frequency = Counter(
                {int(key): value for key, value in resume["token_frequency"].items()}
            )
            results = resume["results"]
            next_boundary_index = int(resume["next_boundary_index"])
            restore_rng_state(resume["rng_state"], device)
            resume = None
        elif resume is not None:
            if resume["phase"] == "phase1" and phase == "phase1":
                raise AssertionError("unreachable resume state")
            if resume["phase"] == "phase3" and phase == "phase1":
                model.load_state_dict(resume["model"])
                results = resume["results"]
                continue

        evidence = manifest["streams"][phase]
        boundaries = evidence["checkpoint_boundaries"]
        previous_documents = (
            0
            if next_boundary_index == 0
            else int(boundaries[next_boundary_index - 1]["documents"])
        )
        for boundary_index in range(next_boundary_index, len(boundaries)):
            boundary = boundaries[boundary_index]
            terminal_documents = int(boundary["documents"])
            segment = phase_documents[phase][previous_documents:terminal_documents]
            train_documents(
                model=model,
                tokenizer=tokenizer,
                documents=segment,
                optimizer=optimizer,
                accounting=accounting,
                phase_raw_bytes=int(evidence["actual_raw_bytes"]),
                phase_trainable_parameters=expected_trainable,
                peak_lr=peak_lr,
                spec=spec,
                device=device,
                token_frequency=token_frequency,
            )
            if accounting.raw_bytes_consumed != int(boundary["raw_bytes"]):
                raise AssertionError(f"{phase} checkpoint byte boundary drift")
            if accounting.documents_consumed != terminal_documents:
                raise AssertionError(f"{phase} checkpoint document boundary drift")
            active = active_parameter_count(
                nominal, tokenizer.vocab_size, dimension, token_frequency
            )
            checkpoint_record = {
                "phase": phase_index,
                "fraction": boundary["fraction"],
                "accounting": asdict(accounting),
                "nominal_parameters": nominal,
                "phase_trainable_parameters": expected_trainable,
                "actively_exercised_parameters": active,
                "active_vocabulary_rows": len(token_frequency),
            }
            evaluation_start = time.monotonic()
            checkpoint_record["evaluation"] = evaluate(
                model,
                tokenizer,
                evaluation_documents,
                maximum_sequence_length,
                device,
            )
            checkpoint_record["evaluation_wall_clock_seconds"] = (
                time.monotonic() - evaluation_start
            )
            results[phase].setdefault("checkpoints", []).append(checkpoint_record)
            checkpoint_path = output_dir / (
                f"{phase}-q{round(float(boundary['fraction']) * 100):03d}.pt"
            )
            save_checkpoint(
                path=checkpoint_path,
                run=run,
                phase=phase,
                next_boundary_index=boundary_index + 1,
                model=model,
                optimizer=optimizer,
                accounting=accounting,
                token_frequency=token_frequency,
                results=results,
                device=device,
                production_manifest_sha256=production_manifest_sha,
            )
            previous_documents = terminal_documents
            print(
                f"{run_id} {phase} {boundary['fraction']:.0%}: "
                f"{accounting.raw_bytes_consumed:,} bytes, "
                f"{accounting.target_tokens_predicted:,} targets"
            )
            sys.stdout.flush()

        if accounting.raw_bytes_consumed != int(evidence["actual_raw_bytes"]):
            raise AssertionError(f"{phase} did not reach frozen byte boundary")
        results[phase]["evaluation"] = results[phase]["checkpoints"][-1]["evaluation"]
        results[phase]["token_activity_ledger"] = {
            "active_rows": len(token_frequency),
            "frequency_by_token_id": [
                [token_id, count] for token_id, count in sorted(token_frequency.items())
            ],
        }
        results[phase]["accounting"] = asdict(accounting)
        results[phase]["parameter_counts"] = {
            "nominal_parameters": nominal,
            "phase_trainable_parameters": expected_trainable,
            "actively_exercised_parameters": active_parameter_count(
                nominal, tokenizer.vocab_size, dimension, token_frequency
            ),
        }
        (output_dir / "training_results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )

    results["status"] = "complete"
    results["completed_at_unix_seconds"] = time.time()
    (output_dir / "training_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"complete {run_id}: phase3 held-out "
        f"BPB={results['phase3']['evaluation']['corpus_bpb']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=HERE / "tok2_production_spec.json"
    )
    parser.add_argument(
        "--manifest", type=Path, default=HERE / "tok2_production_manifest.json"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement before expensive training starts",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=HERE / "tok2_production_runs"
    )
    args = parser.parse_args()
    if args.preflight:
        preflight(args.spec.resolve(), args.manifest.resolve())
        return
    if not args.run_id or not args.execute:
        parser.error("training requires both --run-id and --execute")
    run_cell(
        run_id=args.run_id,
        spec_path=args.spec.resolve(),
        manifest_path=args.manifest.resolve(),
        output_root=args.output_root.resolve(),
        requested_device=args.device,
        resume_path=None if args.resume is None else args.resume.resolve(),
    )


if __name__ == "__main__":
    main()
