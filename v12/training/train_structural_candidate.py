#!/usr/bin/env python3
"""Train the six selective U16/B16 pre-tokenization candidates.

Unlike the historical SentencePiece pilot, these candidates use one
serializable Hugging Face `tokenizers` pipeline for both training and runtime.
Whitespace is preserved as model input, and all 256 `<0xNN>` pieces are
ordinary model vocabulary entries, providing exact byte fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, decoders, models, trainers

try:
    from .pretokenization import PRE_TOKENIZATIONS, build_pre_tokenizer, definition
except ImportError:  # direct script execution
    from pretokenization import PRE_TOKENIZATIONS, build_pre_tokenizer, definition


TRAINING_DIR = Path(__file__).resolve().parent
V12_ROOT = TRAINING_DIR.parent
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
BYTE_TOKENS = [f"<0x{value:02X}>" for value in range(256)]


def corpus_iterator(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)["text"]
    else:
        with path.open(encoding="utf-8") as handle:
            yield from handle


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trained_unigram(vocab_size: int, pre_tokenizer, corpus: Path, show_progress: bool) -> Tokenizer:
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.train_from_iterator(
        corpus_iterator(corpus),
        trainers.UnigramTrainer(
            vocab_size=vocab_size,
            show_progress=show_progress,
            special_tokens=SPECIAL_TOKENS + BYTE_TOKENS,
            unk_token="<unk>",
            max_piece_length=32,
        ),
    )
    serialized = json.loads(tokenizer.to_str())
    vocab = [tuple(item) for item in serialized["model"]["vocab"]]
    return Tokenizer(models.Unigram(vocab, unk_id=1, byte_fallback=True))


def _trained_bpe(vocab_size: int, pre_tokenizer, corpus: Path, show_progress: bool) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.train_from_iterator(
        corpus_iterator(corpus),
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            show_progress=show_progress,
            special_tokens=SPECIAL_TOKENS + BYTE_TOKENS,
            max_token_length=32,
        ),
    )
    serialized = json.loads(tokenizer.to_str())
    merges = [
        tuple(merge) if isinstance(merge, list) else tuple(merge.split(" ", 1))
        for merge in serialized["model"]["merges"]
    ]
    return Tokenizer(
        models.BPE(
            vocab=serialized["model"]["vocab"],
            merges=merges,
            unk_token="<unk>",
            byte_fallback=True,
        )
    )


def train(
    algorithm: str,
    vocab_size: int,
    pre_tokenization: str,
    corpus: Path,
    output_dir: Path,
    candidate_id: str,
    show_progress: bool = True,
    require_exact_vocab: bool = False,
) -> dict:
    pre_tokenizer = build_pre_tokenizer(pre_tokenization)
    if algorithm == "unigram":
        tokenizer = _trained_unigram(vocab_size, pre_tokenizer, corpus, show_progress)
    elif algorithm == "bpe":
        tokenizer = _trained_bpe(vocab_size, pre_tokenizer, corpus, show_progress)
    else:
        raise ValueError("algorithm must be unigram or bpe")

    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = decoders.ByteFallback()
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in SPECIAL_TOKENS])
    actual_vocab_size = tokenizer.get_vocab_size()
    if require_exact_vocab and actual_vocab_size != vocab_size:
        raise ValueError(
            f"{candidate_id} requested {vocab_size} pieces but training produced "
            f"{actual_vocab_size}; this is not a valid matched-vocabulary arm"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)

    metadata = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "arm": ("U" if algorithm == "unigram" else "B") + str(vocab_size // 1000),
        "algorithm": algorithm,
        "implementation": "huggingface-tokenizers",
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "model_visible_row_accounting": {
            "total_rows": actual_vocab_size,
            "byte_fallback_rows": len(BYTE_TOKENS),
            "special_control_rows": len(SPECIAL_TOKENS),
            "ordinary_learned_rows": actual_vocab_size - len(BYTE_TOKENS) - len(SPECIAL_TOKENS),
            "extra_rows_added_after_limit": 0,
        },
        "pre_tokenization": definition(pre_tokenization),
        "byte_fallback": True,
        "special_token_ids": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "corpus_path": str(corpus.resolve()),
        "corpus_sha256": file_sha256(corpus),
        "tokenizer_json": str(tokenizer_path.resolve()),
        "tokenizer_json_sha256": file_sha256(tokenizer_path),
    }
    (output_dir / "candidate.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=["unigram", "bpe"])
    parser.add_argument("--vocab-size", type=int, default=16000, choices=[16000])
    parser.add_argument("--pre-tokenization", required=True, choices=PRE_TOKENIZATIONS)
    parser.add_argument("--corpus", type=Path, default=V12_ROOT / "corpus" / "c8_corpus_v3.jsonl")
    parser.add_argument("--candidate-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    family = "U16" if args.algorithm == "unigram" else "B16"
    candidate_id = args.candidate_id or f"{family}_{args.pre_tokenization}_c8v3"
    output_dir = args.output_dir or TRAINING_DIR / "candidates" / candidate_id
    metadata = train(
        args.algorithm,
        args.vocab_size,
        args.pre_tokenization,
        args.corpus,
        output_dir,
        candidate_id,
        show_progress=not args.no_progress,
        require_exact_vocab=True,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
