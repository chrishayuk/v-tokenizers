#!/usr/bin/env python3
"""Trains one real TOK-1 grid candidate on the v0 C8 corpus.

The first genuinely-trained TOK-1 candidate tokenizer in this funnel --
real SentencePiece training, not simulated. Emits both the native .model
and a vocab.json in the same {"pieces":[...], "special": {...}} shape
tokenizer_bench.py already reads, so census/intrinsics run against it
unmodified.

Run: python3 v12/training/train_candidate.py --algorithm unigram --vocab-size 4000
"""
import argparse
import json
from pathlib import Path

import sentencepiece as spm

TRAINING_DIR = Path(__file__).resolve().parent
V12_ROOT = TRAINING_DIR.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=["unigram", "bpe"], default="unigram")
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--candidate-id", default=None)
    ap.add_argument("--corpus-version", default="v1", help="which corpus/c8_corpus_<version>.txt to train on")
    ap.add_argument("--seed-t-core", action="store_true",
                     help="seed T-core (targets/t_core.jsonl) as SentencePiece user_defined_symbols, "
                          "guaranteeing each is a single vocab piece -- tests whether v11's real "
                          "fertility advantage comes from explicit priority-token injection")
    ap.add_argument("--byte-fallback", action="store_true",
                     help="enable SentencePiece's byte_fallback: any character not covered by a "
                          "learned piece encodes via <0xNN> byte pieces instead of <unk>. Tests "
                          "whether this fixes the round-trip/UNK hard-reject shared by every "
                          "unigram_sp/bpe_sp candidate (no piece for literal tab/newline).")
    args = ap.parse_args()
    corpus_txt = V12_ROOT / "corpus" / f"c8_corpus_{args.corpus_version}.txt"

    candidate_id = args.candidate_id or f"{args.algorithm}_sp_{args.vocab_size}_{args.corpus_version}"
    if args.seed_t_core:
        candidate_id += "_tcoreseed"
    if args.byte_fallback:
        candidate_id += "_bytefallback"
    out_dir = TRAINING_DIR / "candidates" / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(out_dir / candidate_id)

    train_kwargs = dict(
        input=str(corpus_txt),
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.algorithm,
        character_coverage=0.9995,
        split_digits=True,  # C6: single-char digits
        byte_fallback=args.byte_fallback,
        # identity, not the sentencepiece-library default nmt_nfkc: matches
        # the RESOLVED_2026_07_19_canonical_tokenizer_decision (tokenizer.json's
        # non-NFKC, non-whitespace-collapsing behavior is canonical for v11 --
        # there's no principled reason for freshly-trained v12 candidates to
        # use different default normalization).
        normalization_rule_name="identity",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        input_sentence_size=0,  # use all lines
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1000000,
    )
    if args.seed_t_core:
        # Seed BOTH the bare and metaspace-prefixed ("▁tok") forms --
        # matching v11-builder's own real technique (src/main.rs:
        # `format!("\u{2581}{tok}")`) for injecting priority tokens.
        # First attempt seeded only the bare form and measured
        # t_core_fertility=2.0 exactly, not ~1.0: SentencePiece's
        # add_dummy_prefix normalization prepends "▁" before matching, so
        # encoding a standalone word emits ['▁', 'tok'] (2 tokens) unless
        # the prefixed form is ALSO a registered piece.
        t_core_path = V12_ROOT / "targets" / "t_core.jsonl"
        bare = sorted({json.loads(l)["text"] for l in t_core_path.read_text().splitlines() if l.strip()})
        symbols = sorted(set(bare) | {f"▁{s}" for s in bare})
        train_kwargs["user_defined_symbols"] = symbols

    spm.SentencePieceTrainer.train(**train_kwargs)

    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    pieces = [{"id": i, "text": sp.id_to_piece(i), "score": sp.get_score(i)} for i in range(sp.vocab_size())]
    vocab_json = {
        "version": 1,
        "special": {"pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3},
        "pieces": pieces,
    }
    vocab_path = out_dir / f"{candidate_id}.vocab.json"
    vocab_path.write_text(json.dumps(vocab_json))

    print(json.dumps({
        "candidate_id": candidate_id,
        "algorithm": args.algorithm,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": sp.vocab_size(),
        "model_path": str(model_prefix + ".model"),
        "vocab_json_path": str(vocab_path),
    }, indent=2))


if __name__ == "__main__":
    main()
