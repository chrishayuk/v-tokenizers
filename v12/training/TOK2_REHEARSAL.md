# TOK-2 decisive-matrix rehearsal

Status: passed 2026-07-25. This is an orchestration canary, not a model result.

The rehearsal traversed every declared cell:

| Dimension | Count |
|---|---:|
| Tokenizer arms | 8 |
| Parameter controls | 2 |
| Paired run labels | 3 |
| Total cells | 48 |
| Phase-1 eliminations | 0 |
| Phase-3 completions | 48 |
| Held-out evaluations | 48 |

The eight runtime artifacts are pinned by
`tok2_tokenizer_arms.json`: the newer 71,260-row v11 whitespace-exact arm,
six exact-16,000-row U16/B16 structural tokenizers trained on frozen C8 v3,
and the 260-row pure-byte arm.

An independent retraining audit found mixed builder reproducibility. All three
B16 artifacts rebuilt byte-for-byte. The `tokenizers` 0.22.2 Unigram trainer
exposes no RNG seed, and all three U16 rebuilds differed from their frozen
artifacts while still producing exactly 16,000 runtime rows. Therefore the
U16 runtime hashes are authoritative and the ignored artifacts must be
archived before production runs. See `tok2_tokenizer_retrain_audit.json`.

Every cell tokenized identical complete documents in identical order. Phase 1
used five C8 documents and 1,833 raw bytes, phase 3 used a disjoint five
documents and 1,961 raw bytes, and evaluation used five frozen C3 documents
and 5,668 raw bytes. Every stream contained one document from each C8 domain.
The stream SHA-256 values are frozen in `tok2_rehearsal_results.json`.

## What was exercised

- tokenizer artifact hashes and runtime model-visible row counts;
- whole-input round trip and absence of UNK on every rehearsal document;
- phase 1, phase 3 FFN freezing, attention reinitialization, and evaluation;
- paired, independently seeded trunk and vocabulary-shaped initialization;
- byte-identical fixed-trunk tensors across tokenizer arms;
- byte-identical attention initialization and phase-3 attention
  reinitialization across relevant paired cells;
- raw bytes, documents, input tokens, predicted tokens, optimizer steps,
  estimated production FLOPs, wall time, memory, and vocabulary-row activity;
- the no-phase-one-elimination rule.

The rehearsal also instantiated the pinned TinyModel implementation on
PyTorch's meta device. Actual module counts matched every distinct production
configuration, including:

| Configuration | Nominal parameters | Phase-3 trainable |
|---|---:|---:|
| v11, FFN 2048 | 115,149,312 | 52,224,512 |
| 16K, FFN 2048 | 86,856,192 | 23,931,392 |
| byte, FFN 2048 | 78,797,312 | 15,872,512 |
| 16K, FFN 2968 | 115,118,592 | 23,931,392 |
| byte, FFN 3232 | 115,169,792 | 15,872,512 |

## Deliberate limits

The canary model has dimension 16 and one layer. Its FFN widths preserve the
control routing but do not match 115M total parameters. It makes only one
pass over a tiny stream. Consequently its loss, BPB, FLOP estimate, memory,
and timing fields cannot rank tokenizers, estimate final quality, or justify
eliminating an arm. The results file labels every cell accordingly.

The rehearsal proves that the frozen data, tokenizers, paired RNG streams,
architecture controls, phase transition, evaluation path, and accounting
schema can run together before expensive training begins.

## Reproduction

With the ignored C8 and C3 text artifacts materialized, prepare or verify all
tokenizer arms:

```bash
python3 v12/training/prepare_tok2_tokenizer_arms.py
```

To audit retraining without overwriting any frozen runtime artifact:

```bash
python3 v12/training/prepare_tok2_tokenizer_arms.py \
  --audit-retrain-root /tmp/tok2-retrain-audit
```

Run the full rehearsal matrix:

```bash
python3 v12/training/rehearse_tok2_matrix.py
```

The tracked evidence is `tok2_rehearsal_results.json`. The next gate is
archiving the six structural runtime artifacts, then freezing the real
byte-scheduled training/checkpoint/evaluation harness before the 48 expensive
phase-1 and phase-3 runs.
