# TOK-2 production harness

Status: frozen, validated, and not started.

The production harness turns the decisive protocol into 48 immutable cells:
eight tokenizer arms, two parameter controls, and three paired run labels.
It consumes complete C8 v3 documents on a raw-byte schedule, evaluates the
frozen C3 slice at each quarter-byte checkpoint, runs phase 1 and phase 3 for
every correctness-valid arm, and records the accounting required by the
protocol.

## Frozen evidence

- `tok2_production_spec.json` pins byte budgets, optimizer and LR rules,
  checkpoints, evaluation, phase-3 transition, accounting, and statistics.
- `tok2_production_manifest.json` pins all 48 run IDs, architecture shapes,
  seed streams, tokenizer hashes, corpus hashes, ordered stream hashes, and
  exact complete-document checkpoint boundaries.
- `tok2_tokenizer_archive_manifest.json` records the six generated runtime
  tokenizers in the experiment registry. Their registry hashes match the
  local frozen hashes, and all six report producing run
  `RUN-20260718-224711-00385`.
- `train_tok2_production.py` refuses expensive work unless both a frozen run
  ID and `--execute` are supplied.
- `analyze_tok2_production.py` implements the paired three-run differences,
  caveated 90% Student-t interval, primary paired-document bootstrap, and
  equal-FLOP/equal-training-wall-clock interpolation.

The historical raw-byte exposure is retained: the nominal phase-1 budget is
53,727,334 bytes and the nominal phase-3 budget is 26,863,667 bytes. Complete
document boundaries make the actual frozen exposures 53,727,169 and
26,861,261 bytes respectively. Those boundaries are identical for every arm.

## Safe validation

From the repository root:

```bash
python3 v12/training/freeze_tok2_production.py --check
python3 v12/training/train_tok2_production.py --preflight
python3 -m unittest discover -s v12/training -p 'test_tok2_production.py'
```

Preflight loads every tokenizer, exercises adversarial whitespace and Unicode
plus all 200 C3 documents, constructs every distinct TinyModel shape on the
PyTorch meta device, checks parameter counts, and takes zero optimizer steps.

## Launching one frozen cell

The worker must have this repository and `tiny-model` checked out as sibling
directories. The latter must be at commit
`6cc4868fc0cd64f648844419532959d19c5e12ef`, and the model source itself is
hash-checked before use.

Generated inputs are intentionally gitignored. Materialize C8 v3 and C3 using
the frozen procedure in `../corpus/C8_V3_FREEZE.md`, and restore the six U16/B16
tokenizers from the registry URIs in `tok2_tokenizer_archive_manifest.json`.
The runner rejects any restored or rebuilt file whose hash differs. The v11
and pure-byte artifacts are tracked directly.

```bash
python3 v12/training/train_tok2_production.py \
  --run-id tok2-fixed_total-U16_code_aware-seed0 \
  --device cuda \
  --execute
```

Checkpoints and results are written below the ignored
`v12/training/tok2_production_runs/` directory. A resume must name the same
frozen run and manifest:

```bash
python3 v12/training/train_tok2_production.py \
  --run-id tok2-fixed_total-U16_code_aware-seed0 \
  --device cuda \
  --resume v12/training/tok2_production_runs/tok2-fixed_total-U16_code_aware-seed0/phase1-q050.pt \
  --execute
```

No phase-1 quality result eliminates an arm. Operational catastrophes are
rerun from the same manifest.

## Final analysis

After all 48 `training_results.json` files are complete:

```bash
python3 v12/training/analyze_tok2_production.py
```

Until then, `--allow-incomplete` reports the exact missing run IDs and does
not emit a partial ranking.
