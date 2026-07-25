# TOK-2 decisive comparison protocol

Status: pre-run protocol, written 2026-07-25 and tightened after review on the
same date. No result is implied by this document.

## Entry conditions

1. C8 v3 is marked `frozen`, every source is revision- and SHA-256-pinned,
   C3 exclusions are applied, and the build is reproducible.
2. Every planned corpus scale uses the same byte shares:
   45% natural prose, 20% code, 15% maths/reasoning, 15% JSON/tool/cell
   syntax, and 5% noisy Unicode/mixed text.
3. Every arm passes the whole-input contract in `STREAMING_CONTRACT.md` and
   Rust/Python/Hugging Face differential round-trip and offset tests before
   model training. Arbitrary stateless chunks are prohibited.
4. Raw-byte streams, evaluation sets, optimizer settings, checkpoint steps,
   and the statistical analysis script are frozen before the first run.
5. The complete architecture/accounting table and all FFN widths are frozen
   before the first run.

## Primary arms

| Arm | Model-visible rows | Paired run labels | Training |
|---|---:|---:|---|
| v11 whitespace-exact revision/adapter | 71,260 | 0, 1, 2 | phase 1 + phase 3 |
| U16 × whitespace_split | exactly 16,000 | 0, 1, 2 | phase 1 + phase 3 |
| U16 × digit_isolating | exactly 16,000 | 0, 1, 2 | phase 1 + phase 3 |
| U16 × code_aware | exactly 16,000 | 0, 1, 2 | phase 1 + phase 3 |
| B16 × whitespace_split | exactly 16,000 | 0, 1, 2 | phase 1 + phase 3 |
| B16 × digit_isolating | exactly 16,000 | 0, 1, 2 | phase 1 + phase 3 |
| B16 × code_aware | exactly 16,000 | 0, 1, 2 | phase 1 + phase 3 |
| byte | 260 | 0, 1, 2 | phase 1 + phase 3 |

The v11 arm is exactly `training/v11_ws_exact_manifest.json`, not the older
71,261-row native SentencePiece model.

All arms consume identical raw bytes in identical order within a seed. Model
initialization and stochastic training state vary by seed; corpus order does
not vary between tokenizer arms.

The six structural configurations are full protocol arms, not a
tokenizer-only ranking stage. Their structural profiles remain diagnostic, so
none may be selected or discarded using those profiles before model results.
The U16/B16 rows in the architecture table apply identically to all three
pre-tokenization variants in their family.

### Paired randomness

Run labels 0, 1, and 2 are paired across all arms. The frozen run manifest
must assign one numeric seed per label for each independent RNG stream:

- shared transformer-trunk initialization;
- vocabulary-shaped embedding/output initialization;
- dropout and other model stochasticity;
- data/sample selection;
- evaluation ordering.

The trunk is initialized independently of vocabulary-shaped tensors so that
the same run label produces byte-identical shared trunk tensors in every
fixed-trunk arm. Raw document order is identical across tokenizer arms, not
merely generated from a nominally equal seed.

Report the three raw paired differences before any summary:

```text
candidate run 0 BPB - v11 run 0 BPB
candidate run 1 BPB - v11 run 1 BPB
candidate run 2 BPB - v11 run 2 BPB
```

Three training runs do not support elaborate distributional claims. A
document bootstrap within each checkpoint characterizes evaluation-set
uncertainty, not training-run uncertainty, and must be labelled accordingly.

### Vocabulary-row definition

“16K” means exactly **16,000 embedding/output rows visible to the model**.
That count includes:

- all 256 byte-fallback pieces;
- PAD, UNK, BOS, and EOS;
- every added special or control token;
- every ordinary learned piece.

No token may be appended after enforcing the 16,000-row limit. Candidate
metadata must report the count from the runtime artifact actually loaded by
the model.

## Parameter controls

Results are reported as two separate experiments.

### Fixed trunk

Transformer depth, width, heads, and FFN dimensions are identical. Embedding
and output-head sizes vary with vocabulary, so total parameter counts differ.
This estimates tokenizer quality for a fixed representation trunk.

### Fixed total parameters

Complete models are matched to 115M nominal parameters within 0.25%. Compact-vocabulary
arms reinvest the saved embedding/output parameters **only by changing FFN
intermediate width**. Model width, layer count, attention-head count, head
dimension, positional representation, and every other architectural choice
remain fixed. Choose the nearest hardware-valid FFN width and match complete
nominal parameters to 115M within **0.25%**; if that is impossible, record the
nearest two widths and pin one before training rather than changing another
dimension.

The architecture-matching table must record depth, model width, head count,
head dimension, FFN width, embedding/output parameters, trunk parameters, and
all three parameter counts:

1. **Nominal parameters** — every stored scalar in the complete model.
2. **Trainable parameters** — parameters with gradient updates enabled in the
   current phase.
3. **Actively exercised parameters** — rows/parameters reached by the frozen
   training stream in that phase, using the token-utility ledger's observed
   row activity rather than an estimate.

Dormant or protected v11 rows therefore count toward nominal storage, may
count toward trainable capacity, and count as active only when the training
stream actually exercises them.

The exact frozen table is `training/tok2_architecture_controls.json`.
For the current tied-embedding TinyModel shape it pins FFN widths 2048 for
v11, 2968 for U16/B16, and 3232 for pure byte under fixed total. The compact
arms differ from v11's 115,149,312 nominal parameters by -0.027% and +0.018%,
respectively. `training/validate_tok2_controls.py` recomputes every count from
the architecture formula.

Fixed-trunk and fixed-total results must never be pooled into one ranking.

## Data exposure, schedule, and accounting

The authoritative progress coordinate is cumulative raw UTF-8 bytes consumed.
All arms see the same complete documents in the same order and stop at the
same byte boundary. The learning-rate schedule, phase boundary, checkpoint
schedule, and evaluation schedule are parameterized by cumulative raw bytes,
not tokenizer tokens or optimizer steps.

Every checkpoint records:

```text
raw bytes consumed
documents consumed
input tokens consumed
target tokens predicted
optimizer steps
estimated training FLOPs
wall-clock time
peak device and host memory
```

The primary comparison is at equal raw-byte exposure. Equal estimated-FLOP
and equal wall-clock interpolations are additional resource views; they do not
replace the primary endpoint.

## Outcomes and statistics

### Primary

Held-out bits per raw UTF-8 byte after phase 3 at equal raw-byte exposure,
averaged across the three paired runs, under the **fixed-total-parameter**
control. This is the system-quality decision endpoint.

Report every checkpoint, the three paired differences, their mean and standard
deviation, and a plainly caveated interval over three training runs. Also
report paired per-document BPB differences with a document-bootstrap
confidence interval, explicitly labelled as evaluation-set uncertainty.

### Secondary

- phase-3 BPB under the fixed-trunk control (the tokenizer-quality mechanism);
- phase-1 BPB under both controls;
- training FLOPs, wall-clock time, and peak memory;
- tokens per raw byte;
- BPB at equal estimated FLOPs and equal wall-clock budget;
- BPB by C8 v3 domain and on a frozen non-TinyStories external evaluation set.

### Diagnostic only

- AST leaf-boundary alignment and code compilation/parsing;
- identifier fragmentation;
- JSON key/value and Cell80 frame boundaries;
- arithmetic digit boundaries;
- natural-language morphological alignment;
- token utilization and token-utility ledger profiles.

Structural metrics explain model results in this protocol. They are not a
post-hoc override, hard gate, weighted score, or tie-breaker. A future
preregistered protocol may promote a diagnostic metric only after its
relationship with model performance is understood. No tokenizer wins solely
on aggregate TinyStories BPB.

## Selective pre-tokenization screen

Run only these six 16K configurations before considering the historical
48-cell grid:

```text
U16 × whitespace_split
U16 × digit_isolating
U16 × code_aware
B16 × whitespace_split
B16 × digit_isolating
B16 × code_aware
```

Use `training/train_structural_candidate.py`; its runtime artifact is the same
serializable `tokenizer.json` used during training. The pre-run gate may reject only
correctness failures (wrong row count, UNK, round-trip, offset, or
training/runtime parity). Compression and the structural profile are reported
for explanation; model training decides among correctness-valid arms.

## No phase-one elimination

Every protocol arm that passes the frozen pre-run correctness gates continues
through phase 3. Phase-1 model quality, speed, or memory ranking cannot
eliminate an arm. A run may stop only for an operational catastrophe that
prevents a valid checkpoint—non-finite loss, corrupt data/artifact, hardware
failure, or invariant violation—and must be rerun from the same frozen
configuration rather than replaced with a different arm.

Phase one is diagnostic. Phase three decides.

## Decision rule

1. Use the fixed-total primary endpoint for the system decision and the
   fixed-trunk secondary endpoint to interpret the tokenizer mechanism; never
   pool their rankings.
2. Show all three paired run differences. Treat inconsistent signs or an
   interval crossing zero as unresolved, without implying that three runs
   precisely characterize training variance.
3. Start a broader v11.1 only if the knowledge-rich branch remains competitive enough
   to justify a new 71K-vocabulary model.
4. Start v13 only if U16 remains the compact ancestor after phase 3 and the
   short-train ranking is shown to predict the full-run ranking.
5. Treat compact-output/knowledge-overlay as a separate architecture line,
   not a v12 tokenizer mutation.
