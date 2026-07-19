# v12 Tokenizer — TOK-0/TOK-1 Harness + First Real Candidates

Implements the TOK-0 deliverables from `v12-tokenizer-design-funnel.md`
(DRAFT v0.5): the harness, the pin file, target sets, the candidate
grid, and the dormant-block map schema — plus, as of 2026-07-19, a real
(prototype-scale, hardened once) C8 corpus and genuinely-trained TOK-1
candidates.

**What's actually been run vs. what's designed**: `candidate_grid.yaml`
specifies a 4-algorithm × 4-vocab-size × 3-pre_tokenization grid (48
subword configs) plus the pure-byte branch. What exists today is an
**algorithm × vocab-size pilot with a byte baseline** — the
`pre_tokenization` axis (`whitespace_split`/`digit_isolating`/
`code_aware`) is entirely unimplemented; every candidate trained so far
uses one implicit pretokenization. Real progress, not the full design.

Tracked in the `chuk-experiments` server under programme `v12-tokenizer`,
experiments `tok-0-harness-pinning` through `tok-5-freeze`.

Lives in the standalone `v-tokenizers` repo (moved 2026-07-19 from
`tiny-model/tokenizer/v12/`, fresh git history) alongside `v11/` — see
the repo-root README's Status section before assuming v11 is
production-ready; it currently is not (not byte-safe). v12 is NOT
published — it's an active, pre-registered research funnel and stays
private to this repo until a candidate wins Gate G1/G2/G3 and is
promoted.

## Layout

```
v-tokenizers/
  bench/                      COMMON harness, shared across v11 and v12 —
                               it already spans both (drives v11's compiled
                               binary for the real roundtrip check AND
                               implements v12's census/intrinsics/grid-screen)
    tokenizer_bench.py        CLI: census, intrinsics, g0-smoke, roundtrip, grid-screen
    g1_selection.py           Gate G1 selection algorithm (Pareto frontier + rejects + tie-break)
    test_g1_selection.py      unit test against synthetic fixtures, 11/11 pass
    msi/
      canonicalizer.py             MSI class-1/2/3 transducer (Python)
      parity_vectors.jsonl         golden vectors shared by both languages
      test_canonicalizer_parity.py Python side of the parity check
      frame_battery.jsonl          T-num probes x 4 framing contexts
    reports/                  run output lands here (gitignored)

  v11/                        stable, publishable — see repo-root README

  v12/                        this directory
    pins/tok0_pins.yaml       frozen commitments C0-C10 + open numeric PINs
    candidate_grid.yaml       TOK-1 4x4x3 grid + ABL-1 + Branch B + shadow lane
    dormant_block_map.yaml    C7 immutable vocab-block schema
    targets/                  t_core / t_cell / t_num / t_scale seed sets

    corpus/                   C8 corpus assembly (scripts + manifests tracked;
                               the bulk .jsonl/.txt dumps are gitignored --
                               regenerate via the scripts below, or pull the
                               real artifact from chuk-experiments)
      build_code_corpus.py       harvests this repo's own source into the C8 code domain
      c8_code_corpus_manifest.json    sha-pinned per file
      assemble_c8_corpus.py      streams TinyStories + samples cn7/cn8 math + loads code
      c8_manifest.json           the real C8 manifest: domains, byte counts, mixture, seed
      c8_code_corpus.jsonl / c8_corpus_v0.jsonl / .txt   (gitignored, regenerable)

    training/                 candidate training + real evaluation (scripts
                               tracked; trained candidates/ and candidates.jsonl
                               are gitignored -- they're registered as real
                               artifacts in chuk-experiments instead)
      train_candidate.py         trains a real SentencePiece candidate on ../corpus/c8_corpus_v0.txt
      train_byte_level_bpe.py    trains a real byte-level BPE candidate (tokenizers library)
      build_pure_byte_vocab.py   builds the deterministic 260-token Branch B vocab
      evaluate_candidate.py      evaluates every candidate through its own real library
      candidates/<id>/           (gitignored) trained .model + vocab.json per candidate
      candidates.jsonl           (gitignored) real per-candidate evaluation rows

    msi/                      Rust side of the MSI canonicalizer, workspace member
      Cargo.toml               (package name v12-msi)
      src/lib.rs               canonicalize_identity + #[test] against the same
                                parity_vectors.jsonl the Python side uses
```

A `core/` crate at the repo root is reserved for logic shared between
`v11/` and `v12/` (vocab loading, trie/error types, special-token
handling) — not extracted yet, since v12 doesn't have a Rust-side
tokenizer implementation of its own to de-duplicate against (its
candidates are currently Python-only, via `sentencepiece`/`tokenizers`).
Worth doing once a v12 candidate wins the funnel and gets a Rust port.

## Status

- `bench/tokenizer_bench.py` — working CLI: `census`, `intrinsics`, `g0-smoke`.
  Runs today against the real v11 artifacts (`../v11/artifacts/v11.vocab.json`)
  and the real (small) sample corpus (`../v11/corpus/**`). This is a
  **smoke test of the harness code**, not Gate G0 itself — G0 requires
  the pinned 24M-token C2 atlas stream, which lives in the corpus-atlas
  programme and is not present in this repo yet.
- **MSI canonicalizer class-1 parity: done, both languages, verified.**
  `bench/msi/canonicalizer.py::canonicalize_identity` and the `v12-msi`
  Rust crate's `canonicalize_identity` both run against the same 9 golden
  vectors in `bench/msi/parity_vectors.jsonl` and agree line-for-line —
  `python3 bench/msi/test_canonicalizer_parity.py` and
  `cargo test -p v12-msi` both pass. The class-2 collapsing-merge case is
  still a stub on both sides — it needs a real TOK-1 candidate's merge
  table, which doesn't exist yet.
- **Round-trip + UNK gate: run for real against the compiled v11 binary — found a genuine
  incumbent defect, root-caused, and FIXED.** `bench/tokenizer_bench.py roundtrip` shells
  out to `target/release/v11` (build first: `cargo build --release -p
  v11-cli` from the repo root) and checks exact round-trip on all 32 files in `../v11/corpus`. Original
  result: UNK gate passed (0 UNK anywhere), but round-trip failed on 32/32 files. Root
  cause: `v11-core`'s `pretokenize_chunks()`/`metaspace()` treated any ASCII whitespace as
  a chunk delimiter, but real HF `Metaspace` splits only on the literal space character
  (0x20) — verified directly against the `tokenizers` library. **Patched** (both functions
  now split on 0x20 only; 18/18 v11-core tests pass, 4 new regression tests added) and
  **verified**: the compiled Rust binary now matches real HF `tokenizers` token-for-token
  on the file that originally exposed the gap (115 tokens, 1 UNK, identical). Disclosed
  side effect, not hidden: this correctly reveals a separate, pre-existing vocab coverage
  gap — v11's vocab has no piece for a literal tab/newline, so `unk_gate_pass` now
  correctly fails (662 UNK across the 32-file sample; code files hit hardest via
  indentation) instead of masking it. Round-trip still fails 32/32 files, now for the
  honest reason. Full detail in `pins/tok0_pins.yaml` `incumbent_ledger` and the
  `tok-0-harness-pinning` writeup (chuk-experiments).
- **Bigger, unresolved finding surfaced while verifying the fix: a real 3-way tokenizer
  divergence.** Native SentencePiece loading the actual, sha-verified `v11.model` (the file
  `corpus-atlas`'s pipeline used to build the already-frozen C2 eval stream) disagrees with
  both the old and the now-patched Rust/`tokenizer.json` pair — 99 tokens/0 UNK vs 115/1 on
  the same file. Root cause: `v11.model`'s own `normalizer_spec` (`nmt_nfkc`,
  `remove_extra_whitespaces: True`) collapses all whitespace before tokenizing;
  `tokenizer.json` has `normalizer: null` — that step was dropped at export time, a
  distinct and earlier bug than the one just fixed. This means the pinned C2 stream was
  built with a tokenizer that disagrees with the artifacts this repo's harness and Rust
  runtime actually use — real tension with commitment C5. **Not resolved here** —
  reconciling it means deciding which artifact is canonical, which is Chris's call, not an
  engineering decision to make solo mid-session. See `pins/tok0_pins.yaml`
  `incumbent_ledger.MAJOR_2026_07_19_three_way_divergence`.
- **Gate G1 selection algorithm: implemented and unit-tested, not just scaffolded.**
  `bench/g1_selection.py::select_survivors` runs the doc's own funnel verbatim
  (hard rejects → threshold rejects → Pareto frontier over the four pinned axes →
  ≤1 survivor per family via tie-break). `bench/test_g1_selection.py` proves it
  correct against 12 hand-designed synthetic candidates (11/11 checks pass) —
  dominance, tie-break, both threshold axes, both hard-reject paths, incumbent/
  Branch-B exemption, shadow passthrough. Wired into `tokenizer_bench.py
  grid-screen`, which correctly refuses to run against this repo's real,
  still-undecided pins rather than inventing defaults. What's blocked is the
  *input* (real TOK-1 candidates need the C8 corpus + training tooling), not
  the algorithm.
- **C2/C3 corpora: found, real, referenced in place (not copied).** A
  sibling repo, `cell80/experiments/corpus-atlas`, already has the exact
  24M-token atlas eval stream the doc describes (16M seed42 + 8M seed43,
  MAX_SEQ 256, tokenized with the real v11 model, sha-verified) and a
  held-out slice (`div0_heldout.json`) with a genuine prefreeze-exposure
  appendix. Full provenance recorded in `pins/tok0_pins.yaml`. The
  held-out slice's own rule says it's scored once and not to be
  reselected — treated as a reference pattern for v12, not reused outright.
- **C8 corpus: real v0 mixture assembled, not just planned.** Prose
  streams live from `roneneldan/TinyStories` (same pinned hub revision as
  C2). Math+structured is sampled from real cn7/cn8 cell-calling corpora
  in `cell80/experiments/cell-native-architectures`, referenced in place.
  Code didn't exist anywhere on this machine, so `build_code_corpus.py`
  harvested this repo's own Rust/Python/etc. source (95 files, 7
  languages, 299KB). `assemble_c8_corpus.py` combines all three into
  5,595 rows / 1.9MB — see `c8_manifest.json` for exact composition,
  mixture proportions (**provisional v0**, not the frozen commitment),
  and what's not yet implemented (dedup, C3-exclusion, repeated-identifier
  cap).
- **First real TOK-1 candidate: trained, not simulated.**
  `train_candidate.py` ran actual `sentencepiece` unigram training
  (vocab 4000, `split_digits=True` per C6) on the v0 C8 corpus —
  `candidates/unigram_sp_4000_v0/`. Evaluated with `census`/`intrinsics`
  against the same sample corpus used for v11's g0-smoke: five-way census
  `{1:1, 2:3, 3:0, 4:0, 5:3206}`, compression 0.328 tok/byte, fertility
  17.09 pieces/word (genuinely mediocre — a tiny prose-heavy vocab
  evaluated against a code-heavy sample; reported as measured, not tuned).
  Along the way, found and fixed a real harness bug: `intrinsics` was
  counting unmapped/failed matches toward `total_tokens`, silently
  inflating compression/fertility on any candidate with nonzero unmapped
  chars.
- **Grid expanded to 9 real candidates + the v11 incumbent, all evaluated
  through real libraries.** `train_byte_level_bpe.py` (via the
  `tokenizers` library's `ByteLevelBPETokenizer`) and
  `build_pure_byte_vocab.py` (deterministic 260-token byte map, no
  training) added the `byte_level_bpe` and `pure_byte` (Branch B) families
  alongside `unigram_sp`/`bpe_sp`. `evaluate_candidate.py` evaluates all
  10 rows (each through its own real backend — sentencepiece / tokenizers
  / a direct byte map — not the harness's approximate simulator) into
  `candidates.jsonl`, in exactly the schema `g1_selection.select_survivors`
  consumes. Three real findings, all disclosed:
  - SentencePiece hits hard vocab ceilings on the 1.9MB v0 C8 corpus —
    unigram ≤6923, bpe ≤20743 — a corpus-scale artifact, not a library
    bug; trained near-ceiling candidates instead of the originally
    planned sizes.
  - `pure_byte_v0` measures MSI-strict = 1.0 exactly across all 116
    `frame_battery.jsonl` instances — an empirical confirmation of the
    design doc's pre-registered prediction P10'(b).
  - Every `unigram_sp`/`bpe_sp` candidate shows `round_trip_pass=false,
    unk_count=144` on the real v11 corpus sample; `byte_level_bpe` and
    `pure_byte` both show `round_trip_pass=true, unk_count=0` — the
    compression-vs-coverage tradeoff Gate G1's Pareto frontier exists to
    surface.
  - Also found, while building the incumbent's evaluation path: a genuine
    Python/Rust parity gap distinct from the whitespace bug above (114 vs
    115 tokens, 0 vs 1 UNK on the same file via two v11 loaders) — see
    `pins/tok0_pins.yaml` `incumbent_ledger.newly_discovered_2026_07_19_b`.
    `candidates.jsonl`'s `v11_incumbent` row carries a `CAVEAT_parity_gap`
    field rather than being presented as authoritative.
- **Gate G1 threshold pins decided, and the first real selection run
  completed (2026-07-19).** Per Chris's explicit instruction,
  `census_F_max=2.0`, `census_R_max=0.95`, `MSI_canonical_min=0.25` are
  now set — full independently-motivated reasoning, plus the honest
  caveat that this was decided *after* the candidate grid already
  existed (not blind pre-registration), in `pins/tok0_pins.yaml` →
  `g1_threshold_decision_log_2026_07_19`. First run gave
  `survivors = [byte_level_bpe_8000_v0]` — but that was against a stub
  T-core and a 1.9MB prototype corpus. See the hardening pass below: it
  doesn't hold up.
- **Hardened, same day: real T-core, 11x bigger corpus, zero survivors —
  and here's exactly why.** Replaced the 31-row stub T-core with 538
  real items from `v11/config.json` (`targets/build_t_core.py`:
  language keywords, math/greek symbols, technical acronyms — the same
  data v11's own vocab injects for guaranteed single-token coverage).
  Scaled C8 11.4x (1.9MB → 21.6MB) and retrained the grid at
  `candidate_grid.yaml`'s real vocab sizes (bpe_sp/byte_level_bpe reach
  the full [4000,8000,16000,32000]; unigram_sp tops out at 13000 —
  its SentencePiece ceiling only moved from 6923 to 13331 despite the
  11.4x corpus increase, a real non-linear-scaling finding). **Real
  caveat, not hidden**: scaling only bumped `N_PROSE`/`N_MATH` — code
  didn't scale with them (it's a fixed real-source harvest), so code's
  share of the mixture fell from 15.7% to 0.5%. Any conclusion about
  code-aware behavior from this corpus is unreliable until domain
  proportions are frozen **by bytes** at every scale (prose/code/math
  at minimum; JSON/tool-call/multilingual/noisy text would be a further
  improvement) instead of letting one domain passively shrink — not
  done yet, a real next step. Re-running
  `grid-screen`: **`survivors = []`.** Every candidate's real T-core
  fertility roughly doubled once measured against genuine priority
  concepts instead of trivial common words — `v11_incumbent` scores
  1.320, no vanilla-trained candidate clears `F_max=2.0`. The original
  survivor was a measurement artifact of an easy target set, not a real
  advantage — hardening caught it before any GPU compute got spent on
  it. Confirmatory follow-up: seeding T-core into SentencePiece training
  as `user_defined_symbols` (both bare and `▁`-prefixed forms, matching
  `v11-builder`'s own real technique) closes the fertility gap exactly
  (1.0), proving a real, replicable path for a future candidate to
  compete — though it still hits the same round-trip/UNK gap as every
  other SentencePiece candidate, a separate, still-open problem. Full
  detail: `pins/tok0_pins.yaml` → `hardening_pass_2026_07_19`.
- **Round 2, same day: byte_fallback fixes UNK completely, round-trip
  needed a real HF-Tokenizer wrapper — and then there was a real
  survivor.** SentencePiece's `byte_fallback` option eliminates UNK
  entirely (0 across the sample corpus) for retrained unigram_sp/bpe_sp
  candidates, but round-trip still failed via native
  `SentencePieceProcessor` — root-caused to SentencePiece's mandatory
  internal metaspace step, which collapses runs of literal spaces
  regardless of normalization settings, unreachable via any public
  training-time toggle. Fix: load the SAME trained pieces+scores into a
  real `tokenizers.Tokenizer` (`models.Unigram` + explicit
  non-collapsing `Metaspace`, matching the canonical-tokenizer decision)
  instead of native SentencePiece encode/decode
  (`HFWrappedSPBackend` in `evaluate_candidate.py`). Found and fixed a
  real bug along the way: byte-fallback `<0xNN>` tokens need
  `decoders.ByteFallback()` chained *before* `Metaspace` on decode, or
  they come back as literal escape-string text instead of the actual
  byte. Result: **0 UNK, 0/32 round-trip failures — a complete fix.**
  Verified this does *not* fix v11 itself: v11's own `tokenizer.json`
  already uses this exact canonical pretokenizer via the plain HF
  library and still shows 543 UNK / 32-of-32 round-trip fail — v11's
  vocab simply has no byte-fallback pieces, a different, deeper,
  separate gap (fixing it means changing v11's frozen vocab, tied to
  already-trained model weights — not attempted). Combining this fix
  with the T-core seeding fix into one candidate produced
  **the funnel's first real, non-exempt Gate G1 survivor**:
  `bpe_sp_16000_v1_tcoreseed_bytefallback` — `t_core_fertility=1.0`,
  `round_trip_pass=true`, `unk_count=0`. A real, earned screening-stage
  result on prototype-scale data, not a claim it's the final production
  tokenizer (that's TOK-2/TOK-3's job). Full detail: `pins/tok0_pins.yaml`
  → `wrapper_fix_and_first_real_survivor_2026_07_19`.
- `pins/tok0_pins.yaml` — commitments C0–C10 are filled in from the
  design doc; most numeric bands (δ_switch, Δ, ε_match, census_N, teg_*,
  mini_ladder_sizes, tok4_candidate_cap) remain explicit
  `PIN: null # not yet decided` — research decisions for Chris, not
  something to invent, and not needed by anything that's run so far.

## Not done here (needs compute / corpus / a human decision)

Gate G1 has run for real, four times now — `survivors = []` the first
three, then `survivors = [bpe_sp_16000_v1_tcoreseed_bytefallback]` once
both real fixes below landed together — see `hardening_pass_2026_07_19`,
`_round2`, and `wrapper_fix_and_first_real_survivor_2026_07_19` in
`pins/tok0_pins.yaml` for the full sequence. Status of each known gap:
- **T-core fertility (SentencePiece family)**: SOLVED, validated —
  seeding T-core as `user_defined_symbols` closes it to exactly 1.0.
- **T-core fertility (byte_level_bpe)**: NOT solved — the seeding
  technique doesn't transfer cleanly to the `tokenizers` library (two
  approaches tried, both documented as real negative results). Needs a
  custom `PreTokenizer`/normalizer-level intervention.
- **Round-trip/UNK (SentencePiece family)**: SOLVED, validated —
  `byte_fallback` fixes UNK (0 across the sample corpus); wrapping the
  trained pieces+scores in a real `tokenizers.Tokenizer`
  (`HFWrappedSPBackend`, `models.Unigram` + explicit non-collapsing
  `Metaspace` + `ByteFallback` decoder) instead of native
  `SentencePieceProcessor` fixes round-trip completely (0/32 failures).
  Same technique that makes `tokenizer.json` diverge from `v11.model` in
  the first place, now put to use deliberately. NOT applied to v11
  itself — v11's gap is a missing-byte-fallback-pieces vocab problem,
  not a wrapping problem (verified directly), and fixing it means
  changing v11's frozen vocab. A separate, bigger, not-yet-made decision.
- **Corpus domain balance**: code's mixture share drifted from 15.7% to
  0.5% when prose/math were scaled and code wasn't (see above) — needs
  domain proportions frozen by bytes, not done yet.
- **Corpus scale**: 21.6MB is still far from a frozen C8; T-core (538
  real items) is a real improvement but not the design doc's frozen
  commitment; `census_R_max` (0.95) is still deliberately permissive,
  not a considered bar, pending category-5 measurement at real scale.

None of the above needs GPU compute — they're corpus/engineering work.
TOK-2a/b/c through TOK-4 model
training and TOK-5 freeze need GPU compute and/or the frozen C10
mini-ladder corpus (not yet built, no precedent found anywhere). Those
are registered as experiments and queued as runs in `chuk-experiments`.
The class-2 canonicalizer (and its parity check) is blocked the same
way — it needs a real candidate's merge table.
