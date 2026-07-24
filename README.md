# v-tokenizers

Knowledge-first tokenizer research and production line. Moved 2026-07-19
out of `tiny-model` into its own repo so `v11` can be published and reused
independently of any one model project, and so the harness/CI around it
doesn't have to live inside a model-training repo. Fresh git history —
`tiny-model`'s own history has the full backstory if it's ever needed.

## Layout

```
v-tokenizers/
  v11/            stable core algorithm, now byte-safe (see Status below).
                   v11/README.md and v11/SPEC.md have the design and crate
                   layout (v11-core/v11-builder/v11-cli/v11-bench/v11-demos/
                   v11-python).
  v12/            active, pre-registered research funnel (TOK-0..TOK-5).
                   NOT published -- see v12/README.md and
                   v12/pins/tok0_pins.yaml. Promoted to v11's status only
                   once a candidate wins Gate G1/G2/G3.
  bench/          COMMON harness/CLI shared across v11 and v12 -- it
                   already spans both versions (drives v11's compiled
                   binary for the real roundtrip gate check, and
                   implements v12's census/intrinsics/grid-screen). Kept
                   at the repo root deliberately, not nested under either
                   version, since future tokenizer generations should
                   plug into the same harness rather than each growing
                   their own.
  core/           RESERVED, not yet created. Meant for Rust logic actually
                   shared between v11 and v12 (vocab loading, trie/error
                   types, special-token handling) once v12 has a Rust-side
                   tokenizer implementation of its own to de-duplicate
                   against -- today v12's candidates are Python-only
                   (sentencepiece/tokenizers libraries), so there's nothing
                   real to extract yet. Don't build this speculatively.
  .github/workflows/
    ci.yml         build + test + fmt --check + clippy -D warnings +
                    coverage (cargo-llvm-cov) + the Python harness smoke
                    tests, on every push/PR.
    publish.yml    Real release pipeline (crates.io + PyPI + HuggingFace
                    Hub), manual workflow_dispatch only -- v11's algorithm/
                    tests/fmt/clippy are clean and it is now byte-safe (real
                    round-trip/UNK gate passes on its own corpus, see Status
                    below). Requires typing "publish" to confirm plus
                    CARGO_REGISTRY_TOKEN/PYPI_TOKEN/HF_TOKEN repo secrets,
                    none of which are configured yet -- this workflow
                    existing is not an endorsement to run it unattended.
```

## Status

- **v11 is now byte-safe -- FIXED 2026-07-24.** `v11-core`'s own Rust
  tokenizer (`v11/core/src/tokenizer.rs`) had no runtime byte-fallback:
  literal tabs/newlines/multi-space runs and any character outside the
  vocab encoded to `<unk>`, which `decode()` then dropped entirely (not
  even a placeholder). Real, measured consequence on v11's own 32-file
  sample corpus: 662 UNK total, 32/32 files failed exact round-trip --
  worst on code (indentation-heavy files hit 19-53 UNK each). Root cause,
  found by tracing the actual algorithm: the vocab already carried 256
  `<0xNN>` byte-fallback pieces, but they were only ever matched as
  literal 6-character strings in the trie (real text essentially never
  contains that string) -- nothing routed an uncovered byte to its
  fallback piece by *value*. Fix: `Tokenizer` now builds a `byte_ids:
  [Option<u32>; 256]` table (byte value -> vocab id) at construction; on
  encode, a byte with no trie match becomes its own one-byte lattice edge
  via that table instead of `unk_id` (continuation bytes naturally repeat
  the same path byte-by-byte, so an uncovered multi-byte UTF-8 char
  becomes N byte-fallback ids); on decode, a byte-fallback id contributes
  its raw byte value, not its `<0xNN>` piece text, so the original bytes
  reassemble exactly. Verified against this repo's own gate, not just
  new unit tests: `python3 bench/tokenizer_bench.py roundtrip` now
  reports **0 UNK, 0 round-trip mismatches, 32/32 files pass** (was 662 /
  32-of-32-fail before the fix) -- the CI `roundtrip` job is a real,
  enforced gate now, not `continue-on-error`.
  **The `tokenizer.json` artifact itself had the identical bug, separately**
  -- checked directly against the same 32-file corpus via the real HF
  `tokenizers` Python library (not v11-core): 543 UNK, 18/32 round-trip
  fail, before any fix. (An earlier, narrower check against three plain-
  English prose samples during the tinystories-train-video session showed
  0 UNK and wrongly read as "this path is fine" -- it simply never
  exercised a tab or an uncovered character. Corrected here so the record
  doesn't overclaim.) Root cause was the JSON's own declared pipeline, not
  v11-core: `model.byte_fallback` was unset and `decoder` was a bare
  `Metaspace` -- so HF's own Unigram implementation, loading this exact
  file, had no route from an uncovered byte to its `<0xNN>` piece either.
  Fixed the same way v12's wrapper fix works: `model.byte_fallback: true`
  plus `decoder: Sequence([ByteFallback, Metaspace])` (order matters --
  reassemble bytes first, un-replace `▁` second). Fixed at the source in
  `v11-builder`'s `write_hf_tokenizer_json` (so a future vocab rebuild
  doesn't regress it) and applied to the checked-in `v11/artifacts/
  tokenizer.json` (vocab entries byte-for-byte unchanged, only the model
  flag + decoder differ). Re-verified against the same 32-file corpus via
  the HF `tokenizers` library directly: 0 UNK, 0/32 mismatches.
  See `v12/pins/tok0_pins.yaml` `incumbent_ledger` for the v12 fix both of
  these were adapted from (SentencePiece `byte_fallback` + non-collapsing
  Metaspace, validated there 2026-07-19) and the entry recording this v11
  fix.
- **v11 algorithm/implementation**: builds, all 18 core unit tests pass,
  `cargo fmt --check` and `cargo clippy -- -D warnings` both clean.
  Verified token-for-token identical to real HF `tokenizers` on
  `tokenizer.json` (a Python/Rust parity bug was found and fixed
  2026-07-19). The native SentencePiece `v11.model` used to build the
  frozen C2 eval stream elsewhere diverges from `tokenizer.json` (a real
  3-way implementation-divergence finding) -- RESOLVED 2026-07-19:
  `tokenizer.json`'s behavior is canonical for v11 going forward
  (`incumbent_ledger.RESOLVED_2026_07_19_canonical_tokenizer_decision`).
  Note this is a provenance/authority decision about which artifact is
  canonical, not a claim that either one is byte-safe -- see above.
  "Byte-identical to the `tokenizer.json`/`tokenizers`-library path" is
  the accurate claim; "byte-identical to SentencePiece" is not.
- **v12**: mid-funnel, hardened three times (2026-07-19) -- and the
  third time produced **the funnel's first real, non-exempt Gate G1
  survivor**: `bpe_sp_16000_v1_tcoreseed_bytefallback`. Sequence: real
  538-item T-core (from `v11/config.json`, not a stub) plus an
  11.4x-bigger corpus first revealed the original "winner" was a
  measurement artifact of a trivially-easy target set (`survivors = []`);
  seeding T-core directly into SentencePiece training (matching
  `v11-builder`'s own real technique) closed the fertility gap to
  exactly 1.0; `byte_fallback` fixed UNK completely but round-trip still
  failed via native SentencePiece (a structural behavior -- its
  mandatory metaspace step collapses runs of literal spaces -- reachable
  by no public training-time toggle); wrapping the trained vocab in a
  real `tokenizers.Tokenizer` (explicit non-collapsing Metaspace +
  `ByteFallback` decoder) instead of native `SentencePieceProcessor`
  fixed round-trip completely (0/32 failures). Combining the seeding fix
  and the wrapper fix in one candidate produced the survivor above --
  `t_core_fertility=1.0`, `round_trip_pass=true`, `unk_count=0`. This
  does **not** fix v11 itself: checked directly, v11's own
  `tokenizer.json` already uses this exact canonical pretokenizer and
  still shows 543 UNK / 32-of-32 round-trip fail -- its vocab simply has
  no byte-fallback pieces, a different, separate, not-yet-made decision
  (changing v11's frozen vocab, tied to already-trained model weights).
  Also tried, real negative result: the T-core seeding technique does
  not transfer cleanly to `byte_level_bpe` via the `tokenizers` library
  (tested two approaches, both documented). This is a real, earned
  screening-stage result on prototype-scale data -- not a claim it's the
  final production tokenizer, which needs TOK-2/TOK-3 real model
  training. See `v12/README.md` and `v12/pins/tok0_pins.yaml` ->
  `hardening_pass_2026_07_19`, `_round2`, and
  `wrapper_fix_and_first_real_survivor_2026_07_19` for full detail.
- **This screen prunes candidates; it does not pick a production
  tokenizer.** Compression/fertility/round-trip are hard/threshold
  rejection criteria (TOK-1, Gate G1) -- they tell you what's obviously
  unsuitable or dominated, not what a trained model will actually do
  best with. That question is TOK-2/TOK-3's, and needs real model
  training at real compute, not run here.
- **Coverage**: no per-file threshold enforced in CI yet.
  `v11-cli`/`v11-builder`/`v11-bench` have ~0 dedicated unit tests today
  (their logic is currently exercised indirectly via the bench harness's
  real subprocess-driven checks, not real unit coverage) — getting every
  file to a real 90% is tracked as follow-up work, not claimed as done.
- **Publishing**: not done, but the pipeline is real now (2026-07-24) --
  `publish.yml` actually publishes to crates.io (v11-core, v11-builder,
  v11-cli, in dependency order, polling the sparse index in between),
  PyPI (v11-python via maturin), and HuggingFace Hub (`chrishayuk/
  v11-tokenizer` by default, overridable per-dispatch), each independently
  toggleable, gated on typing `publish` to confirm. Running it still needs
  `CARGO_REGISTRY_TOKEN`/`PYPI_TOKEN`/`HF_TOKEN` repo secrets, none of
  which are configured yet, and is still a deliberate human action, not
  something CI triggers on its own.

## Consuming from tiny-model

`tiny-model`'s model-training code depends on `v11` via a local Cargo/
Python path dependency (`../../v-tokenizers/v11/...`), not a copy — both
repos are expected to live as siblings under the same parent directory on
a given machine. See `tiny-model/model/v11-train/` for the exact wiring.
