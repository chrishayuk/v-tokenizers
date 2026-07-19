# v-tokenizers

Knowledge-first tokenizer research and production line. Moved 2026-07-19
out of `tiny-model` into its own repo so `v11` can be published and reused
independently of any one model project, and so the harness/CI around it
doesn't have to live inside a model-training repo. Fresh git history —
`tiny-model`'s own history has the full backstory if it's ever needed.

## Layout

```
v-tokenizers/
  v11/            stable core algorithm, NOT byte-safe yet -- see Status
                   below before calling this publish-ready. v11/README.md
                   and v11/SPEC.md have the design and crate layout
                   (v11-core/v11-builder/v11-cli/v11-bench/v11-demos/v11-python).
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
    publish.yml    SKELETON, manual workflow_dispatch only, actual publish
                    steps commented out -- v11's algorithm/tests/fmt/clippy
                    are clean, but it is NOT byte-safe (real round-trip/UNK
                    gate fails on its own corpus, see Status below) and
                    should not be published claiming production-readiness
                    until that's fixed. This workflow existing is not an
                    endorsement to run it.
```

## Status

- **v11 is NOT byte-safe -- this is a real release blocker, not a
  footnote.** No runtime byte-fallback: literal tabs/newlines/multi-space
  runs embedded in text currently encode to `<unk>`, and `<unk>` is
  dropped entirely on decode (not even a placeholder). Real, measured
  consequence on v11's own 32-file sample corpus: 662 UNK total, 32/32
  files fail exact round-trip -- worst on code (indentation-heavy files
  hit 19-53 UNK each). The CI `roundtrip` job is `continue-on-error:
  true` for exactly this reason: it's a known, tracked, currently-failing
  gate, not a passing one. Until this is fixed, v11 is a research
  control / restricted-domain tokenizer (fine for plain prose without
  heavy indentation or tab-formatted text) -- not something to publish
  or deploy claiming production/byte-safety guarantees. See
  `v12/pins/tok0_pins.yaml` `incumbent_ledger` for the full diagnosis; a
  concrete fix path (SentencePiece `byte_fallback` + explicit
  non-collapsing Metaspace handling) was validated this session for
  freshly-trained v12 candidates but has NOT been applied to v11's own
  vocab/artifacts.
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
- **v12**: mid-funnel, hardened twice (2026-07-19), `survivors = []` both
  times, for increasingly precise reasons. Real 538-item T-core (from
  `v11/config.json`, not a stub) plus an 11.4x-bigger corpus revealed the
  first-pass "winner" was a measurement artifact of a trivially-easy
  target set, not a real advantage -- no vanilla-trained candidate beats
  the incumbent's deliberately-engineered vocab on real priority-concept
  fertility. Confirmatory follow-up: seeding T-core directly into
  SentencePiece training (matching `v11-builder`'s own real technique)
  closes that fertility gap to exactly 1.0 -- a real, validated path for
  a future candidate. Separately, SentencePiece's `byte_fallback` option
  fully fixes the UNK half of the round-trip/UNK hard-reject for v12
  candidates (0 UNK across the sample corpus) -- but round-trip itself
  still fails, root-caused precisely to a structural SentencePiece
  behavior (its mandatory internal metaspace step collapses runs of
  literal spaces) that no public training-time toggle controls. This is
  the same *class* of gap blocking v11 itself above, now understood in
  more depth, not yet fixed in either place. Also tried, real negative
  result: the T-core seeding technique does not transfer cleanly to
  `byte_level_bpe` via the `tokenizers` library (tested two approaches,
  both documented). See `v12/README.md` and
  `v12/pins/tok0_pins.yaml` -> `hardening_pass_2026_07_19` /
  `hardening_pass_2026_07_19_round2` for full detail, including what
  wasn't attempted.
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
- **Publishing**: not done. `publish.yml` exists as a reviewed skeleton
  only; running it requires both an explicit `workflow_dispatch`
  confirmation and `CARGO_REGISTRY_TOKEN`/PyPI trusted-publisher secrets
  that aren't configured yet.

## Consuming from tiny-model

`tiny-model`'s model-training code depends on `v11` via a local Cargo/
Python path dependency (`../../v-tokenizers/v11/...`), not a copy — both
repos are expected to live as siblings under the same parent directory on
a given machine. See `tiny-model/model/v11-train/` for the exact wiring.
