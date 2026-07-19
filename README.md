# v-tokenizers

Knowledge-first tokenizer research and production line. Moved 2026-07-19
out of `tiny-model` into its own repo so `v11` can be published and reused
independently of any one model project, and so the harness/CI around it
doesn't have to live inside a model-training repo. Fresh git history —
`tiny-model`'s own history has the full backstory if it's ever needed.

## Layout

```
v-tokenizers/
  v11/            stable, tested, publish-ready. See v11/README.md and
                   v11/SPEC.md for the design and crate layout
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
                    steps commented out -- v11 is publish-*ready* (tests
                    pass, clean fmt/clippy) but has not been published
                    anywhere yet. That's a separate, explicit action for
                    later, not something to wire up to fire automatically.
```

## Status

- **v11**: builds, all 18 core unit tests pass, `cargo fmt --check` and
  `cargo clippy -- -D warnings` both clean. Verified token-for-token
  identical to real HF `tokenizers` on `tokenizer.json` (a Python/Rust
  parity bug was found and fixed 2026-07-19 -- see
  `v12/pins/tok0_pins.yaml` `incumbent_ledger`). The native SentencePiece
  `v11.model` used to build the frozen C2 eval stream elsewhere diverges
  from `tokenizer.json` (a real 3-way implementation-divergence finding)
  -- RESOLVED 2026-07-19: `tokenizer.json`'s behavior is canonical for
  v11 going forward (`incumbent_ledger.RESOLVED_2026_07_19_canonical_tokenizer_decision`);
  `v11.model`/native-SentencePiece is a documented divergent artifact,
  not an open question. "Byte-identical to the `tokenizer.json`/
  `tokenizers`-library path" is the accurate claim; "byte-identical to
  SentencePiece" is not, and shouldn't be claimed when publishing.
- **v12**: mid-funnel, hardened once (2026-07-19). First real Gate G1 run
  (prototype-scale C8, stub T-core) gave `survivors = [byte_level_bpe_8000_v0]`.
  Hardening (real 538-item T-core from v11/config.json, C8 scaled 11.4x,
  grid retrained at real vocab sizes) changed the result to
  `survivors = []` -- the original survivor was a measurement artifact of
  an easy target set, not a real advantage. A confirmatory experiment
  (seeding T-core into training, matching v11-builder's own technique)
  proved a real path to close the gap exists, but no candidate combines
  it with a fix for the pre-existing round-trip/UNK gap yet. See
  `v12/README.md` and `v12/pins/tok0_pins.yaml` -> `hardening_pass_2026_07_19`.
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
