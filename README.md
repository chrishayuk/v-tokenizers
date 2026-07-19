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
  `v12/pins/tok0_pins.yaml` `incumbent_ledger`). **Not** verified against
  the native SentencePiece `v11.model` used to build the frozen C2 eval
  stream elsewhere -- there's a real, disclosed, unresolved divergence
  there (`v11_three_way_implementation_divergence` in chuk-experiments).
  Don't publish claiming "byte-identical to SentencePiece" until that's
  resolved; "byte-identical to the `tokenizer.json`/`tokenizers`-library
  path" is the accurate, verified claim today.
- **v12**: mid-funnel. 9 real trained candidates + the v11 incumbent
  evaluated; Gate G1 selection is implemented and unit-tested but blocked
  on 3 still-undecided threshold pins (a research decision, not
  engineering). See `v12/README.md`.
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
