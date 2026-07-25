# v-tokenizers

Knowledge-first tokenizer research and production line. Moved 2026-07-19
out of `tiny-model` into its own repo so `v11` can be published and reused
independently of any one model project, and so the harness/CI around it
doesn't have to live inside a model-training repo. Fresh git history —
`tiny-model`'s own history has the full backstory if it's ever needed.

[ROADMAP.md](ROADMAP.md) has what's done, what's open, and what's deliberately
not being worked on across all three lines.

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
  v13/            PRE-REGISTRATION ONLY, nothing built or run. EVO-TOK:
                   evolutionary/MCTS search over vocabularies whose fitness
                   is a trained model's held-out BPB per FLOP, rather than
                   BPE's pair frequency or Unigram's corpus likelihood.
                   Blocked on v12 TOK-2b settling, since it evolves around
                   the best vocabulary v12 finds. See
                   v13/EVO-TOK-preregistration.md.
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
                    tests + the dataset-catalog sync check, on every push/PR.
    publish.yml    Release pipeline, manual workflow_dispatch only, typing
                    "publish" to confirm. Four independently toggleable
                    destinations: crates.io, PyPI (five-platform abi3 wheel
                    matrix), the tokenizer to HF (via
                    scripts/publish_tokenizer.py), and the corpus to HF as a
                    dataset repo -- then a tag-release job that tags the
                    released commit if none of them failed. v11 0.1.0 shipped
                    to all three registries 2026-07-24; see Releases below.
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
- **Publishing: v11 is live on all three registries (2026-07-24).** See
  [Releases](#releases) for the version table, what 0.1.1 changes, and how
  tagging works.

  ```sh
  pip install v11-tokenizer
  cargo add v11-core
  ```

  Note the PyPI **distribution** name is `v11-tokenizer` and the import is
  `import v11` (maturin's `module-name`), even though the crate directory and
  `publish.yml`'s input are called `v11-python` — three names for one thing,
  and none of them interchangeable. Don't go looking for a `v11-python` on
  PyPI.
  The Hub's `tokenizer.json` is byte-identical to `v11/artifacts/tokenizer.json`
  (sha256 `10dd5110…`), i.e. the post-byte-safety-fix build, and to the copy
  vendored in `tinystories-train-video/training/harness_pretrain/`.

  `publish.yml` remains manual (`workflow_dispatch`, confirm-gated) and needs
  `CARGO_REGISTRY_TOKEN`/`PYPI_TOKEN`/`HF_TOKEN`. Tokenizer publication goes
  through `scripts/publish_tokenizer.py`, not a raw upload -- see below. It is idempotent — each crate
  is skipped if already on the sparse index — so it is safe to redispatch after
  a partial failure. v12 is still deliberately excluded.

## Releases

| Destination | Name | Current |
|---|---|---|
| crates.io | `v11-core`, `v11-builder`, `v11-cli` | 0.1.2 |
| PyPI | **`v11-tokenizer`** (import `v11`) | 0.1.2 — 5 abi3 wheels + sdist |
| HF Hub (model) | `chrishayuk/v11-tokenizer` | built at `ee502e0` |
| HF Hub (dataset) | `chrishayuk/v11-corpus` | built at `ee502e0` |
| HF Hub (dataset) | `chrishayuk/v11-wordnet-lemmas` | not published |

**The Hub artifact does not carry the crate version, and that is deliberate.**
Its identity is `sha256(tokenizer.json)` (`10dd5110…`), because a version
string is an assertion and a content hash is a fact -- see *Publishing a
tokenizer* below. The crate/wheel version tracks the *code* that reads the
vocabulary; the vocabulary itself is unchanged since the 2026-07-24
byte-safety fix. Do not expect them to move together.

`v11-wordnet-lemmas` stays unpublished on purpose: it derives from Princeton
WordNet, and redistributing a derivative is a licence decision for a human,
not a workflow default. The `publish_wordnet` toggle is off by default.

### What 0.1.2 changed

Documentation and one wrong string. No vocabulary change, no algorithm change,
no API change — `tokenizer.json` is the same bytes it has been since the
2026-07-24 byte-safety fix, so the Hub artifact is untouched by this release.

- **`v11.__version__` stopped lying.** It was a hardcoded `"0.1.0"` literal in
  `v11/python/src/lib.rs`, so all five 0.1.1 wheels reported the previous
  version. It now reads `env!("CARGO_PKG_VERSION")` — the literal was never in
  *Cutting a release*'s bump list below, which is exactly why it drifted, and a
  value derived from the manifest cannot drift again.
- **`v11/README.md` still documented the pre-byte-fallback behaviour**, saying
  the `<0xNN>` pieces were "never matched in practice" and that tabs and
  newlines become `<unk>`. That has been false since 2026-07-24 and contradicted
  this file, the enforced CI gate, and the published artifact.
- **The import name was given two ways.** One paragraph above said `import
  v11_tokenizer`; the release table and `pyproject.toml`'s `module-name` say
  `v11`, which is correct.
- **The layout above omitted v13**, which the roadmap has listed as an active
  line since its pre-registration landed.

The first item is the reason this is a release rather than a doc commit: PyPI
renders a package's description from the uploaded distribution and there is no
way to update it in place, so 0.1.1's page keeps its stale README until a newer
version ships. Publishing is the only way to correct what PyPI shows.

**And the release found a bug in the pipeline again — the third in three.** The
`dry_run` rehearsal added for 0.1.1 could only rehearse a version that was
*already published*. It packaged each crate separately, so packaging
`v11-builder` resolved `v11-core = "^0.1.2"` against crates.io and failed
because that is precisely the version not there yet. Both of its green runs had
been at 0.1.1, after 0.1.1 was live, so the case it exists for had never once
been exercised. Fixed by packaging all three in one `cargo package`, which
stages them into a temporary local registry — that also makes `v11-builder` and
`v11-cli` fully *verify* (build from their own archive) rather than merely
package, removing an asymmetry the job had documented as unavoidable.

### What 0.1.1 changed

Published 2026-07-25. Two things, both consequences of 0.1.0 having been cut
before they were noticed:

- **`v11 vocab` reached the published CLI.** The subcommand landed in `ee502e0`
  about three hours after `v11-cli` 0.1.0 went to crates.io, so
  `cargo install v11-cli` got a binary without it.
- **Wheels for platforms other than one.** 0.1.0 put a single
  `cp312-macosx_11_0_arm64` wheel on PyPI, so `pip install v11-tokenizer`
  on Linux, Windows, an Intel Mac, or any non-3.12 Python fell through to the
  sdist and needed a Rust toolchain to build it. `v11-python` now builds
  against PyO3's stable ABI (`abi3-py39`) across five target triples, so one
  wheel per platform covers every Python >= 3.9. The `cp39-abi3` wheel built
  under 3.12 installs and round-trips under 3.14.

**The release itself found a third bug, in the pipeline.** The first 0.1.1
dispatch published all five wheels to PyPI, published *nothing* to crates.io,
and reported success. `publish-crates` asked whether each crate had a
sparse-index file at all -- true forever once the first version ships -- so
every release after the first would have skipped all three crates and gone
green having uploaded nothing. It surfaced only because the registries were
checked afterwards rather than the run's own verdict being trusted.

The check is now per-version, and the job asserts all three versions are
actually on the index before it can pass. Two details worth keeping:

- **No `curl | grep -q`.** Under `set -o pipefail` grep exits at the first
  match, curl dies of SIGPIPE, and the pipeline reports failure on the
  *success* path -- which would republish a version that is already live.
  The body is captured and matched instead.
- **The assertion is the actual fix.** The bug was invisible because every
  individual step succeeded; a release that publishes nothing has to fail
  loudly rather than be inferred from control flow.

### Tags

Every release commit carries an annotated `vX.Y.Z` tag, created by the
`tag-release` job in `publish.yml` once no publish job has failed.

| tag | commit | |
|---|---|---|
| `v0.1.0` | `8f9c942` | retroactive, approximate — see below |
| `v0.1.1` | `6745f40` | created by `tag-release` |
| `v0.1.2` | `3ed5f3c` | created by `tag-release` |

A release's own commit cannot name its tag — the commit has to exist before
`tag-release` can point at it — so this row always lands in the commit after.

0.1.0 predates that job and is tagged retroactively at `8f9c942`. That tag is
approximate by necessity and says so in its own message: `v11-core` and the
PyPI wheel actually shipped from the parent commit `60ab7c2` before a partial
failure was fixed and redispatched, so no single commit is exactly right. The
two differ only in `.github/workflows/publish.yml`, so the published crate and
wheel *sources* are identical at both. This ambiguity is the reason the job
exists.

### Cutting a release

1. Bump `version` in five places: the root `Cargo.toml` `[workspace.package]`
   and its `[workspace.dependencies].v11-core`, then `v11/python/Cargo.toml`'s
   `[package].version` *and* its own `v11-core` dep line, and
   `v11/python/pyproject.toml`. `v11/python` is workspace-excluded, so nothing
   it carries is inherited — it is the one that gets missed.

   Nothing in the tree is a version literal that has to be hand-edited beyond
   these. `v11.__version__` reads `CARGO_PKG_VERSION` precisely so it can't
   become a sixth one (it was, through 0.1.1, and reported the wrong version).
2. `cargo test --workspace && cargo clippy --all-targets -- -D warnings`, then
   `cargo check --manifest-path v11/python/Cargo.toml`. Both lockfiles are
   tracked and both must be committed; the second command is the only thing
   that updates `v11/python/Cargo.lock`, since a workspace build never touches
   it.
3. Rehearse it first. `dry_run` builds and packages every destination and
   uploads nothing — three crates packaged *and* built from their own
   archives, five wheels built, both HF pushes staged — and tags nothing. No
   confirmation word, because it cannot publish:

   ```sh
   gh workflow run publish.yml -f dry_run=true
   ```

4. Dispatch **Release (manual)** for real, typing `publish` to confirm:

   ```sh
   gh workflow run publish.yml -f confirm=publish \
     -f publish_crates=true -f publish_pypi=true \
     -f publish_hf=true -f publish_datasets=true
   ```

   Every destination is independently toggleable, and each is idempotent
   (crates skip if already on the sparse index, `maturin publish` uses
   `--skip-existing`, the tokenizer push refuses to overwrite a differing
   hash), so redispatching after a partial failure is safe and expected.
5. The `tag-release` job pushes `vX.Y.Z`. It refuses to move a tag that
   already points somewhere else -- bump the version instead. It does not run
   on a dry run.

## Consuming from tiny-model

`tiny-model`'s model-training code depends on `v11` via a local Cargo/
Python path dependency (`../../v-tokenizers/v11/...`), not a copy — both
repos are expected to live as siblings under the same parent directory on
a given machine. See `tiny-model/model/v11-train/` for the exact wiring.

## Publishing a tokenizer

`scripts/publish_tokenizer.py` publishes one tokenizer as an immutable,
`AutoTokenizer`-loadable **model** repo (not a dataset repo). Acceptance
criterion: a clean environment loads it with no clone, no training code and no
`trust_remote_code` -- verified against the *downloaded* artifact before the
script reports success.

```sh
uv run scripts/publish_tokenizer.py \
  --tokenizer-json v11/artifacts/tokenizer.json \
  --repo-id chrishayuk/v11-tokenizer --status adopted --dry-run
```

Five things it enforces that a plain `upload_folder` cannot:

- **Identity is the content hash, not the Hub revision.** A commit oid changes
  when you re-push identical bytes or fix a typo in the card. The anchor is
  `sha256(tokenizer.json)` -- the same value a checkpoint records as
  `tokenizer_hash`. The revision is recorded as a retrieval coordinate, never
  as the gate.
- **Immutability is a precondition.** If the repo exists, its `tokenizer.json`
  is fetched and hashed *before* anything uploads; a mismatch refuses the push
  and tells you to publish under a new name. There is no `--force`.
- **Cross-major loadability.** transformers 5.x writes
  `tokenizer_class: "TokenizersBackend"`, which 4.x rejects outright
  (measured: 4.46.3 raises `Tokenizer class TokenizersBackend does not exist`).
  The staged config is pinned to `PreTrainedTokenizerFast`, which both resolve.
  Verified on 4.46.3 and 5.14.1, 116/116 golden vectors each.
- **Golden-vector verification, not a length check.** `len(tok)` plus
  `special_tokens_map` equality passes straight through a changed normalizer or
  post-processor. Every frame in `bench/msi/frame_battery.jsonl` -- call
  operands, post-delimiter, mid-string punctuation -- is replayed id-for-id
  against the downloaded artifact.
- **Provenance is a pointer.** `provenance.json` carries the chuk-datasets
  dataset/version/sha and a chuk-experiments run id. It does not copy the corpus
  manifest or the results ledger into a second source of truth.

`--status` defaults to `candidate`, which stamps "candidate, NOT adopted" into
the card. Anything published before TOK-4 adjudicates must stay there.

**When to publish at all:** when an artifact acquires *external* identity --
referenced by a model repo, a paper, or a reviewer. EVO-TOK will generate
populations; publishing every candidate would fill the namespace with island
members. Everything else stays content-addressed in the experiment server.

## Publishing a dataset

`scripts/publish_dataset.py` pushes a directory or single file to a **dataset**
repo, then proves what landed: every file is downloaded back at the resulting
revision and sha256-compared against the working tree. Missing files mean a
partial upload; unexpected files mean the repo carries something this push did
not put there, which a reader would reasonably assume is current.

```sh
# audit the live repo against the working tree, uploading nothing
uv run scripts/publish_dataset.py --root v11/corpus \
  --repo-id chrishayuk/v11-corpus --verify-only
```

`--verify-only` exits non-zero if the published dataset has drifted, so it works
as a standalone audit rather than only as part of a release. Cards live in
`scripts/cards/` rather than inline in the workflow, so they can be reviewed and
diffed as text.

## Datasets in the catalog

`datasets.json` declares what this repo registers in
[chuk-datasets](https://chuk-datasets.fly.dev) (`v11/corpus`,
`v11/wordnet-lemmas`, `v11/tokenizer`). CI verifies on every push that each
`content_sha` recomputed from disk still matches the catalog, so the corpus the
round-trip gate measures cannot drift away from the corpus the catalog says was
measured. Verification needs no credentials; registration needs a write-scoped
`CHUK_DATASETS_API_KEY`:

```sh
uv run <chuk-datasets-server>/jobs/register-files/register.py verify   datasets.json
uv run <chuk-datasets-server>/jobs/register-files/register.py register datasets.json
```
