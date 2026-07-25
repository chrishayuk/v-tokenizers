# Roadmap

What is done, what is open, and what is deliberately not being worked on.
Written 2026-07-25.

This file is an index over decisions recorded elsewhere, not a second source
of truth. Where it summarises a result, the authoritative record is the pin
file, pre-registration, or experiment-server entry it points at. If they
disagree, they are right and this is stale.

| Line | State | Next thing that has to happen |
|---|---|---|
| **v11** | Published at 0.1.2, stable, byte-safe | Decide whether v11.1 is worth a vocabulary rebuild — and that waits on v12 |
| **v12** | TOK-2 harness prepared; incumbent gate and C8 code source frozen | Freeze the other four C8 v3 domains, then run the multi-seed panel |
| **v13** | Pre-registration only, nothing built | Blocked on v12 TOK-2b settling |

---

## v11 — shipped

Live on crates.io, PyPI, and the HF Hub since 2026-07-24, at **0.1.2** since
2026-07-25 (0.1.2 is docs plus a wrong `__version__` string — same vocabulary,
same API). Byte-safe since 2026-07-24: `0 UNK, 32/32 files round-trip`,
enforced in CI as a real gate rather than `continue-on-error`. The Rust encoder
is verified token-for-token identical to HF `tokenizers` on the published
`tokenizer.json`.

Treat the shipped vocabulary as frozen. Its ids are baked into already-trained
model weights, so anything that changes the pieces publishes under a **new
name**, never as a new revision of `chrishayuk/v11-tokenizer`.

### Open, ready to do

- **Test coverage.** `v11-cli`, `v11-builder` and `v11-bench` have ~0 dedicated
  unit tests; their logic is currently exercised only indirectly, through the
  bench harness's subprocess-driven checks. No per-file threshold is enforced
  in CI because one would not currently pass — this is follow-up test-writing
  work, not a switch to flip.
- **Leading-space and streaming semantics.** Published v11 keeps its immutable
  canonical `Metaspace(prepend_scheme="always")` behavior. TOK-2 uses the
  separately named `v11-ws-exact` whole-document adapter; its manifest pins the
  newer 71,260-row canonical base and its Rust/Python behavior. Arbitrary-input
  streaming remains unsupported until the stateful interface exists.

### Open, needs a decision first

**v11.1 — the vocabulary gaps.** Three measured coverage failures are recorded
in [`v11/FOLLOWUPS.md`](v11/FOLLOWUPS.md) with their measurements:

| Gap | Measured | Fix cost |
|---|---|---|
| Sentence-initial TitleCase function words fragment | 55/95 (58%) | ~95 slots |
| Common plurals / inflections fragment | 32/38 (84%) | ~4-6K slots |
| Wikidata city coverage | 5/13 capitals single-token | ~200 slots |

All three are `v11-builder` **config** changes — no algorithm change, and the
runtime, HF compatibility and compilation pipeline carry over unchanged.

The reason this is a decision and not a task: a rebuild means a new vocabulary,
which means a new tokenizer repo *and* a model retrain to use it. That is only
worth spending if v11 is the line being taken forward — which is exactly what
v12's TOK-2 has not yet settled (below). Doing v11.1 first would be committing
to the 71K-vocab branch before the comparison that decides it has finished.

## v12 — mid-funnel

Pre-registered TOK-0..TOK-5 design funnel, tracked on the `chuk-experiments`
server under programme `v12-tokenizer`. **Not published, deliberately** — it
stays inside this repo until a candidate wins Gate G1/G2/G3.

Where it actually is:

- **TOK-0/TOK-1 done, and hardened three times.** The third pass produced the
  funnel's first real, non-exempt Gate G1 survivor,
  `bpe_sp_16000_v1_tcoreseed_bytefallback` (`t_core_fertility=1.0`,
  `round_trip_pass=true`, `unk_count=0`). Two earlier "winners" did not
  survive hardening — the first was a measurement artifact of a
  trivially-easy target set.
- **The candidate grid is a pilot, not the design.** `candidate_grid.yaml`
  specifies 4 algorithms × 4 vocab sizes × 3 pre-tokenizations (48 configs).
  What has run is algorithm × vocab-size with a byte baseline. The useful
  six-cell subset (U16/B16 × whitespace/digit/code-aware) is now implemented
  with one serializable training/runtime pipeline, but has not been trained
  at full scale.
- **TOK-2 is not decided in either direction.** A pinned-revision replication
  of v11 (`7d3691e`) reversed TOK-2's earlier "v12 beats v11" conclusion:

  | candidate | held-out BPB (verified-clean) |
  |---|---|
  | **v11, pinned replication (phase1 only)** | **0.6846** |
  | unigram_sp_16000 (U16) | 0.7058 |
  | unigram_sp_18000 (U18) | 0.7062 |
  | bpe_sp_16000 | 0.7461 |

  The honest reading is that this is one run per side, phase1-only, and
  TinyStories-only, with v11's phase3/frozen-FFN retrain deliberately not yet
  run. It is not "v11 wins" — it is "the question is open, and the earlier
  answer was wrong."

  Note these are different vocab-budget classes. The compact candidates are
  being bought for their embedding tax (≈9-11% for U16/U18 vs ≈32% for v11's
  71K vocab), not for raw BPB, so a BPB loss at ≤16K is not automatically a
  loss overall.

**Next:** freeze the remaining real sources in `v12/corpus/c8_v3_spec.json`,
then follow
`v12/TOK2_DECISIVE_PROTOCOL.md`: at least three seeds, phase one plus phase
three, identical raw-byte exposure, and two separately reported controls
(fixed trunk and fixed total parameters). The builder now enforces constant
45/20/15/15/5 byte proportions and fails rather than repeating an undersized
domain. It does not make missing full-scale source data disappear; the spec is
honestly marked draft until those sources, C3 exclusions, and the
repeated-identifier cap are pinned.

The protocol now also pins paired run labels, an FFN-width-only
fixed-total match within 0.25%, byte-parameterized schedules, exactly 16,000
model-visible compact-vocabulary rows, diagnostic-only structural metrics,
and no phase-one elimination. `v12/STREAMING_CONTRACT.md` prohibits arbitrary
stateless chunks. The separately named `v11-ws-exact` adapter now passes the
whole-input gate across Hugging Face, Transformers, and Rust on 23 fixed plus
200 generated Unicode cases, while leaving published v11 unchanged.

The code allocation is now frozen. `v12/corpus/audit_code_pool.py` inventories
immutable Git objects with repository/commit/path/hash/language/licence
provenance and applies the selected pre-model caps. Seven pinned repositories
produce 16.16 MB across TypeScript, Python, C/C++, Go, Rust, configuration, and
smaller language strata. `v12/corpus/CODE_POOL_FREEZE.md` records the
strict/balanced/permissive comparison. Prose, maths/reasoning, JSON/tool/cell,
noisy Unicode, and C3 exclusion evidence remain open.

Until that panel lands, neither promoting a v12 candidate nor committing to
v11.1 is a decision the evidence supports.

## v13 — pre-registered only

EVO-TOK: evolutionary/MCTS search over vocabularies whose fitness signal is
the trained model's held-out BPB per FLOP, rather than BPE's adjacent-pair
frequency or Unigram's corpus likelihood.

**Nothing has been built or run.** The full design is in
[`v13/EVO-TOK-preregistration.md`](v13/EVO-TOK-preregistration.md), including
two corrections made to the source proposal while distilling it, both flagged
in the document rather than silently fixed.

- **Ancestor pinned: U16** (`unigram_sp_16000_v2_tcoreseed_bytefallback`),
  confirmed 2026-07-20. Not U18 — U18's 18,317 tokens violate the programme's
  own ≤16,384 ceiling, and U16/U18 are statistically indistinguishable anyway.
- **Blocked on v12.** It evolves around the best vocabulary v12 found, so it
  cannot start before TOK-2b settles. Any writeup must re-check TOK-2's state
  rather than trusting the pre-registration's tables as current.
- **Scope fixed in advance**: single population, no island ecology; T-core and
  byte-fallback excluded from the genome; MCTS macro-mutation is a separate
  later experiment. Running two novel things at once would make a positive
  result uninterpretable.

## Repo-level

- **`core/` stays unbuilt.** Reserved for Rust genuinely shared between v11 and
  v12, but v12's candidates are Python-only today, so there is nothing real to
  extract. Do not build it speculatively.
- **`v11-wordnet-lemmas` stays unpublished.** Derived from Princeton WordNet;
  redistributing a derivative is a licence decision for a human, not a workflow
  default. The `publish_wordnet` toggle exists and is off.
- **The bench harness is shared on purpose.** `bench/` sits at the repo root
  rather than under either version because it already spans both, and future
  generations should plug into it rather than each growing their own.
- **`v11-bench` and `v11-demos` are not crates.io artifacts.** Their defaults
  are repo-relative (`v11/artifacts`, `v11/corpus`), so an installed copy has
  nothing to point at. Both now carry `publish = false`; before that only
  `v11-demos` did, and a stray `cargo publish -p v11-bench` would have gone
  through. They are exercised by the release checklist and CI, not shipped.

### The release pipeline works, and is not yet trustworthy on its own

Both releases so far produced a bug in the pipeline rather than in the
tokenizer, and in both cases every job reported success while something had
silently not happened. The standing lesson: **check the registries, not the
run's own verdict.**

| Release | What went wrong | What catches it now |
|---|---|---|
| 0.1.0 | Sparse-index path used the web-API shape, so a genuinely successful publish looked like a failure | Correct path derivation, documented in the job |
| 0.1.1 | Skip guard asked whether the crate existed on the index at all — true forever after the first version — so it published nothing and passed | Per-version check, plus an assertion that all three versions are actually on the index |
| 0.1.2 | The rehearsal could only rehearse versions that were *already published*. Packaging `v11-builder`/`v11-cli` one crate at a time resolves `v11-core = "^<new>"` against crates.io, which by definition lacks the new version | One `cargo package` naming all three, which stages them into a temporary local registry and verifies against that |

Every publishing surface now proves its own outcome: crates asserts each
version is really on the sparse index, the tokenizer push replays golden
vectors against the downloaded artifact, and the dataset push (as of
2026-07-25, `scripts/publish_dataset.py`) downloads every file back at the
resulting revision and sha256-compares it against the working tree. No step
still reports success purely on the strength of an accepted request.

**The whole pipeline rehearses.** `workflow_dispatch` takes `dry_run`, which
builds and packages every destination and uploads nothing: all three crates
packaged, all five wheels built, and both HF pushes staged through their
scripts' own `--dry-run`. Nothing is tagged. It needs no confirmation word,
since it cannot publish. This is the check that would have caught 0.1.0's
PyO3-vs-Python-3.14 mismatch without spending a version number.

All three crates are packaged **and** built from the packaged archive, in one
`cargo package` invocation naming all three. That single detail is what makes
it possible: cargo stages the crates being packaged into a temporary local
registry and verifies each against it, so `v11-builder`/`v11-cli` build against
the `v11-core` archive the same run just produced.

Until 0.1.2 this was per-crate, and documented an asymmetry — `v11-core` fully
verified, the other two packaged only — on the grounds that their verification
would need a registry copy of `v11-core` at the new version. The reasoning was
right about the constraint and wrong about the conclusion: batching removes it
entirely. Worse, the per-crate shape did not merely check less, it *could not
run at all* against an unreleased version, so a rehearsal only ever passed for
a version that was already live. Two green rehearsals at 0.1.1 hid this,
because 0.1.1 was on the index by the time they ran. It failed the moment it
was pointed at 0.1.2. Same lesson as the two rows above, arriving a third way:
the rehearsal had never been exercised on the case it exists for.

`publish_dataset.py --verify-only` additionally audits a live dataset repo
against the working tree at any time, no token required.

Open:

- **Transient registry failures still read as release failures.** The
  manylinux legs pull `quay.io/pypa/manylinux2014_*`, which timed out
  mid-release (2026-07-25, "context deadline exceeded") and — because
  `tag-release` requires no job to have failed — silently cost that release
  its tag. The image is now pre-pulled with five attempts and backoff before
  maturin-action runs, so a blip has to persist for over two minutes to fail
  a leg. That narrows the window rather than closing it: the underlying
  coupling, where any red job blocks tagging, is still there by design.
- **`cargo publish` is not transactional across crates.** Publishing three
  crates in dependency order means a failure partway leaves the registry with
  some versions live and some not. It is recoverable — the job is idempotent
  and a re-dispatch finishes the job, which is exactly what happened on
  2026-07-24 — but "0.1.0" briefly meant different things on different
  crates, and no mechanism prevents that.

## Not on the roadmap

- Publishing v12 candidates. They are funnel members; publishing every one
  would fill the namespace with island members. Content-addressed in the
  experiment server is where they belong until one wins.
- Making `v11.model` (native SentencePiece) byte-safe or canonical. Resolved
  2026-07-19: `tokenizer.json` is canonical for v11, and `v11.model` is a
  documented divergent artifact, not an open question.
- A per-file coverage gate in CI before the tests that would pass it exist.
