# Roadmap

What is done, what is open, and what is deliberately not being worked on.
Written 2026-07-25.

This file is an index over decisions recorded elsewhere, not a second source
of truth. Where it summarises a result, the authoritative record is the pin
file, pre-registration, or experiment-server entry it points at. If they
disagree, they are right and this is stale.

| Line | State | Next thing that has to happen |
|---|---|---|
| **v11** | Published at 0.1.1, stable, byte-safe | Decide whether v11.1 is worth a vocabulary rebuild — and that waits on v12 |
| **v12** | Mid-funnel, first real G1 survivor | TOK-2 is *not decided* — phase3 retrain for v11's side |
| **v13** | Pre-registration only, nothing built | Blocked on v12 TOK-2b settling |

---

## v11 — shipped

Live on crates.io, PyPI, and the HF Hub since 2026-07-24, at **0.1.1** since
2026-07-25. Byte-safe since 2026-07-24: `0 UNK, 32/32 files round-trip`,
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
  What has run is algorithm × vocab-size with a byte baseline; the
  `pre_tokenization` axis is entirely unimplemented.
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

**Next:** the phase3 retrain that makes the two sides comparable. Until that
lands, neither promoting a v12 candidate nor committing to v11.1 is a decision
the evidence supports.

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

## Not on the roadmap

- Publishing v12 candidates. They are funnel members; publishing every one
  would fill the namespace with island members. Content-addressed in the
  experiment server is where they belong until one wins.
- Making `v11.model` (native SentencePiece) byte-safe or canonical. Resolved
  2026-07-19: `tokenizer.json` is canonical for v11, and `v11.model` is a
  documented divergent artifact, not an open question.
- A per-file coverage gate in CI before the tests that would pass it exist.
