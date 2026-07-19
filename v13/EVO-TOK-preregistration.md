# v13 — Evolutionary/MCTS Tokenizer Search: the experiment programme (draft v0.1)

**Status: pre-registration only. Nothing has been built or run.** This distills an external
design proposal (pasted into chat 2026-07-19/20, reasoning about BPE/Unigram's proxy
objectives vs. an evolutionary search over actual model-and-deployment outcome) into the
scoped, falsifiable, gated form this project's other pre-registrations use
(`evolved-cells-preregistration.md`, `deterministic-ecology.md`, the TOK-0..TOK-5 funnel in
`v12-tokenizer`). Two corrections were made to the source proposal while distilling it —
both flagged explicitly below rather than silently fixed, per the same discipline that
caught the false Gate G1 survivor during TOK-1's hardening pass.

**Depends on:** `v12-tokenizer`'s TOK-1 survivors and TOK-2b's trained checkpoints/held-out
BPB numbers as the ancestor pool — this program does not evolve from raw bytes, it evolves
around the best vocabulary v12 already found.

## 0. The claim being tested

> BPE optimises adjacent-pair frequency. Unigram optimises independent-token corpus
> likelihood. Neither directly asks whether keeping a given token lowers the trained
> transformer's held-out BPB per FLOP. An evolutionary search whose fitness signal is that
> actual quantity — not a proxy for it — can find a ≤16,384-token vocabulary that beats the
> best Unigram candidate on held-out BPB at matched training FLOPs.

This is falsifiable in the plain sense the other pre-registrations require: it fails if,
after a real multi-fidelity search (not just a handful of hand-picked mutants), no evolved
candidate clears the incumbent survivor on held-out BPB at matched byte/FLOP budget.

## 1. Correction #1 to the source proposal — the ancestor pin

The source proposal says: *"U18 is currently the strongest measured starting point, at
0.7427 held-out BPB versus 0.7932 for B16 and 0.8059 for v11."* Those three numbers are
real, but they're the **pre-contamination-fix** TOK-2b numbers (`fb3bbd8` fixed a real
held-out-eval contamination bug the same day). The verified-clean numbers are:

| candidate | held-out BPB (contamination-fixed) |
|---|---|
| v11 (incumbent) | 0.7887 |
| bpe_sp_16000 | 0.7426 |
| unigram_sp_18000 | 0.7047 |
| unigram_sp_16000 | 0.7044 |

Two things follow, neither of which the source proposal had:
- **U16 and U18 are statistically indistinguishable** (0.7044 vs 0.7047) — this *confirms*
  the design doc's own pre-registered P3 prediction ("BPB plateau ~16k"), already recorded
  in `v12-tokenizer`'s TOK-2b writeup.
- **U18 (18,317 tokens) does not satisfy this proposal's own stated `vocabulary <= 16,384`
  fitness constraint.** Picking U18 as ancestor while also pinning a 16,384 ceiling is
  internally inconsistent — the genome would either have to start over-budget or drop ~2K
  tokens before generation 1, muddying the causal story the whole design is trying to keep
  clean (see the source text's own "keep most of the tokenizer architecture fixed... for
  causal interpretation" reasoning).

**Pin: ancestor = `unigram_sp_16000_v2_tcoreseed_bytefallback`** (U16), not U18. It's at
least as strong on the decision metric, it fits the vocabulary ceiling exactly, and it's
already the candidate `v12-tokenizer` most recently trained and held-out-evaluated (least
stale artifact).

## 2. Correction #2 — v11's own contamination status, and why it's not blocking

The source proposal's baseline table uses v11=0.7887 (now current), but flags a standing
gap this repo already knew about: v11's original training never pinned a HF hub revision,
so its own training-consumed document set couldn't be reconstructed to check for held-out
overlap. A pinned-revision replication (`train_v11_replication.py`) was started for exactly
this reason — phase1 finished 2026-07-19 (`model/v11/artifacts_pinned_replication/`,
loss_full=2.1710, 0.23% from the original unpinned run), phase3 intentionally deferred, and
**the held-out BPB for this pinned-replication checkpoint has not actually been computed
yet** (`eval_held_out.py` hasn't been re-run against it — the training_results.json note
promising a `v11_pinned_replication_full` entry is aspirational, not yet real). This
matters for EVO-TOK's own baseline table (§6) but does not block starting: the ancestor and
survivors under test are U16/U18/B16, not v11, and v11 only appears as a comparison
baseline. Treat "v11's contamination-verified held-out BPB" as an open input to backfill
into the baseline table before EVO-TOK's Level 4 (full training) stage reports a final
verdict, not before Level 0-1 screening starts.

## 3. Scope, fixed now, not after seeing what's hard

Matching the source proposal's own "keep most of the architecture fixed" instinct, and this
project's C0-C10 commitment discipline from TOK-0:

**Fixed for all of v1 (no exceptions without a new pre-registration):**
- Ancestor: U16 (§1). C8 pretokenizer. Byte fallback. Protected 538-item T-core (real
  `v11/config.json` items, not the earlier 31-row stub). Unigram Viterbi segmentation.
  Model architecture and training harness (`train_tok2.py`'s byte-matched-budget scheme).
  Vocabulary ceiling ≤16,384.
- **Genome = `normal_tokens: set[~15.5K] + token_scores: map`, T-core and byte-fallback
  excluded from the genome entirely** (protection mutation — T-core membership itself
  becoming evolvable — is explicitly out of scope for v1, exactly as the source proposal's
  own "Later generations could also evolve..." section frames it. This is the same kind of
  scope discipline `evolved-cells-preregistration.md` used for arity-1-only: a real,
  narrower thing, not the full ambition, so a clean pass or fail is interpretable).
- **Single population for v1, no island ecology.** The source proposal's island model
  (prose/maths/code/deployment/robustness islands with periodic migration) and MAP-Elites
  niching are real ideas but a second, independent scope expansion on top of "does
  model-loss-directed evolution beat Unigram at all" — deferred to a v2 pre-registration if
  v1's single-population result is positive enough to justify the added machinery. Running
  both novel things at once (multi-fidelity model-fitness evolution *and* island ecology)
  would make a positive result uninterpretable: which part worked?

**What v1 does NOT attempt** (mirrors the source text's own "Where MCTS fits" conclusion):
MCTS macro-mutation refinement is a **separate, later, dependent experiment** (§7,
`evo-tok-3`), gated on v1 producing both a working mutation operator set and a fitness
surrogate with usable uncertainty estimates. The source proposal's own recommendation is
adopted verbatim: *"Build evolution first... Add MCTS only after useful macro-mutations and
a reliable fitness surrogate exist. Without those, MCTS has no sensible action prior or
affordable rollout value."*

## 4. Method — what's reused vs. what still needs building

**Already built, reused as-is** (from `v12-tokenizer`'s TOK-1/TOK-2 work):
- `v-tokenizers/v12/training/evaluate_candidate.py` + `bench/g1_selection.py` — round-trip,
  UNK count, T-core fertility, vocab-size checks. This is most of "Level 0: deterministic
  rejection" already, not a rebuild.
- `v-tokenizers/v12/training/train_tok2.py` — byte-matched-budget training at any fraction
  of the full budget (just pass a shorter target), reusable directly for both "Level 3:
  short training" (2-5%) and "Level 4: full training" (100%).
- `model/v11-train/eval_held_out.py` (tiny-model repo) — contamination-fixed held-out BPB,
  the actual fitness signal for Levels 3-4.
- 538-item real T-core, C8 v2 corpus (80MB/200,031 rows) as the mutation-candidate source
  corpus.

**Not built yet — required before any run:**
1. **Candidate token pool assembly.** Union of U16/U18/B16 vocabularies + v11 entries +
   frequent C8 substrings + morphological fragments + code identifiers/operators +
   Unigram-pruning rejects. The source proposal's most novel pool source — spans with
   unusually high per-token loss under a trained checkpoint — needs a loss-attribution pass
   over one of TOK-2b's checkpoints (not built; straightforward given the checkpoint and
   eval harness both already exist).
2. **Mutation operators + T-core safety check.** Swap/split/merge/family-mutation/domain-
   migration, each required to (a) never touch a protected T-core entry (fixed for v1, §3)
   and (b) preserve exact byte coverage and round-trip correctness as a hard invariant, not
   a soft preference — the same "sandbox-safe by construction, invalid mutant is a counted
   stillbirth" discipline `deterministic-ecology.md`'s EX-2 used for bytecode mutation.
3. **Level 0/1 fitness harness wiring.** Level 0 mostly exists (item 1 above); Level 1
   (Unigram corpus likelihood, entropy/conditional-entropy, domain-imbalance) needs
   implementing against the C8 v2 corpus.
4. **Level 2 surrogate model.** Needs the accumulated TOK-1/TOK-2 candidate data as training
   examples (currently a handful of points — `candidates.jsonl` plus the 4 TOK-2b
   checkpoints) — almost certainly too few points for a real surrogate at v1's start. This
   is flagged as a likely early bottleneck, not assumed away: v1 may need to run several
   generations on Level 0/1 signal alone, banking Level 3/4 results as they come in, before
   a surrogate is worth training. That's fine and expected, not a failure of scope.
5. **Level 3/4 evaluation loop wiring.** Mechanical — call `train_tok2.py` +
   `eval_held_out.py` per candidate, already both exist.

## 5. Fitness — multi-objective, not collapsed to one score (adopted verbatim from source)

```
minimise:
    held_out_BPB                 (primary decision metric, per v12-tokenizer's own TEG framing)
    FLOPs_to_target_BPB
    worst_domain_BPB
    tokens_per_byte
    dead_token_fraction
    tokenizer_latency

subject to:
    vocabulary <= 16,384          (hard; see Correction #1)
    T-core coverage = 100%        (hard; T-core is genome-excluded in v1, so this is
                                    automatic, not a soft constraint to search around)
    round_trip = exact            (hard; this is the exact TOK-1 gate the incumbent and
                                    every un-fixed SentencePiece candidate FAILED — the
                                    HFWrappedSPBackend fix that produced TOK-1's first real
                                    survivor must carry forward into every evolved genome)
    byte coverage = complete      (hard)
```

Keep the Pareto frontier, don't collapse to a weighted scalar — same MAP-Elites-flavoured
instinct as the source proposal, deferred to v2 for the actual multi-niche machinery (§3),
but the *evaluation* stays multi-objective from v1 day one so the data needed for a future
niched population isn't thrown away by only ever recording one collapsed number.

## 6. Baselines

Required, not optional, before any evolved candidate is claimed better than anything:
- `unigram_sp_16000_v2_tcoreseed_bytefallback` (ancestor, §1)
- `unigram_sp_18000_v2_tcoreseed_bytefallback`
- `bpe_sp_16000_v1_tcoreseed_bytefallback`
- v11 incumbent — contamination-verified number once available (§2), pre-fix 0.7887 used as
  an interim, explicitly-labelled-interim number until then
- **Random token-swap negative control** — same mutation *mechanism*, no fitness selection
  (accept every mutant regardless of score). This is the actual falsification lever: if
  randomly-swapped vocabularies do about as well as selected ones, the fitness signal isn't
  doing the work the hypothesis claims it is. Source proposal listed this; it is promoted
  here from "nice to have" to "required for the claim in §0 to mean anything," matching
  `deterministic-ecology.md`'s EX-1/EX-3 use of a matched no-selection control to rule out
  the mundane explanation before accepting the interesting one.

## 7. The three experiments (funnel, not a single run)

```
evo-tok-1-population-screen  (Level 0-1 only, cheap, decides if there's a signal worth
                               spending model-training compute on)
        │
        ▼
evo-tok-2-model-fitness-ladder  (Level 2 surrogate + Level 3 short-train + Level 4 full-
                                  train, elites only, against the §6 baselines)
        │
        ▼
evo-tok-3-mcts-refinement  (macro-mutation MCTS over evo-tok-2's elites — gated, per §3,
                             on evo-tok-2 actually producing a working mutation set + a
                             surrogate with usable uncertainty; may not fire this cycle)
```

### evo-tok-1-population-screen
**Question.** Does loss/statistics-directed mutation of U16 produce a population that's
measurably different from random mutation on cheap (Level 0-1) signal alone — before
spending any GPU-hour on model training?
**Method.** Population of 32 (U16 ancestor + B16-derived hybrid + U18-pruned-to-16K + 29
mutated descendants at 8/32/128-token mutation budgets, per the source proposal's own
"concrete first experiment" sizing) plus the random-swap negative control (§6) at matched
mutation counts. Score everyone on Level 0 (hard gates) then Level 1 (statistical proxy).
**Gate.** Selected-mutation population's Level 1 distribution is distinguishable from the
random-swap control's (a real, stated statistical test — not "looks different"). **Kill.**
Indistinguishable from random → the cheap signal has nothing to say and either the mutation
operators or the Level 1 metrics need rethinking before any model-training compute is spent.

### evo-tok-2-model-fitness-ladder
**Question.** Do evo-tok-1's top candidates, trained for real, beat the §6 baselines on
held-out BPB at matched byte budget?
**Method.** Top 8 from evo-tok-1 get Level 3 (short training, 2-5% budget); top 2 of those
get Level 4 (full `train_tok2.py` budget) plus the random-swap control's own best survivor
(so the negative control gets a fair shot at the same compute, not just the cheap stage).
**Gate.** At least one full-budget evolved candidate beats U16's real held-out BPB
(0.7044 as of the contamination-fixed number), and does so by more than the random-swap
control's best full-budget result — the second clause is what makes this about selection,
not just "any 16K vocabulary near U16 is about this good."
**Kill.** No evolved candidate beats U16, or the margin over random-swap is not real →
report the negative result plainly (Unigram's independent-likelihood objective was already
close enough to model-loss-optimal at this scale/budget that this search didn't find
daylight) rather than reframe the bar after seeing the number.

### evo-tok-3-mcts-refinement
**Question.** Given evo-tok-2's elites and a working mutation-operator set, can short
sequences of coordinated macro-edits (the source proposal's example: remove
`"computationally"`, add `"ly"`, add a maths token, evaluated as a bundle rather than
one-token-at-a-time) find a state a greedy single-mutation optimiser can't reach?
**Explicitly gated, may not run this cycle.** Only proceeds if evo-tok-2 both (a) produces
a real winner and (b) evo-tok-2's accumulated candidate data is enough to fit a surrogate
with usable uncertainty (§4 item 4) — the source proposal's own precondition for MCTS
having "a sensible action prior or affordable rollout value" at all.
**Method (sketched, not finalized — depends on what evo-tok-2 actually produces).**
State = evo-tok-2's winning genome + N accepted macro-mutation bundles. Action = a named
macro-mutation (remove-K/add-K bundle, family rebuild, domain-budget shift). Depth 4-12
macro-actions. Reward = the Level 2 surrogate's predicted BPB during search, promoted to a
real Level 3/4 evaluation only for the search's most promising leaves.

## 8. Success criterion — draft, needs sign-off, not decided unilaterally

Matching `evolved-cells-preregistration.md`'s discipline of writing a number down before the
run rather than after: **evo-tok-2 counts as a positive result if its best full-budget
evolved candidate beats U16's held-out BPB (0.7044) by at least 1%, AND beats its own
random-swap control's best result by a statistically real margin.** This is a placeholder,
not a considered number — Chris sets the real bar or rejects the framing. Writing it down
now means "1%" can't quietly become "well, 0.3% with an asterisk" after seeing the result
without that being a visible deviation.

## 9. What this would not show, even on a clean pass

- **Not evidence that evolution beats Unigram/BPE in general** — only that it beats *this*
  Unigram candidate, at *this* vocab ceiling, on *this* corpus (C8 v2, prose+math+code+
  structured mixture), at *this* model scale (115M, TOK-2b's trunk). A different scale or
  corpus mixture is a different, unrun experiment.
- **Not a claim about the island/MAP-Elites ecology** — v1 deliberately runs a single
  population (§3); a positive result here says nothing about whether niching would do
  better or worse, only that a v2 pre-registration for it would be worth writing.
- **Not a claim that MCTS refinement helps** — evo-tok-3 is gated and may not run at all
  this cycle; a positive evo-tok-2 result stands on its own regardless.
- **The Level 1-2 proxies are estimates**, same caveat `evolved-cells-preregistration.md`
  gave its chain-cycle proxy: the real word is always the Level 3/4 held-out BPB number,
  never the cheap-stage prediction.

## 10. Next step

Nothing runs until: (a) the ancestor pin (§1: U16, not U18) and the single-population-only
scope (§3) are confirmed rather than assumed from this doc alone, (b) the §8 success bar is
set or the framing rejected, and (c) the four unbuilt Level 0-4 pieces in §4 exist. Filed as
three `draft`-status experiments in chuk-experiments' new `v13-tokenizer-evolution`
programme, `depends_on_experiment` chained to `v12-tokenizer`'s `tok-1-intrinsic-screen` /
`tok-2b-fixed-trunk` for the ancestor pool and baseline numbers.
