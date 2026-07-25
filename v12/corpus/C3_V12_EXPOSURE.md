# C3 v12 pre-freeze exposure appendix

Status: frozen 2026-07-25, before TOK-2 model results.

C3 v12 is a fresh held-out slice for the v12/TOK-2 protocol. It does not
reuse the spent 40-item corpus-atlas C3 instrument. The new slice contains
40 items from each C8 v3 domain, 200 items total, selected without replacement
by `sha256(seed, domain, text_sha256)` with seed `20260726`.

The tracked `c3_v12_manifest.json` contains the exact domain, provenance, text
hash, and byte count for each item. The held-out texts remain in the ignored
`c3_v12_heldout.jsonl` artifact, whose frozen SHA-256 is
`52ab2ebb20b80585d56fe6f2ba292750b95c04eeef1af12673d6ad651710d08f`.

## Exposure before freeze

The parent corpora were known and previously available:

- TinyStories was already used by the tokenizer evaluation pipeline.
- The CN7/CN8 files were generated for earlier Cell80 experiments.
- The code repositories were audited while freezing the C8 code pool.
- The FineWeb shard was present in the local Hugging Face cache.

No TOK-2 model result, tokenizer score, loss, structural profile, or
per-item C3 score was inspected when the C3 seed or selection rules were
chosen. The 200-item subset was selected mechanically, and its full text is
not committed.

The separately frozen code pool had only 151,568 unique bytes above its
16 MB target. Therefore, before the hash draw, code C3 eligibility was
limited to rows no larger than 2,048 bytes and the 40 selected rows were
bounded to 81,920 bytes total. The actual code holdout is 35,213 bytes.
This constraint is disclosed in `c3_v12_spec.json`; it prevents the holdout
from making the already-frozen training allocation infeasible. It makes C3's
code slice a correctness/exclusion instrument, not a length-representative
code benchmark.

The final C8 build excludes all 200 exact UTF-8 text hashes. An independent
post-build scan found zero C3 overlaps.
