# C8 v3 corpus freeze

Status: frozen 2026-07-25, before TOK-2 model results.

C8 v3 contains exactly 80,000,000 UTF-8 text bytes. The domain proportions
are invariant at every corpus scale:

| Domain | Share | Bytes |
|---|---:|---:|
| Natural prose | 45% | 36,000,000 |
| Code | 20% | 16,000,000 |
| Maths/reasoning | 15% | 12,000,000 |
| JSON/tool/cell syntax | 15% | 12,000,000 |
| Noisy Unicode/mixed text | 5% | 4,000,000 |

`c8_v3_spec.json` is the executable freeze. It pins the source-pool hashes,
the fresh C3 manifest, identity normalization, exact deduplication, selection
seed `20260725`, and final corpus hash
`9eee14cd0b1b5b4f2d476d4fbde55c50a40b9aec09d204fd63cf1a036282e466`.
The final JSONL has 266,528 rows.

## Sources

Natural prose comes from shard 0 of the TinyStories train split at revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The raw Arrow file is pinned by
SHA-256 and the upstream dataset declares CDLA-Sharing-1.0.

Maths/reasoning uses the original text of four project-owned Cell80 CN7/CN8
corpora. JSON/tool/cell syntax is a separate canonical JSON rendering of
those rows: metadata, arithmetic traces, and `<call>...</call>` frames are
given explicit fields. The two domains therefore expose different surface
text, and the full builder also removes any exact cross-domain duplicates.
Each raw Cell80 file and the containing repository revision are pinned.

Noisy Unicode/mixed text is selected without normalization or synthetic
corruption from a pinned FineWeb sample-10BT Arrow shard. Eligible rows must
contain both non-ASCII text and a line-breaking or tab character. The
upstream dataset declares ODC-By.

Code is the separately frozen, repository-stratified pool documented in
`CODE_POOL_FREEZE.md`.

## Selection and exclusion

Each generated source pool is chosen without replacement by a stable
SHA-256 order. The full C8 builder exact-deduplicates UTF-8 text within and
across domains, removes every text hash in `c3_v12_manifest.json`, then
selects without replacement. It fails rather than repeat an undersized
domain.

The only truncation is a UTF-8-safe prefix of the final selected row needed
to reach an exact byte allocation. Unicode and source line endings otherwise
remain unchanged.

## Reproduction

Materialize and verify the non-code pools:

```bash
python3 v12/corpus/build_c8_v3_sources.py
```

Regenerate the ignored C3 text artifact and verify its tracked manifest:

```bash
python3 v12/corpus/freeze_c3_v12.py
```

Regenerate the separately frozen code pool if it is absent, then build C8:

```bash
python3 v12/corpus/fetch_code_pool.py --reuse-local-candidates
python3 v12/corpus/audit_code_pool.py
python3 v12/corpus/build_c8_v3.py
```

Two independent full builds were byte-identical. The second build also
independently parsed every output row and confirmed the exact per-domain byte
counts above and zero overlap with all 200 C3 text hashes.
