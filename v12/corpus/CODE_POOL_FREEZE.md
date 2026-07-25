# C8 v3 code-pool freeze

Status: frozen 2026-07-25, before TOK-2 model results.

This freezes the 20% code source allocation. C8 v3 as a whole was subsequently
frozen in `C8_V3_FREEZE.md`, including prose, maths/reasoning,
JSON/tool/cell, noisy Unicode, and fresh C3 exclusion evidence.

## Source strata

The pool contains seven immutable Git revisions:

- project-relevant tokenizer and Cell80 code from `v-tokenizers`;
- C/C++ library, application, test, and build code from `llama.cpp`;
- C++, C, Python, and build code from `mlx`;
- Rust library, CLI, server, and test code from `mistral.rs`;
- Python and Swift applications from `mlx-examples`;
- Go protocol, example, test, and conformance code from the MCP Go SDK;
- TypeScript/JavaScript protocol, application, test, and configuration code
  from the MCP TypeScript SDK.

Every repository entry in `code_pool_spec.json` pins its URL, full commit,
licence classification, licence-evidence path, and licence-evidence SHA-256.
The audit reads blobs from those Git objects, never mutable worktrees.

## Diagnostic and cap decision

Before caps, the accepted pool contained 3,389 files and 41,466,136 bytes.
It was not balanced: `llama.cpp` contributed 24,668,509 bytes (59.5%), and
C++ contributed 16,700,227 bytes (40.3%). Forty-seven files exceeded 128 KiB
and together contributed 11,707,341 bytes.

Three cap scenarios were declared before selection:

| Scenario | Selected bytes | Outcome |
|---|---:|---|
| strict | 13,057,321 | rejected: cannot fill the 16 MB allocation |
| balanced | 16,159,545 | selected |
| permissive | 17,000,000 | rejected: retains more concentration than needed |

The frozen balanced caps are:

| Cap | Value |
|---|---:|
| bytes per file | 131,072 |
| bytes per repository | 5,000,000 |
| share per language | 30% |
| occurrences per normalized identifier | 7,500 |
| files per SimHash near-duplicate cluster | 4 |

Identifier counting removes comments and quoted strings, lowercases ASCII
identifiers, normalizes digit runs, and excludes one-character names and a
pinned cross-language keyword list. The identifier cap therefore measures
source-specific lexical concentration rather than occurrences of `const`,
`return`, or similar syntax.

After caps, the selected 2,340 files contain 16,159,545 text bytes. C8 ignores
three empty files; exact deduplication then leaves 2,324 eligible rows and
16,151,568 bytes. The largest repository contributes 4,999,992 bytes; the
largest language is TypeScript at 4,684,560 bytes (29.0% of the selected
pool). Python, C++, Go, and Rust remain substantial independent strata.

## Reproduction

Use existing verified sibling clones when available:

```bash
python3 v12/corpus/fetch_code_pool.py --reuse-local-candidates
```

On a clean machine, omit `--reuse-local-candidates` to populate the ignored
canonical cache from the pinned URLs. Then run:

```bash
python3 v12/corpus/audit_code_pool.py
```

The generated inventory, audit report, selected 16 MB JSONL, and repository
cache are ignored. `code_pool_frozen_manifest.json` is the tracked compact
evidence artifact. `c8_v3_spec.json` pins the selected JSONL hash; the full C8
builder refuses a mismatched or undersized source.

Independent stateless chunking and tokenizer results played no role in these
caps. They were chosen solely from the pre-model source diagnostic.
