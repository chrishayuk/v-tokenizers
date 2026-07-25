---
license: apache-2.0
language: [en]
tags: [tokenizer, evaluation, round-trip]
---

# v11 tokenizer gate corpus

{n_files} files of prose and source code, hand-assembled to exercise a
tokenizer where real text does: indentation, tabs, operators, Greek letters,
maths symbols, and non-ASCII characters.

This is the corpus the [v11 tokenizer](https://huggingface.co/chrishayuk/v11-tokenizer)
round-trip gate runs against. Current result: **0 UNK, 32/32 files round-trip
exactly**.

It exists because an earlier check against three samples of plain English prose
passed cleanly while the tokenizer was in fact dropping 662 characters on real
input -- it never contained a tab. A gate corpus is only worth what its hardest
file is worth.

Registered in the chuk-datasets catalog as `v11/corpus`; its content identity is
verified against that catalog on every CI run, and every file here is verified
by sha256 against the source tree at publish time.

## Layout

```
prose/   science, history, tech, geography
code/    c, go, java, javascript, python, rust, typescript
```

## Not training data

v11's vocabulary is assembled from tree-sitter grammars, WordNet, Wikidata and
hand-curated lists -- it is not trained on a corpus at all. This is a
benchmark and spot-check sample.
