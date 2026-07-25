# Whole-input and streaming tokenizer contracts

Status: contract pinned for TOK-2 preparation on 2026-07-25. The stateful
streaming API is specified here but not yet implemented.

## Whole-input contract

Every tokenizer admitted to TOK-2 must satisfy, for every valid UTF-8 string:

```text
decode(encode(x)) == x
```

The conformance corpus must include at least:

```text
"x"
" x"
"  x"
"\tx"
"\n x"
""
```

as well as CRLF, combining characters, NFC/NFD variants, zero-width
characters, emoji sequences, unusual scripts, embedded NUL, special-looking
strings under an explicit special-token policy, and arbitrary Unicode.

Published v11 does **not** currently satisfy this contract: its canonical
`Metaspace(prepend_scheme="always")` path drops a literal leading space. The
published artifact remains immutable. TOK-2 cannot silently patch it in place;
before the experiment runs, the knowledge-rich arm needs either:

- a separately named, hashed whitespace-exact v11.1 tokenizer revision; or
- a separately named, hashed experiment adapter with identical Rust, Python,
  Hugging Face, and model-training behavior.

Whichever route is chosen becomes a distinct arm artifact. The protocol and
results must not label it as byte-identical to published v11.

## Stateless chunking is not a supported contract

In general:

```text
encode(a) + encode(b) != encode(a + b)
```

Subword decisions may cross an arbitrary chunk boundary; beginning-of-input
Metaspace behavior and whitespace boundaries add further state. TOK-2 must
therefore consume complete documents, or use the stateful interface below.
It must never pass arbitrary storage/network chunks to the whole-input encoder
as if each chunk were a complete document.

## Stateful streaming contract

The streaming API is defined over bytes so split UTF-8 sequences are handled
explicitly:

```text
(tokens_a, state_a) = encode_chunk(initial_state, bytes_a)
(tokens_b, state_b) = encode_chunk(state_a, bytes_b)
tokens_tail          = finish(state_b)

tokens_a + tokens_b + tokens_tail == encode(bytes_a + bytes_b)
```

This equality must hold for every byte split of every conformance case. Decode
of the concatenated token sequence must reproduce the original bytes.

`encode_chunk` may retain input and emit no token until a decision is
irrevocable. State requires at least:

- beginning-of-stream/finalized state;
- incomplete UTF-8 bytes (up to three pending bytes);
- pending normalization state, even when the pinned normalizer is identity;
- whether and what kind of whitespace ended the committed prefix;
- the unfinished pre-tokenization segment;
- enough algorithm lookback to prevent a later byte changing Unigram/BPE
  segmentation (bounded by the pinned maximum piece length).

Carrying only a `previous_was_whitespace` bit is insufficient: a chunk can
split inside an identifier or any multi-character vocabulary piece.

## Training-harness rule

The frozen raw stream is a sequence of complete, hashed documents. Accounting
uses each document's original UTF-8 byte length. The default harness calls the
whole-input encoder exactly once per document. A future streaming loader may
replace it only after every possible split in the conformance corpus passes
the stateful equality above for every tokenizer arm.
