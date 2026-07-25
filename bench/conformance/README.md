# Tokenizer conformance corpus

`cases.jsonl` contains stable adversarial cases. `generate_cases.py` adds a
deterministic random-Unicode lane, and `run_conformance.py` checks:

- exact encode/decode round-trip;
- offset validity and input coverage;
- a diagnostic inventory of failures across independent streaming-style
  chunk splits, including reconstructed text and token-ID mismatches (reported
  separately because a real streaming tokenizer must carry boundary state);
- token-ID parity with `transformers.AutoTokenizer`;
- optional token-ID and decode parity with the Rust CLI and installed Python
  binding.

Run the artifact-only checks:

```bash
python3 bench/conformance/run_conformance.py --auto-tokenizer
```

Compare the current Rust build:

```bash
cargo build -p v11-cli
python3 bench/conformance/run_conformance.py \
  --auto-tokenizer \
  --rust-cli target/debug/v11
```

The Python binding is opt-in because a globally installed wheel may be older
than the source checkout:

```bash
python3 bench/conformance/run_conformance.py --python-binding
```

Check `python_binding_version` in the report. The frozen whitespace-exact run
uses v11 binding 0.1.2; an older installed wheel is not evidence about the
newer canonical artifacts.

Run the separately named whitespace-exact TOK-2 incumbent across Hugging
Face, Transformers, and Rust:

```bash
cargo build -p v11-ws-exact
python3 bench/conformance/run_conformance.py \
  --ws-exact \
  --auto-tokenizer \
  --rust-cli target/debug/v11-ws-exact
```

The Rust adapter's batch diagnostic command loads the 71K vocabulary once;
it does not define a public streaming protocol.

Literal special-token cases remain in the corpus but carry
`expect_decode_roundtrip=false`: decoders commonly expose an explicit
skip/preserve-specials policy, so their raw-text round-trip is not treated as
a universal invariant. Their token-ID parity is also policy-dependent because
Hugging Face recognizes registered special-looking substrings before model
tokenization.

The leading-space case intentionally remains a strict round-trip check. At the
time this corpus was added it exposed a real limitation in v11's canonical
`Metaspace(prepend_scheme="always")` artifact: a leading literal space is
indistinguishable from the synthetic metaspace prefix and is lost on decode.
The full TOK-2 whole-input and stateful byte-streaming requirements are pinned
in `v12/STREAMING_CONTRACT.md`; this suite now includes each of its minimal
whitespace cases.
