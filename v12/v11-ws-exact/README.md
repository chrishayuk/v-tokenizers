# v11-ws-exact

`v11-ws-exact` is the separately named TOK-2 experiment adapter over the
newer canonical v11 artifacts. It does not alter or replace published v11,
does not use the older 71,261-row SentencePiece model, and adds no model rows.

The canonical 71,260-row tokenizer consumes one real leading ASCII space as
its synthetic Metaspace prefix. For a complete input beginning with U+0020,
the adapter emits the existing `<0x20>` row (ID 36) before the canonical
encoding of the complete input. Its decoder restores that one byte. Inputs
without a leading ASCII space retain exactly their canonical v11 IDs.

The frozen identity, source hashes, and exact encode/decode contract are in
`../training/v11_ws_exact_manifest.json`.

```bash
cargo build -p v11-ws-exact
target/debug/v11-ws-exact \
  --model v11/artifacts/v11.vocab.bin \
  encode --json --text "  x"
```

This is a whole-document adapter. It does not claim stateless chunk
composability or implement the future stateful streaming interface.
