# v11-tokenizer — Python bindings

Pure-Rust knowledge-first tokenizer, exposed as a Python module via PyO3.

## Install

```bash
pip install v11-tokenizer
```

The distribution is `v11-tokenizer`; the import is `v11`. Wheels are built
against PyO3's stable ABI, so one wheel per platform covers every Python >= 3.9
— Linux (x86_64/aarch64), macOS (Intel/Apple silicon) and Windows x64. No Rust
toolchain needed. (Use 0.1.1 or newer: 0.1.0 shipped only a macOS arm64 / cp312
wheel, so everywhere else pip fell back to building the sdist.)

For development, from source:

```bash
cd v-tokenizers/v11/python
maturin develop --release
```

Then from any Python script in the same venv:

```python
import v11

tok = v11.Tokenizer.from_file("../v11/artifacts/v11.vocab.bin")
print(tok)                                         # <v11.Tokenizer vocab_size=71260>
print(tok.vocab_size)                              # 71260
print(tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id)  # 0 1 2 3

ids = tok.encode("def fibonacci(n):")
pieces = tok.encode_pieces("def fibonacci(n):")
text = tok.decode(ids)
```

## API

| method | returns | notes |
|---|---|---|
| `Tokenizer.from_file(path)` | `Tokenizer` | Load from a `.vocab.bin` file |
| `tok.encode(text)` | `list[int]` | Tokenize to ids |
| `tok.encode_pieces(text)` | `list[str]` | Tokenize to piece strings |
| `tok.decode(ids)` | `str` | Reverse to text |
| `tok.id_to_piece(id)` | `str \| None` | Look up a single id |
| `tok.piece_to_id(piece)` | `int \| None` | Look up a single piece |
| `tok.vocab_size` | `int` | Total pieces |
| `tok.pad_id`, `unk_id`, `bos_id`, `eos_id` | `int` | Special token ids |
| `len(tok)` | `int` | Same as `vocab_size` |

## Performance

Encode throughput on a single thread (release build):

```
~9.3M tokens/sec
~28 MB/sec
```

That's roughly 10-40× faster than the Python `sentencepiece` package
on comparable inputs, because the longest-match is a single
Aho-Corasick pass with no C++ FFI hops.

## Build a wheel for distribution

```bash
maturin build --release
ls target/wheels/
# v11_tokenizer-0.1.2-cp39-abi3-macosx_11_0_arm64.whl
```

The `cp39-abi3` tag is the stable-ABI build: that one file installs on any
CPython >= 3.9, whatever interpreter built it. Releases produce one per
platform via the matrix in `.github/workflows/publish.yml`.

## License

Apache 2.0
