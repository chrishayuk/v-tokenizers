"""Whitespace-exact experiment adapters over immutable v11 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


@dataclass(frozen=True)
class ExactEncoding:
    ids: list[int]
    offsets: list[tuple[int, int]]
    tokens: list[str]


class WhitespaceExactV11:
    """Wrap a `tokenizers.Tokenizer` without changing its vocabulary."""

    adapter_id = "v11-ws-exact"

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.space_byte_id = tokenizer.token_to_id("<0x20>")
        if self.space_byte_id is None:
            raise ValueError("v11-ws-exact requires the existing <0x20> byte-fallback row")

    @classmethod
    def from_file(cls, path: str | Path) -> "WhitespaceExactV11":
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> ExactEncoding:
        base = self.tokenizer.encode(text)
        if not text.startswith(" "):
            return ExactEncoding(
                ids=list(base.ids),
                offsets=list(base.offsets),
                tokens=list(base.tokens),
            )
        return ExactEncoding(
            ids=[self.space_byte_id, *base.ids],
            offsets=[(0, 1), *base.offsets],
            tokens=["<0x20>", *base.tokens],
        )

    def encode_ids(self, text: str) -> list[int]:
        return self.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if ids and ids[0] == self.space_byte_id:
            return " " + self.tokenizer.decode(
                ids[1:], skip_special_tokens=skip_special_tokens
            )
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


class WhitespaceExactAutoTokenizer:
    """Equivalent adapter for a local Transformers fast-tokenizer backend."""

    adapter_id = "v11-ws-exact-transformers"

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.space_byte_id = tokenizer.convert_tokens_to_ids("<0x20>")
        if self.space_byte_id is None or self.space_byte_id == tokenizer.unk_token_id:
            raise ValueError("v11-ws-exact requires the existing <0x20> byte-fallback row")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "WhitespaceExactAutoTokenizer":
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(path, local_files_only=True))

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str) -> ExactEncoding:
        base = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(base["input_ids"])
        offsets = list(base["offset_mapping"])
        tokens = list(self.tokenizer.convert_ids_to_tokens(ids))
        if text.startswith(" "):
            ids.insert(0, self.space_byte_id)
            offsets.insert(0, (0, 1))
            tokens.insert(0, "<0x20>")
        return ExactEncoding(ids=ids, offsets=offsets, tokens=tokens)

    def encode_ids(self, text: str) -> list[int]:
        return self.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        prefix = " " if ids and ids[0] == self.space_byte_id else ""
        suffix_ids = ids[1:] if prefix else ids
        return prefix + self.tokenizer.decode(
            suffix_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
