"""The Rust runtime: same tokens as TurkMorphTokenizer (full preset, deterministic
mode), bit for bit, ~100x faster. Included in the wheel; in a checkout build it
with `bash scripts/build_fast.sh`.

    from turkmorph.fast import fast_tokenizer
    tok = fast_tokenizer()
    ids = tok.encode_ids("Zeytinyağlarını aldım.")
    tok.decode_ids(ids)
    tok.encode_batch(texts, threads=6)        # list of id lists, GIL released

Not covered here: context mode (B) and MorphMerge tokens; use the Python classes.
"""
from __future__ import annotations

from pathlib import Path

from ._fast import FastTokenizer
from ._paths import default_data_dir

DATA = default_data_dir()


class _Vocab:
    """The part of tokenizer.Vocab that callers use: id / tok / len."""

    def __init__(self, ft: FastTokenizer):
        self._ft = ft
        self.toks = ft.vocab()
        self.index = {t: i for i, t in enumerate(self.toks)}

    def id(self, t): return self.index[t]
    def tok(self, i): return self.toks[i]
    def __len__(self): return len(self.toks)


class FastTurkMorph:
    def __init__(self, data_dir=DATA):
        self._ft = FastTokenizer(str(data_dir))
        self.vocab = _Vocab(self._ft)

    def encode(self, text: str) -> list[str]: return self._ft.encode(text)
    def encode_ids(self, text: str) -> list[int]: return self._ft.encode_ids(text)
    def encode_batch(self, texts, threads: int = 6) -> list[list[int]]:
        return self._ft.encode_batch(list(texts), threads)
    def encode_lex(self, word: str) -> list[int]: return self._ft.encode_lex(word)
    def decode(self, toks: list[str]) -> str: return self._ft.decode(list(toks))
    def decode_ids(self, ids) -> str: return self._ft.decode_ids(list(ids))


def fast_tokenizer(data_dir=DATA) -> FastTurkMorph:
    return FastTurkMorph(data_dir)
