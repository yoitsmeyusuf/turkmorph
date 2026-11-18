"""Lossless merge layer over morpheme tokens (design doc: "Kayıpsız birleştirme katmanı").

Frequent morpheme sequences inside a word get one id, BPE-style, but a merged
token is just a named tuple of base tokens: M:S:PL S:P3SG = (S:PL, S:P3SG).
Decoding expands it and runs the normal generator, so losslessness is untouched,
and a model can initialise / tie a merged embedding to the sum of its parts.

Modes:
  "sfx"   only suffix + suffix (-lArH, -DAn+ki ...): roots stay separate ids
  "full"  also root + suffix (R:ol#V + S:PART_DHK): approaches word-level ids
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

PREFIX = "M:"


def components(tok: str) -> list[str]:
    return tok[len(PREFIX):].split(" ") if tok.startswith(PREFIX) else [tok]


def _combine(a: str, b: str) -> str:
    return PREFIX + " ".join(components(a) + components(b))


def _suffix_only(tok: str) -> bool:
    return all(c.startswith("S:") for c in components(tok))


def _can_start(tok: str) -> bool:
    return tok.startswith(("R:", "S:", PREFIX))


def eligible(a: str, b: str, mode: str) -> bool:
    if not _suffix_only(b):
        return False
    if mode == "sfx":
        return _suffix_only(a)
    return _can_start(a) and (a.startswith(("R:", "S:")) or components(a)[0][:2] in ("R:", "S:"))


def units(tokens: list[str]):
    """Split a token list into (start, end) runs: a root/suffix followed by suffixes."""
    i, n = 0, len(tokens)
    while i < n:
        j = i + 1
        if _can_start(tokens[i]):
            while j < n and _suffix_only(tokens[j]):
                j += 1
        yield i, j
        i = j


class _Rev:
    __slots__ = ("p",)

    def __init__(self, p):
        self.p = p

    def __lt__(self, o):
        return self.p > o.p

    def __eq__(self, o):
        return self.p == o.p


class MorphMerge:
    def __init__(self, merges=(), mode: str = "sfx"):
        self.mode = mode
        self.merges: list[tuple[str, str]] = list(merges)
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab = [_combine(a, b) for a, b in self.merges]

    # ---------------------------------------------------------------- train --
    @classmethod
    def train(cls, unit_counts: Counter, num_merges: int, mode: str = "sfx",
              min_count: int = 2) -> "MorphMerge":
        """unit_counts: Counter over tuples of base tokens (one tuple = one unit)."""
        syms = [list(u) for u in unit_counts]
        cnts = list(unit_counts.values())
        pairs: Counter = Counter()
        where: dict[tuple, set] = defaultdict(set)
        for i, (sy, c) in enumerate(zip(syms, cnts)):
            for p in zip(sy, sy[1:]):
                if eligible(*p, mode):
                    pairs[p] += c
                    where[p].add(i)
        heap = [(-c, _Rev(p)) for p, c in pairs.items()]
        heapq.heapify(heap)
        merges = []
        while len(merges) < num_merges and heap:
            negc, rp = heapq.heappop(heap)
            best, cnt = rp.p, -negc
            if pairs.get(best, 0) != cnt:
                continue
            if cnt < min_count:
                break
            merges.append(best)
            new_tok = _combine(*best)
            touched = set()
            for i in where.pop(best, ()):
                old, c = syms[i], cnts[i]
                new = _merge(old, best, new_tok)
                if new == old:
                    continue
                for p in zip(old, old[1:]):
                    if eligible(*p, mode):
                        pairs[p] -= c
                        touched.add(p)
                for p in zip(new, new[1:]):
                    if eligible(*p, mode):
                        pairs[p] += c
                        where[p].add(i)
                        touched.add(p)
                syms[i] = new
            for p in touched:
                if pairs[p] > 0:
                    heapq.heappush(heap, (-pairs[p], _Rev(p)))
                else:
                    pairs.pop(p, None)
        return cls(merges, mode)

    # ---------------------------------------------------------------- apply --
    def apply(self, tokens: list[str]) -> list[str]:
        out = []
        for i, j in units(tokens):
            sym = tokens[i:j]
            while len(sym) > 1:
                cand = [(self.ranks.get((a, b), 1 << 30), k) for k, (a, b) in enumerate(zip(sym, sym[1:]))]
                r, k = min(cand)
                if r == 1 << 30:
                    break
                sym = sym[:k] + [_combine(sym[k], sym[k + 1])] + sym[k + 2:]
            out.extend(sym)
        return out

    @staticmethod
    def expand(tokens: list[str]) -> list[str]:
        return [c for t in tokens for c in components(t)]

    def save(self, path):
        Path(path).write_text(json.dumps({"mode": self.mode, "merges": self.merges},
                                         ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "MorphMerge":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in d["merges"]], d["mode"])


def _merge(sym, pair, new_tok):
    a, b = pair
    out, i = [], 0
    while i < len(sym):
        if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
            out.append(new_tok)
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return out
