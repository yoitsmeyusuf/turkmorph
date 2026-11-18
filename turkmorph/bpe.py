"""Minimal BPE for the LEX namespace (root remnants + foreign words).

Trained on *suffix-stripped stems* and foreign words, never on inflected
Turkish words, so suffix strings like 'lar'/'dan' are not learned as merges
because of their frequency as suffixes.
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

START = "▁"
_LOWER = "abcçdefgğhıijklmnoöprsştuüvyzqwxâîûñáéíóú"
# uppercase too: mixed-case words (iPhone) go to LEX verbatim, case kept in pieces
BASE_ALPHABET = set(_LOWER + "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWXÂÎÛÑÁÉÍÓÚ" + START)


class _Rev:
    """Reverse string order inside a min-heap: ties go to the larger pair."""
    __slots__ = ("p",)

    def __init__(self, p):
        self.p = p

    def __lt__(self, o):
        return self.p > o.p

    def __eq__(self, o):
        return self.p == o.p


class LexBPE:
    def __init__(self):
        self.merges: list[tuple[str, str]] = []
        self.ranks: dict[tuple[str, str], int] = {}
        self.alphabet = set(BASE_ALPHABET)
        self.vocab: set[str] = set(BASE_ALPHABET)

    def train(self, words, num_merges=400, min_count=2):
        """Merge the most frequent pair (ties: larger pair) until num_merges or
        the best pair is rarer than min_count. Incremental pair counts + lazy heap."""
        corpus = Counter()
        for w in words:
            self.alphabet.update(w)
            corpus[START + w] += 1
        self.vocab = set(self.alphabet)
        syms = [list(w) for w in corpus]
        cnts = list(corpus.values())
        pairs: Counter = Counter()
        where: dict[tuple, set] = defaultdict(set)
        for i, (sy, c) in enumerate(zip(syms, cnts)):
            for p in zip(sy, sy[1:]):
                pairs[p] += c
                where[p].add(i)
        heap = [(-c, _Rev(p)) for p, c in pairs.items()]
        heapq.heapify(heap)
        while len(self.merges) < num_merges and heap:
            negc, rp = heapq.heappop(heap)
            best, cnt = rp.p, -negc
            if pairs.get(best, 0) != cnt:
                continue                                  # stale entry
            if cnt < min_count:
                break
            self.ranks[best] = len(self.merges)
            self.merges.append(best)
            self.vocab.add(best[0] + best[1])
            touched = set()
            for i in where.pop(best, ()):
                old, c = syms[i], cnts[i]
                new = _merge(old, best)
                if new == old:
                    continue
                for p in zip(old, old[1:]):
                    pairs[p] -= c
                    touched.add(p)
                for p in zip(new, new[1:]):
                    pairs[p] += c
                    where[p].add(i)
                    touched.add(p)
                syms[i] = new
            for p in touched:
                if pairs[p] > 0:
                    heapq.heappush(heap, (-pairs[p], _Rev(p)))
                else:
                    pairs.pop(p, None)
        return self

    def encode(self, word: str, start: bool = True) -> list[str]:
        """-> list of 'L:<piece>' / 'B:<hex>' tokens."""
        sym = list((START if start else "") + word)
        while len(sym) > 1:
            cand = [(self.ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(sym, sym[1:]))]
            r, i = min(cand)
            if r == 1 << 30:
                break
            sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
        out = []
        for p in sym:
            if p in self.vocab:
                out.append("L:" + p)
            else:                                   # unknown char -> bytes
                out.extend(f"B:{b:02x}" for b in p.encode("utf-8"))
        return out

    def save(self, path):
        Path(path).write_text(json.dumps({"alphabet": sorted(self.alphabet), "merges": self.merges},
                                         ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "LexBPE":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        bpe = cls()
        bpe.alphabet |= set(d["alphabet"])
        bpe.vocab = set(bpe.alphabet)
        for a, b in d["merges"]:
            bpe.ranks[(a, b)] = len(bpe.merges)
            bpe.merges.append((a, b))
            bpe.vocab.add(a + b)
        return bpe


def _merge(sym, pair):
    a, b = pair
    out, i = [], 0
    while i < len(sym):
        if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return out
