"""Word-level language id: Turkish vs foreign, character n-gram Naive Bayes.

Used as the foreign-word gate of the encode cascade: a word judged foreign is
never given a guessed Turkish stem + suffixes (Avrupa -> avrup+DAT, meetings ->
meeting+S?); it goes to LEX whole (unless a known root analysis exists).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


def _grams(word: str, orders=(1, 2, 3, 4)):
    w = f"<{word}>"
    return [w[i:i + n] for n in orders for i in range(len(w) - n + 1)]


class CharNgramLangId:
    def __init__(self, lp_tr: dict, lp_fr: dict, unk_tr: float, unk_fr: float,
                 threshold: float = 0.0):
        self.lp_tr, self.lp_fr = lp_tr, lp_fr
        self.unk_tr, self.unk_fr = unk_tr, unk_fr
        self.threshold = threshold

    @classmethod
    def train(cls, tr_words, foreign_words, min_count=2) -> "CharNgramLangId":
        """tr_words / foreign_words: iterables of word types (each counted once)."""
        def counts(words):
            c = Counter()
            for w in set(words):
                c.update(_grams(w))
            return c
        ct, cf = counts(tr_words), counts(foreign_words)
        vocab = {g for g in ct.keys() | cf.keys() if ct[g] + cf[g] >= min_count}
        nt, nf, v = sum(ct.values()), sum(cf.values()), len(vocab) + 1
        lp_tr = {g: math.log((ct[g] + 1) / (nt + v)) for g in vocab}
        lp_fr = {g: math.log((cf[g] + 1) / (nf + v)) for g in vocab}
        return cls(lp_tr, lp_fr, math.log(1 / (nt + v)), math.log(1 / (nf + v)))

    def score(self, word: str) -> float:
        """Mean per-n-gram log-likelihood ratio, foreign vs Turkish (> 0: foreign)."""
        gs = _grams(word)
        s = sum(self.lp_fr.get(g, self.unk_fr) - self.lp_tr.get(g, self.unk_tr) for g in gs)
        return s / len(gs)

    def is_foreign(self, word: str) -> bool:
        return len(word) >= 3 and self.score(word) > self.threshold

    def calibrate(self, tr_heldout, max_tr_false_positive=0.01):
        """Smallest threshold that flags at most the given share of Turkish words."""
        scores = sorted((self.score(w) for w in tr_heldout), reverse=True)
        k = int(len(scores) * max_tr_false_positive)
        self.threshold = scores[k] if scores else 0.0
        return self.threshold

    def save(self, path):
        Path(path).write_text(json.dumps({
            "lp_tr": self.lp_tr, "lp_fr": self.lp_fr, "unk_tr": self.unk_tr,
            "unk_fr": self.unk_fr, "threshold": self.threshold}, ensure_ascii=False),
            encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CharNgramLangId":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["lp_tr"], d["lp_fr"], d["unk_tr"], d["unk_fr"], d["threshold"])
