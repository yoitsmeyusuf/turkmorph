"""Lossless handling of misspelled words: analyse the intended word, record the typo.

A misspelled word would otherwise fall to LEX/BPE pieces. Instead it is encoded as
the analysis of the word it most likely means plus a tiny, exact description of
how the written form differs, so decode still returns the text byte for byte:

  cocuklarimiz  -> <ascii> R:çocuk#N S:PL S:P1PL        (written without Turkish letters)
  tesbit        -> R:tespit#N E:sub:3:b                 (one letter differs: one token)

Edit ops act on the generated lowercase surface c, before casing:
  del p      written = c[:p] + c[p+1:]
  ins p ch   written = c[:p] + ch + c[p:]
  sub p ch   written = c[:p] + ch + c[p+1:]
  swap p     written = c[:p] + c[p+1] + c[p] + c[p+2:]

Guards (precision first; a wrong "correction" is still lossless but misleads the model):
only words with no known analysis (ASCII spelling also when the Turkish spelling is
DOMINANT times more frequent: vardi -> vardı, not the rare root 'varit'), >= MIN_LEN
letters for edits (4-letter words need SHORT_RATIO), Damerau distance 1, the intended
word >= RATIO times more frequent than the written one, not foreign, not a word that
the corpus mostly writes capitalized (a name).
"""
from __future__ import annotations

from pathlib import Path

ASCII = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")
EDIT_CHARS = "abcçdefgğhıijklmnoöprsştuüvyzqwxâîû"
MAX_POS = 32
EDIT_OPS = ("del", "ins", "sub", "swap")
EDIT_TOKENS = ([f"E:{o}:{p}" for o in ("del", "swap") for p in range(MAX_POS)]
               + [f"E:{o}:{p}:{c}" for o in ("ins", "sub") for p in range(MAX_POS) for c in EDIT_CHARS])


VOWELS = set("aeıioöuüâîû")
# letters Turkish writers swap for each other (no Turkish keyboard, ı/i confusion)
TR_PAIRS = {frozenset(p) for p in ("cç", "gğ", "ıi", "oö", "sş", "uü")}
_ROWS = ["qwertyuıopğü", "asdfghjklşi", "zxcvbnmöç"]
KEY_NEIGHBOURS: dict[str, set] = {}
for _r, _row in enumerate(_ROWS):
    for _c, _ch in enumerate(_row):
        near = {_row[j] for j in (_c - 1, _c + 1) if 0 <= j < len(_row)}
        for _rr in (_r - 1, _r + 1):
            if 0 <= _rr < len(_ROWS):
                near |= {_ROWS[_rr][j] for j in (_c - 1, _c, _c + 1) if 0 <= j < len(_ROWS[_rr])}
        KEY_NEIGHBOURS[_ch] = near


def op_prior(op, correct: str, written: str) -> float:
    """How plausible a typo is, beyond word frequency: dropping a vowel (geçş for
    geçiş), hitting a neighbouring key, confusing ı/i or s/ş, doubling a letter."""
    kind, p, ch = op
    if kind == "del":                                  # writer dropped correct[p]
        return 4.0 if correct[p] in VOWELS else 2.0
    if kind == "ins":                                  # writer added ch: doubled key, or a
        near = written[p - 1:p] + written[p + 1:p + 2]  # neighbour of the keys around it
        if ch in near:
            return 3.0
        return 1.5 if any(ch in KEY_NEIGHBOURS.get(n, ()) for n in near) else 0.3
    if kind == "sub":                                  # writer typed ch for correct[p]
        if frozenset((correct[p], ch)) in TR_PAIRS:
            return 4.0
        return 3.0 if ch in KEY_NEIGHBOURS.get(correct[p], ()) else 1.0
    return 2.5                                         # swap


def asciify(s: str) -> str:
    return s.translate(ASCII)


def apply_edit(c: str, op) -> str:
    kind, p, ch = op
    if kind == "del":
        return c[:p] + c[p + 1:]
    if kind == "ins":
        return c[:p] + ch + c[p:]
    if kind == "sub":
        return c[:p] + ch + c[p + 1:]
    return c[:p] + c[p + 1:p + 2] + c[p:p + 1] + c[p + 2:]          # swap


def edit_op(correct: str, written: str):
    """The single op turning `correct` into `written`, or None (Damerau distance != 1)."""
    lc, lw = len(correct), len(written)
    i = 0
    while i < min(lc, lw) and correct[i] == written[i]:
        i += 1
    if lw == lc - 1 and correct[:i] + correct[i + 1:] == written:
        return ("del", i, "")
    if lw == lc + 1 and correct[:i] + written[i] + correct[i:] == written:
        return ("ins", i, written[i])
    if lw == lc and i < lc:
        if correct[:i] + written[i] + correct[i + 1:] == written:
            return ("sub", i, written[i])
        if i + 1 < lc and correct[:i] + correct[i + 1] + correct[i] + correct[i + 2:] == written:
            return ("swap", i, "")
    return None


def op_tokens(op) -> list[str]:
    kind, p, ch = op
    return [f"E:{kind}:{p}" + (f":{ch}" if ch else "")]


def parse_op(tok: str):
    """'E:sub:3:b' -> ('sub', 3, 'b'); None if malformed."""
    parts = tok.split(":")
    if len(parts) not in (3, 4) or parts[1] not in EDIT_OPS or not parts[2].isdigit():
        return None
    return parts[1], int(parts[2]), parts[3] if len(parts) == 4 else ""


class Normalizer:
    MIN_LEN = 4
    RATIO = 20
    ED2_MIN_LEN = 6              # two-letter typos: geliyrom -> geliyorum
    ED2_RATIO = 100
    ED2_INDEX_MIN_COUNT = 100
    SHORT_RATIO = 50             # 4-letter words: deil -> değil, kaar -> kadar
    ASCII_RATIO = 5
    DOMINANT = 50                # override a known (rare) analysis: vardi -> vardı

    def __init__(self, counts: dict[str, int], known: set[str], proper=None, foreign=None):
        self.counts = counts                      # corpus word-type frequencies
        self.known = known                        # types with a known-root analysis
        # lowercased word -> times seen capitalized mid-sentence / before an apostrophe
        self.proper = proper if isinstance(proper, dict) else {w: 1 << 30 for w in (proper or ())}
        self.foreign = foreign
        self.ascii_map: dict[str, str] = {}
        best: dict[str, tuple] = {}                   # most frequent, ties: smallest word
        for w in known:                               # (deterministic: set order is hash-seeded)
            a = asciify(w)
            if a != w:
                key = (-counts.get(w, 0), w)
                if a not in best or key < best[a]:
                    best[a] = key
                    self.ascii_map[a] = w

    FOREIGN_OVERRIDE = 500       # a foreign-looking word still gets a far more frequent Turkish target

    def _is_name(self, w: str) -> bool:
        return self.proper.get(w, 0) * 2 >= max(1, self.counts.get(w, 0))

    def _blocked(self, w: str) -> bool:
        """A name (mostly written capitalized) or a foreign word: leave it alone."""
        return self._is_name(w) or bool(self.foreign and self.foreign(w))

    def ascii_target(self, w: str, ratio: float | None = None) -> str | None:
        """cocuk -> çocuk, if the word is all-ASCII and its Turkish spelling dominates."""
        t = self.ascii_map.get(w)
        # no foreign gate: ASCII Turkish (calisma, bagli) looks foreign to the char model,
        # and a dominant Turkish spelling is strong evidence on its own
        if t is None or asciify(w) != w or self._is_name(w):
            return None
        need = (ratio or self.ASCII_RATIO) * max(1, self.counts.get(w, 0))
        return t if self.counts.get(t, 0) >= need else None

    def dominant_ascii(self, w: str) -> str | None:
        """ASCII spelling of a word whose own analysis is a rare homograph (vardi, icinde)."""
        return self.ascii_target(w, self.DOMINANT)

    def edit_target(self, w: str):
        """(intended word, [ops]) for a one- or two-letter typo, or None. Candidates are
        ranked by corpus frequency x typo plausibility (op_prior)."""
        if len(w) < self.MIN_LEN or self._is_name(w) or any(c not in EDIT_CHARS for c in w):
            return None
        own = max(1, self.counts.get(w, 0))
        need = (self.SHORT_RATIO if len(w) == self.MIN_LEN else self.RATIO) * own
        if self.foreign and self.foreign(w):           # terchi, zengni look foreign; 'mode' must not
            need = max(need, self.FOREIGN_OVERRIDE * own)  # become 'moda
        best = None
        for cand in set(_neighbours(w)):
            c = self.counts.get(cand, 0)
            if c >= need and cand in self.known:
                op = edit_op(cand, w)
                if op is not None and op[1] < MAX_POS:
                    score = (c * op_prior(op, cand, w), cand)
                    if best is None or score > best[0]:
                        best = (score, cand, [op])
        if best is None and len(w) >= self.ED2_MIN_LEN:
            best = self._edit2(w, self.ED2_RATIO * own)
        return (best[1], best[2]) if best else None

    def _edit2(self, w, need):
        """Two edits, via a symmetric-delete index of frequent words."""
        if not hasattr(self, "_del_index"):
            idx: dict[str, list] = {}
            for v in self.known:
                # fixed threshold: the index must not depend on which word asked first
                if len(v) >= self.ED2_MIN_LEN and self.counts.get(v, 0) >= self.ED2_INDEX_MIN_COUNT:
                    for d in {v, *(v[:i] + v[i + 1:] for i in range(len(v)))}:
                        idx.setdefault(d, []).append(v)
            self._del_index = idx
        # w itself, w minus / plus / with a different letter, w with a swap: against the
        # index (words and their 1-deletions) this reaches drop+drop, sub+drop, swap+drop, ...
        keys = {w, *(w[:i] + w[i + 1:] for i in range(len(w))),
                *(w[:i] + c + w[i:] for i in range(len(w) + 1) for c in EDIT_CHARS),
                *(w[:i] + c + w[i + 1:] for i in range(len(w)) for c in EDIT_CHARS),   # sub + drop: geliyrom
                *(w[:i] + w[i + 1] + w[i] + w[i + 2:] for i in range(len(w) - 1))}     # swap + drop
        best = None
        for cand in {c for k in keys for c in self._del_index.get(k, ())}:
            c = self.counts.get(cand, 0)
            if c < need or cand == w:
                continue
            for x in _neighbours(cand):                      # correct -> x -> written
                op2 = edit_op(x, w)
                if op2 is None:
                    continue
                op1 = edit_op(cand, x)
                if op1 and op1[1] < MAX_POS and op2[1] < MAX_POS:
                    score = (c * op_prior(op1, cand, x) * op_prior(op2, x, w), cand)
                    if best is None or score > best[0]:
                        best = (score, cand, [op1, op2])
        return best

    # ------------------------------------------------------------ files --
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for w, c in self.counts.items():
                f.write(f"{w}\t{c}\t{int(w in self.known)}\n")

    @classmethod
    def load(cls, path, proper_path=None, foreign=None) -> "Normalizer":
        counts, known = {}, set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            w, c, k = line.split("\t")
            counts[w] = int(c)
            if k == "1":
                known.add(w)
        proper = {}
        if proper_path and Path(proper_path).exists():
            for line in Path(proper_path).read_text(encoding="utf-8").splitlines():
                w, _, c = line.partition("\t")
                proper[w] = int(c) if c else 1 << 30
        return cls(counts, known, proper, foreign)


def _neighbours(w: str):
    """All strings at Damerau distance 1 (corrections the writer may have meant)."""
    n = len(w)
    for i in range(n):
        yield w[:i] + w[i + 1:]                                # writer added a letter
        if i + 1 < n:
            yield w[:i] + w[i + 1] + w[i] + w[i + 2:]          # swapped
        for c in EDIT_CHARS:
            if c != w[i]:
                yield w[:i] + c + w[i + 1:]                    # wrong letter
    for i in range(n + 1):
        for c in EDIT_CHARS:
            yield w[:i] + c + w[i:]                            # writer dropped a letter
