"""TurkMorph tokenizer: encode / decode with a by-construction lossless guarantee.

Token namespaces (string form; ids via Vocab):
  <...>     specials: <cap> <caps> <glue> <sp> <nl> <tab> <cr> <apos> <apos2> <lit> <raw>
  R:<key>   known root (lexeme + phonology class), e.g. R:kitap#N
  L:<piece> LEX BPE piece; '▁' prefix = start of a root/foreign word
  S:<id>    archiphonemic suffix, e.g. S:ABL (= -DAn)
  D:<d>     digit
  P:<c>     punctuation
  B:<hex>   byte fallback

Word grammar:  case? (R | L▁ L* B*) (apos (S+ | <lit> L*))? S*
               D+ (apos (S+ | <lit> L*))?
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .bpe import LexBPE
from .merge import PREFIX as MERGE_PREFIX
from .merge import MorphMerge
from .normalize import EDIT_TOKENS, apply_edit, asciify, op_tokens, parse_op
from .morphology import (_MARKABLE, DEFAULT_LEXICON, HAND_COSTS, MARKED_IDS, SUFFIX_LIST,
                         CostModel, Lexicon, analyze_guess, analyze_known, analyze_tail, generate,
                         is_suffix_id, suffix_base, underspecify)
from .phonology import number_reading, spelled_reading

# <capsb>: upper-case only before the apostrophe (NATO'ya, TBMM'ye)
# <ascii>: word written without Turkish letters; a one-letter typo is an E:op:pos[:char] token
# <bare>: a suffix chain with no word before it (Türkiye' nin -> <bare> S:GEN!nin)
# <fuse>: two words fused in speech (nolur = ne <fuse> olur): the left word loses its
#         last vowel, the right one a leading y/h (napıyon = ne + yapıyon, naber = ne + haber)
SPECIALS = ["<pad>", "<bos>", "<eos>", "<cap>", "<caps>", "<capsb>", "<glue>", "<sp>", "<nl>",
            "<tab>", "<cr>", "<apos>", "<apos2>", "<lit>", "<raw>", "<ascii>", "<bare>", "<fuse>"]
WORD_PREFIX = {"<glue>", "<cap>", "<caps>", "<capsb>", "<ascii>", "<bare>", "<fuse>"}
# clitics written glued to the previous word by mistake: ikisinide = ikisini de
GLUED_CLITICS = ("de", "da", "ki", "mi", "mı", "mu", "mü")
GLUED_CONJ = {"ya", "ve", "veya", "ama", "hem"}  # ya da, hem de: split even if a rare reading exists
BARE_FIRST = set(_MARKABLE) | {"LHK", "LH"}     # a detached chain: case, PL, poss, copula, 10 'luk
WS_TOK = {" ": "<sp>", "\n": "<nl>", "\t": "<tab>", "\r": "<cr>"}
WS_CHR = {v: k for k, v in WS_TOK.items()}
APOS = {"'": "<apos>", "’": "<apos2>"}
APOS_CHR = {v: k for k, v in APOS.items()}
PUNCT = set(".,;:!?()[]{}\"'-–—…/\\%&*+=<>@#$€₺_|~^`“”‘’«»")

# numbers are ASCII digits only (vocab has D:0-9); other digit scripts are kept as bytes
CHUNK = re.compile(r"([^\W\d_]+(?:['’][^\W\d_]+)?)|([0-9]+(?:['’][^\W\d_]+)?)|(\s+)|(.)", re.S)
APOS_SPLIT = re.compile(r"([^'’]+)(['’])(.+)")
FOREIGN_HINT = re.compile(r"[qwx]|th|sh|ck|ee|oo|ph|ñ|[áéíóú]")


def tr_lower(s): return s.replace("I", "ı").replace("İ", "i").lower()
def tr_upper(s): return s.replace("i", "İ").replace("ı", "I").upper()
def tr_cap(s): return tr_upper(s[:1]) + s[1:] if s else s


class TurkMorphTokenizer:
    def __init__(self, bpe: LexBPE, attested_stems: set[str] | None = None,
                 cache_size: int = 100_000, lexicon: Lexicon | None = None,
                 costs: CostModel | None = None, foreign=None, merge: MorphMerge | None = None,
                 normalizer=None):
        self.bpe = bpe
        self.attested = attested_stems or set()
        self.lex = lexicon or DEFAULT_LEXICON
        self.costs = costs or HAND_COSTS
        # optional word-level language id: foreign(word) -> True blocks the guesser
        self.foreign = foreign
        # optional lossless merge layer (frequent morpheme sequences -> one id)
        self.merge = merge
        # optional typo / ASCII-spelling normalizer (turkmorph.normalize)
        self.normalizer = normalizer
        # audited choices: word -> (root key, suffix chain) preferred among its analyses
        self.overrides: dict[str, tuple] = {}
        # underspecified suffix ids allowed in the vocabulary (P3SG|ACC); empty = off
        self.unions: list[str] = []
        self._union_set: set[str] = set()
        self.vocab = Vocab(self)
        # which encode path produced each word (counted on cache misses only)
        self.stats = {"known": 0, "clitic": 0, "fuse": 0, "bare": 0, "ascii": 0, "edit": 0,
                      "guess": 0, "lex": 0, "apos": 0, "raw": 0}
        # word-type cache: Zipf makes most of a token stream a single lookup
        self.cache: dict[str, tuple[str, ...]] = {}
        self.cache_size = cache_size

    # ------------------------------------------------------------ encode --
    def encode(self, text: str, forced: dict | None = None) -> list[str]:
        """forced: {word index: (root key, suffix chain)} chosen elsewhere (context mode)."""
        toks: list[str] = []
        chunks = list(CHUNK.finditer(text))
        implicit_space = False
        n_word = -1
        for i, m in enumerate(chunks):
            word, num, ws, other = m.groups()
            if ws is not None:
                nxt = chunks[i + 1] if i + 1 < len(chunks) else None
                # a bare number after a bare number would fuse on decode (12 34 -> 1234)
                fuses = nxt is not None and nxt.group(2) and toks[-1:] and toks[-1].startswith("D:")
                if ws == " " and toks and nxt and (nxt.group(1) or nxt.group(2)) and not fuses:
                    implicit_space = True
                    continue
                for ch in ws:
                    toks.extend([WS_TOK[ch]] if ch in WS_TOK else ["<raw>", *_bytes(ch)])
                continue
            if word is not None or num is not None:
                if toks and not implicit_space:
                    toks.append("<glue>")
                implicit_space = False
                if word:
                    n_word += 1
                    choice = forced.get(n_word) if forced else None
                    toks.extend(self._encode_forced(word, *choice) if choice else self._encode_word(word))
                else:
                    nt = self._encode_number(num)
                    toks.extend(self.merge.apply(nt) if self.merge else nt)
                continue
            implicit_space = False
            toks.extend([f"P:{other}"] if other in PUNCT else ["<raw>", *_bytes(other)])
        return toks

    def _encode_number(self, chunk):
        m = re.match(r"([0-9]+)(['’])?(.*)", chunk)
        digits, apos, tail = m.groups()
        toks = [f"D:{d}" for d in digits]
        if apos:
            toks += [APOS[apos], *self._encode_tail(number_reading(digits), tail)]
        return toks

    def _encode_tail(self, reading, tail, flags=frozenset()):
        ch = analyze_tail(reading, tail, flags)
        if ch is not None and generate(reading, flags, list(ch))[len(reading):] == tail:
            return [f"S:{s}" for s in ch]
        return ["<lit>", *self.bpe.encode(tail, start=False)]   # marked fallback

    def _encode_word(self, chunk):
        hit = self.cache.get(chunk)
        if hit is None:
            hit = self._encode_word_uncached(chunk)
            hit = tuple(self.merge.apply(hit) if self.merge else hit)
            if len(self.cache) < self.cache_size:
                self.cache[chunk] = hit
        return list(hit)

    def _encode_forced(self, chunk, key, chain):
        """The given analysis, if it generates exactly this word (else the normal path)."""
        low = tr_lower(chunk)
        case = [] if chunk == low else ["<cap>"] if chunk == tr_cap(low) else \
            ["<caps>"] if len(chunk) > 1 and chunk == tr_upper(low) else None
        if case is None or APOS_SPLIT.match(chunk) or key not in self.lex.by_key:
            return self._encode_word(chunk)
        toks = case + [f"R:{key}", *[f"S:{s}" for s in chain]]
        if self.decode(toks) != chunk:
            return self._encode_word(chunk)
        return self.merge.apply(toks) if self.merge else toks

    def _encode_word_uncached(self, chunk):
        m = APOS_SPLIT.match(chunk)
        if m:
            return self._encode_apos(chunk, *m.groups())
        low = tr_lower(chunk)
        if chunk == low:
            case = []
        elif chunk == tr_cap(low):
            case = ["<cap>"]
        elif len(chunk) > 1 and chunk == tr_upper(low):
            case = ["<caps>"]
        else:
            case = None                                       # mixed case
        if case is not None:
            for kind, cand in self._candidates(low, cased=bool(case)):
                toks = case + cand
                if self.decode(toks) == chunk:                # generate-and-verify
                    self.stats[kind] += 1
                    return toks
        self.stats["raw"] += 1
        return self.bpe.encode(chunk)                         # lossless raw LEX

    def _encode_apos(self, chunk, base, apos, tail):
        """Danimarka'dan, ANKARA'DA, NATO'ya, iPhone'u: case of the base and the
        tail are separate; the tail is a suffix chain read off the base."""
        low, low_base = tr_lower(chunk), tr_lower(base)
        head = None
        if chunk == low:
            case = []
        elif chunk == tr_cap(low):
            case = ["<cap>"]
        elif len(chunk) > 1 and chunk == tr_upper(low):
            case = ["<caps>"]
        elif tail == tr_lower(tail) and len(base) > 1 and base == tr_upper(low_base):
            case = ["<capsb>"]
        elif tail == tr_lower(tail):
            case, head = [], self.bpe.encode(base)            # mixed-case base kept verbatim
        else:
            case = None
        if case is not None:
            reading, flags = spelled_reading(low_base), frozenset()
            if head is None:
                known = [r for r in self.lex.by_form.get(low_base, [])
                         if r.pos in ("PROPER", "N", "PRON")]
                known.sort(key=lambda r: r.pos != "PROPER")
                if known:
                    r = known[0]
                    head = [f"R:{r.key}"]
                    reading, flags = r.reading or r.form, _tail_flags(r)
                else:
                    head = self.bpe.encode(low_base)
            toks = case + head + [APOS[apos], *self._encode_tail(reading, tr_lower(tail), flags)]
            if self.decode(toks) == chunk:
                self.stats["apos"] += 1
                return toks
        self.stats["raw"] += 1
        return self.bpe.encode(chunk)

    def _known(self, low, cased):
        analyses = analyze_known(low, self.lex, self.costs)
        if not cased:                                         # a lowercase word is rarely a name
            analyses.sort(key=lambda a: a.root.pos == "PROPER")
        ov = self.overrides.get(low)
        if ov is not None:                                    # frequency audit (scripts/build_overrides.py)
            analyses.sort(key=lambda a: (a.root.key, a.sfx) != ov)
        out = [[f"R:{a.root.key}", *[f"S:{s}" for s in a.sfx]] for a in analyses[:8]]
        if self._union_set and len(analyses) > 1:             # label-only ambiguity: claim no more
            u = underspecify(analyses[0], analyses[1:])       # than the word shows (kitab+ı: P3SG|ACC)
            if any("|" in s for s in u) and all("|" not in s or s in self._union_set for s in u):
                out.insert(0, [f"R:{analyses[0].root.key}", *[f"S:{s}" for s in u]])
        return out

    def set_unions(self, unions):
        self.unions = list(unions)
        self._union_set = set(self.unions)
        self.vocab = Vocab(self)
        self.cache.clear()

    def _candidates(self, low, cased=False):
        known = self._known(low, cased)
        norm = self.normalizer
        for cl in GLUED_CLITICS:                              # yada = ya da (conjunction + clitic)
            if low.endswith(cl) and low[:-len(cl)] in GLUED_CONJ:   # beats a rare reading ('yad'+DAT)
                left = self._known(low[:-len(cl)], cased)
                if left and left[0][0].endswith("#X") and len(left[0]) == 1:
                    for right in self._known(cl, False)[:1]:
                        yield "clitic", [*left[0], "<glue>", *right]
        # a lowercase word whose only analyses use a name, or whose Turkish spelling is far
        # more frequent (vardi: 'varit'+P3SG vs vardı), gets the normalized reading first
        weak = known and not cased and all(k[0].endswith("#PROPER") for k in known)
        if norm is not None and known and (weak or norm.dominant_ascii(low)):
            t = norm.ascii_target(low)
            for toks in (self._known(t, cased)[:1] if t else []):
                yield "ascii", ["<ascii>", *toks]
            if weak:
                r = norm.edit_target(low)
                for toks in (self._known(r[0], cased)[:1] if r else []):
                    if not toks[0].endswith("#PROPER"):
                        yield "edit", [*toks, *_ops(r[1])]
        for toks in known:
            yield "known", toks
        if known:
            return
        norm = self.normalizer
        t = norm.ascii_target(low) if norm is not None else None
        if t is not None:                                     # cocuk -> <ascii> çocuk (before any split:
            for toks in self._known(t, cased)[:1]:            # yaninda is yanında, not yanin+da)
                yield "ascii", ["<ascii>", *toks]
        for cl in GLUED_CLITICS:                              # ikisinide -> ikisini <glue> de
            if low.endswith(cl) and len(low) - len(cl) >= 2:
                left, right = self._known(low[:-len(cl)], cased), self._known(cl, False)
                if left and right:
                    yield "clitic", [*left[0], "<glue>", *right[0]]
        if len(low) >= 4 and low[0] == "n" and low[1] in "aeıioöuü":   # nolur, naber, napıyon
            for right_word in (low[1:], "y" + low[1:], "h" + low[1:]):
                right = self._known(right_word, False)
                if right and (right[0][0].endswith("#V") or right_word == "haber"):
                    yield "fuse", ["R:ne#PRON", "<fuse>", *right[0]]
                    break
        if 2 <= len(low) <= 7:                                # Türkiye' nin: a detached suffix chain
            ch = analyze_tail("", low)                        # (must open with a case/number/possessive/copula)
            if ch is not None and suffix_base(ch[0]) in BARE_FIRST:
                yield "bare", ["<bare>", *[f"S:{s}" for s in ch]]
        if norm is not None:
            r = norm.edit_target(low)                         # tesbit -> tespit + edit
            if r is not None:
                for toks in self._known(r[0], cased)[:1]:
                    if not toks[0].endswith("#PROPER"):       # never "correct" into a name
                        yield "edit", [*toks, *_ops(r[1])]
        foreign = self.foreign(low) if self.foreign else FOREIGN_HINT.search(low)
        if not foreign:
            for a in analyze_guess(low, self.attested)[:4]:
                yield "guess", [*self.bpe.encode(a.stem), *[f"S:{s}" for s in a.sfx]]
        yield "lex", self.bpe.encode(low)

    # ------------------------------------------------------------ decode --
    def decode(self, toks: list[str]) -> str:
        """Tokens -> text. Total on any token sequence: ungrammatical input (e.g.
        a suffix with no word, as an LM may sample) still decodes, never raises,
        except for strings that are not tokens at all."""
        if any(t.startswith(MERGE_PREFIX) for t in toks):
            toks = MorphMerge.expand(toks)
        out: list[str] = []
        w: _Word | None = None
        raw: list[int] | None = None
        prev = None

        fuse_pending = False

        def flush():
            nonlocal w, fuse_pending
            if w is not None:
                surf = w.surface()
                if fuse_pending and out:
                    surf = _fuse(out.pop(), surf)
                    fuse_pending = False
                out.append(surf)
                w = None

        def start_word():
            nonlocal w
            flush()
            w = _Word(space=bool(out) and prev not in ("<glue>", "<fuse>"))
            if w.space:
                out.append(" ")

        def cur():
            if w is None:
                start_word()
            return w

        for t in toks:
            if raw is not None and not t.startswith("B:"):
                out.append(bytes(raw).decode("utf-8", "replace"))
                raw = None
            if t in ("<cap>", "<caps>", "<capsb>"):
                start_word()
                w.case = t
            elif t in ("<ascii>", "<bare>"):
                if w is None or w.has_stem() or w.sfx:
                    start_word()
                if t == "<ascii>":
                    w.ascii = True
                else:
                    w.bare = True
            elif t.startswith("E:"):
                op = parse_op(t)
                if op is None:
                    raise ValueError(f"bad token {t}")
                cur().edit.append(op)
            elif t.startswith("R:") or (t.startswith("L:▁")):
                if w is None or w.has_stem() or w.sfx or w.bare:   # a suffixed word takes no new stem
                    start_word()
                if t.startswith("R:"):
                    if t[2:] not in self.lex.by_key:
                        raise ValueError(f"bad token {t}")
                    w.root = self.lex.by_key[t[2:]]
                else:
                    w.stem.append(t[3:])
            elif t.startswith("D:"):
                if w is None or w.kind() != "NUM" or w.apos or w.sfx:
                    start_word()
                w.digits += t[2:]
            elif t.startswith("L:"):
                w = cur()
                (w.lit if w.lit is not None else w.stem).append(t[2:])
            elif t.startswith("B:"):
                b = int(t[2:], 16)
                if raw is not None:
                    raw.append(b)
                else:
                    w = cur()
                    (w.lit if w.lit is not None else w.stem).append(bytes([b]))
            elif t.startswith("S:"):
                if not is_suffix_id(t[2:]):
                    raise ValueError(f"bad token {t}")
                cur().sfx.append(t[2:])
            elif t in APOS_CHR:
                cur().apos = APOS_CHR[t]
            elif t == "<lit>":
                cur().lit = []
            elif t == "<raw>":
                flush()
                raw = []
            elif t == "<glue>":
                flush()
            elif t == "<fuse>":
                flush()
                fuse_pending = True
            elif t in WS_CHR:
                flush()
                out.append(WS_CHR[t])
            elif t.startswith("P:"):
                flush()
                out.append(t[2:])
            elif t in ("<bos>", "<eos>", "<pad>"):
                flush()
            else:
                raise ValueError(f"bad token {t}")
            prev = t
        if raw is not None:
            out.append(bytes(raw).decode("utf-8", "replace"))
        flush()
        return "".join(out)

    # ------------------------------------------------------------ helpers --
    def encode_ids(self, text): return [self.vocab.id(t) for t in self.encode(text)]
    def decode_ids(self, ids): return self.decode([self.vocab.tok(i) for i in ids])

    # word groups for a word-level composer (design doc, "Kelime düzeyi composer"):
    # concatenating the groups gives encode(text) back, so nothing is lost.
    def encode_grouped(self, text) -> list[list[str]]:
        toks = self.encode(text)
        groups: list[list[str]] = []
        for t, start in zip(toks, group_flags(toks)):
            if start:
                groups.append([t])
            else:
                groups[-1].append(t)
        return groups

    def pretty(self, text) -> str:
        return " ".join(t for t in self.encode(text))


@dataclass
class _Word:
    space: bool = False
    case: str | None = None
    root: object = None
    stem: list = field(default_factory=list)
    digits: str = ""
    apos: str | None = None
    sfx: list = field(default_factory=list)
    lit: list | None = None
    ascii: bool = False
    bare: bool = False
    edit: list = field(default_factory=list)

    def has_stem(self):
        return self.root is not None or bool(self.stem) or bool(self.digits)

    def kind(self):
        return "NUM" if self.digits else ("R" if self.root else "L")

    def surface(self) -> str:
        tflags = frozenset()
        if self.digits:
            base, reading, flags = self.digits, number_reading(self.digits), frozenset()
        elif self.bare:
            base, reading, flags = "", "", frozenset()
        elif self.root is not None:
            base, flags = self.root.form, self.root.flags
            reading, tflags = self.root.reading or self.root.form, _tail_flags(self.root)
        else:
            base = _join(self.stem).removeprefix("▁")
            reading, flags = spelled_reading(tr_lower(base)), frozenset()
        if self.apos:
            tail = _join(self.lit) if self.lit is not None else \
                generate(reading, tflags, self.sfx)[len(reading):]
            s = (tr_upper(base) if self.case == "<capsb>" else base) + self.apos + tail
        elif self.digits:
            s = base
        else:
            s = generate(base, flags, self.sfx, self.root.inner if self.root is not None else "")
            for op in self.edit:
                s = apply_edit(s, op)
            if self.ascii:
                s = asciify(s)
        if self.case == "<cap>":
            s = tr_cap(s)
        elif self.case == "<caps>":
            s = tr_upper(s)
        return s


def _ops(ops) -> list[str]:
    return [t for op in ops for t in op_tokens(op)]


def _fuse(left: str, right: str) -> str:
    """ne + olur -> nolur, ne + yapıyon -> napıyon, ne + haber -> naber."""
    if left and left[-1].lower() in "aeıioöuü":
        left = left[:-1]
    if len(right) > 1 and right[0].lower() in "yh" and right[1].lower() in "aeıioöuü":
        right = right[1:]
    return left + right


def group_flags(tokens) -> list[int]:
    """1 where a token opens a new word group, 0 where it continues the current one."""
    flags, prefix_open, prev = [], False, ""
    for t in tokens:
        cont = (t.startswith(("S:", "B:", "E:", MERGE_PREFIX)) or (t.startswith("L:") and not t.startswith("L:▁"))
                or t in ("<apos>", "<apos2>", "<lit>") or (t.startswith("D:") and prev.startswith("D:")))
        flags.append(0 if (flags and (cont or prefix_open)) else 1)
        prefix_open = t in WORD_PREFIX
        prev = t
    return flags


def _tail_flags(root):
    # after an apostrophe the base is written as is: only harmony (Kemal'e) and the
    # pronominal n of a compound name (Beyoğlu'nda) carry over
    return root.flags & {"front", "cp3"}


def _join(parts):
    out, buf = [], bytearray()
    for p in parts:
        if isinstance(p, bytes):
            buf += p
        else:
            if buf:
                out.append(buf.decode("utf-8", "replace"))
                buf = bytearray()
            out.append(p)
    if buf:
        out.append(buf.decode("utf-8", "replace"))
    return "".join(out)


def _bytes(ch):
    return [f"B:{b:02x}" for b in ch.encode("utf-8")]


class Vocab:
    """Contiguous id ranges per namespace -> a free token-type embedding."""

    def __init__(self, tk: TurkMorphTokenizer):
        toks = list(SPECIALS)
        self.ranges = {}

        def add(name, items):
            s = len(toks)
            toks.extend(items)
            self.ranges[name] = (s, len(toks))
        add("BYTE", [f"B:{i:02x}" for i in range(256)])
        add("SFX", [f"S:{s.id}" for s in SUFFIX_LIST])
        add("SFX_MARKED", [f"S:{m}" for m in MARKED_IDS])
        add("ROOT", [f"R:{r.key}" for r in tk.lex.roots])
        add("LEX", sorted(f"L:{p}" for p in tk.bpe.vocab))
        add("EDIT", list(EDIT_TOKENS))
        add("DIGIT", [f"D:{d}" for d in "0123456789"])
        add("PUNCT", sorted(f"P:{c}" for c in PUNCT))
        if tk.unions:
            add("SFX_UNION", [f"S:{u}" for u in tk.unions])
        if tk.merge is not None:
            add("MERGE", list(tk.merge.vocab))
        self.toks = toks
        self.index = {t: i for i, t in enumerate(toks)}

    def id(self, t): return self.index[t]
    def tok(self, i):
        if not 0 <= i < len(self.toks):
            raise ValueError(f"token id out of range: {i}")
        return self.toks[i]
    def __len__(self): return len(self.toks)

    def namespace(self, i):
        for k, (a, b) in self.ranges.items():
            if a <= i < b:
                return k
        return "SPECIAL"
