"""Convert Zemberek lexicons (Apache 2.0, github.com/ahmetaa/zemberek-nlp) into Roots.

Zemberek line format:  word [P:Noun, Time; A:Voicing, InverseHarmony; Index:1; Pr:abede]

Attribute mapping (Zemberek -> turkmorph flag):
  Voicing / default*  -> soften      LastVowelDrop -> drop     Doubling -> geminate
  InverseHarmony      -> front       Aorist_I      -> aor_hr   Aorist_A -> aor_ar
  CompoundP3sg        -> cp3         P:...,Time    -> temporal
  * Zemberek's default: a polysyllabic noun/adjective ending in p/ç/t/k softens
    unless marked NoVoicing; monosyllables and verbs soften only if marked Voicing.

Irregular items that need more than a flag (o/bu/şu pronominal n, ben->bana, de->diyor)
stay in the hand lexicon, which takes precedence over Zemberek for the same form+POS.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .morphology import HAND_LEXICON, Root, generate
from .phonology import n_vowels

# causative allomorph classes Zemberek does not encode (-DHr / -t are rule-based)
CAUS_HR = {"bat", "bit", "doğ", "düş", "geç", "iç", "kaç", "piş", "şiş", "taş", "uç", "art",
           "yat", "göç", "şaş", "yit", "kok", "doy", "duy", "sin", "er", "ürk"}
CAUS_AR = {"çık", "kop", "çök", "kalk"}
# take -ki directly like Zemberek's Time nouns (yılki, haftaki, geceki)
TEMPORAL_EXTRA = {"yıl", "ay", "hafta", "gece", "sene", "dakika", "saniye", "asır", "yüzyıl",
                  "mevsim", "bugün", "öğle", "sonbahar", "ilkbahar", "yaz", "kış"}

_LINE = re.compile(r"^(\S+)\s*(?:\[(.*)\])?\s*$")
_WORD = re.compile(r"^[a-zçğıöşüâîû]+$")


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def parse_line(line: str):
    """-> (word, {key: [values]}) or None for comments / blank / multiword lines."""
    line = unicodedata.normalize("NFC", line.strip())
    if not line or line.startswith("#"):
        return None
    m = _LINE.match(line)
    if not m:
        return None
    word, attrs = m.group(1), {}
    for part in (m.group(2) or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            attrs[k.strip()] = [x.strip() for x in v.split(",") if x.strip()]
    return word, attrs


def _flags(form: str, pos: str, P: list, A: set) -> set:
    f = set()
    if "Voicing" in A:
        f.add("soften")
    elif (pos == "N" and "NoVoicing" not in A and form[-1] in "pçtk" and n_vowels(form) > 1):
        f.add("soften")
    if "LastVowelDrop" in A:                         # verbs: only before the passive (çevril-, not *çevren)
        f.add("drop_pass" if pos == "V" else "drop")
    if "Doubling" in A:
        f.add("geminate")
    if "InverseHarmony" in A:
        f.add("front")
    if "CompoundP3sg" in A:
        f.add("cp3")
    if "Time" in P or (pos == "N" and form in TEMPORAL_EXTRA):
        f.add("temporal")
    if P[:1] == ["Num"]:
        f.add("num")                                 # takes -(H)ncH, -(ş)Ar: ikinci, ikişer
    if pos == "V":
        if "Aorist_I" in A:
            f.add("aor_hr")
        if "Aorist_A" in A and n_vowels(form) > 1:
            f.add("aor_ar")
        if form in CAUS_HR:
            f.add("caus_Hr")
        if form in CAUS_AR:
            f.add("caus_Ar")
    return f


# entries the hand lexicon models differently (copula stem i- with idi, ise, iken, imiş)
SKIP = {"imek", "iken", "idi", "ise", "imiş"}


def convert(word: str, attrs: dict, proper: bool = False) -> Root | None:
    P = attrs.get("P", [])
    A = set(attrs.get("A", []))
    head = P[0] if P else ""
    if head in ("Punc", "Abbrv") or "Prop" in P:
        proper = True
    if proper and head == "Abbrv" and not attrs.get("Pr"):
        return None                                  # only abbreviations with a reading
    form = tr_lower(word)
    if form in SKIP:
        return None
    reading = tr_lower(attrs["Pr"][0]) if attrs.get("Pr") else ""
    if head == "Punc" or not _WORD.match(form):
        return None
    if proper:
        pos = "PROPER"
    elif head in ("", "Noun", "Adj", "Num", "Adv", "Det"):
        if not P and re.search(r"m[ae]k$", form) and len(form) > 3:
            pos, form = "V", form[:-3]               # gelmek -> gel
        else:
            pos = "N"
    elif head == "Verb":
        if re.search(r"m[ae]k$", form):
            pos, form = "V", form[:-3]
        else:
            pos = "Q"                                # değil: copular, takes person/tense
    elif head == "Pron":
        pos = "PRON"
    elif head == "Postp":
        pos = "Q"                                    # gibi-ydi, için-dir: copula, no case
    elif head == "Ques":
        pos = "Q"
    else:                                            # Conj, Interj, Dup, ...
        pos = "X"
    if "NoSuffix" in A:
        pos = "X"
    if not form:
        return None
    r = Root(form, pos, frozenset(_flags(form, pos, P, A)), 1, "", reading)
    comps = tuple(tr_lower(c) for c in attrs.get("Roots", [""])[0].split("-") if c)
    if "cp3" in r.flags and len(comps) >= 2:
        COMPOUNDS[form] = COMPOUNDS[form.translate(_PLAIN)] = comps   # zeytinyağı -> (zeytin, yağ)
    return (r, head in ("Adv", "Postp"))


COMPOUNDS: dict[str, tuple] = {}


def compound_inner(r: Root, lex) -> str:
    """'stem|flags' of a CompoundP3sg root without its P3SG, checked by regenerating:
    zeytinyağı (zeytin-yağ) -> 'zeytinyağ|', âdemoğlu (âdem-oğul) -> 'âdemoğul|drop'."""
    comps = COMPOUNDS.get(r.form)
    if not comps:
        return _strip_p3(r.form)
    prefix, last = "".join(comps[:-1]), comps[-1]
    if not r.form.startswith(prefix):                # ademoğlu from âdem-oğul
        prefix, last = prefix.translate(_PLAIN), last.translate(_PLAIN)
    if not r.form.startswith(prefix):                # Roots holds lemmas (bacak-baş): peel instead
        return _strip_p3(r.form)
    # last component may have been pruned as derivable (işbirliği: birlik = bir+LHK)
    fallbacks = [Root(last, "N"), Root(last, "N", frozenset({"soften"})),
                 Root(last, "N", frozenset({"drop"}))]
    for cand in lex.by_form.get(last, []) + fallbacks:
        if cand.pos != "N":
            continue
        fl = cand.flags & {"soften", "drop", "geminate", "front", "y_poss"}
        if generate(prefix + last, fl, ["P3SG"]) == r.form:
            return f"{prefix + last}|{','.join(sorted(fl))}"
    return _strip_p3(r.form)


_UNSOFTEN = {"ğ": "k", "b": "p", "c": "ç", "d": "t"}


def _strip_p3(form: str) -> str:
    """Peel the P3SG surface off a compound and verify by regenerating:
    madensuyu -> madensu (y_poss), düşkırıklığı -> düşkırıklık (soften)."""
    cands = [(form[:-2], ()), (form[:-2], ("y_poss",))]
    stem = form[:-1]
    if stem and stem[-1] in _UNSOFTEN:                # before the plain stem: kırıklık, not kırıklığ
        hard = "g" if stem[-1] == "ğ" and stem[-2:-1] == "n" else _UNSOFTEN[stem[-1]]
        cands.append((stem[:-1] + hard, ("soften",)))
    cands.append((stem, ()))
    for st, fl in cands:
        if len(st) >= 2 and generate(st, frozenset(fl), ["P3SG"]) == form:
            return f"{st}|{','.join(fl)}"
    return ""


def read_dict(path, proper: bool = False):
    """-> (Root, frozen?) ; frozen = Adv/Postp entry that may be an inflected form."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parsed = parse_line(line)
        if parsed:
            r = convert(*parsed, proper=proper)
            if r:
                yield r


# Derivations productive enough to be suffix ids: an entry that one of them
# generates from another root is dropped (yaptır = yap+CAUS, gözlük = göz+LHK).
# -lA/-lAn stay out: they would also split frozen verbs (anla = an+lA, başla = baş+lA);
# so does the passive/reflexive: dokun is not 'doku'+n, bayıl is not 'bay'+ıl.
_DERIV = {"V": ["CAUS"], "N": ["LH", "LHK", "CH", "SHZ", "SAL", "AGT", "VN_HS", "ORD", "DIST"]}
_DERIV_FROM = {"CAUS": "V", "PASS": "V", "LH": "N", "LHK": "N", "CH": "N", "SHZ": "N",
               "SAL": "N", "AGT": "V", "VN_HS": "V", "ORD": "N", "DIST": "N"}
_DERIV_NEEDS = {"ORD": "num", "DIST": "num"}          # only numerals: ikinci, birer
_DERIV_DEFAULT_FLAGS = {"LHK": {"soften"}}


def derivable(roots: list[Root]) -> set[str]:
    """Keys of roots that equal another root + one productive derivational suffix."""
    surface: dict[tuple, Root] = {}
    for r in roots:
        surface.setdefault((r.form, r.pos), r)
    drop = set()
    for r in roots:
        if r.pos not in _DERIV:
            continue
        for sid in _DERIV[r.pos]:
            src = _DERIV_FROM[sid]
            for base in roots_by_prefix.get(r.form[:2], ()):
                if base.pos != src or base.key == r.key or len(base.form) >= len(r.form):
                    continue
                if sid in _DERIV_NEEDS and _DERIV_NEEDS[sid] not in base.flags:
                    continue
                if generate(base.form, base.flags, [sid]) == r.form and \
                        r.flags <= _DERIV_DEFAULT_FLAGS.get(sid, set()) | {"aor_hr", "aor_ar", "num"}:
                    drop.add(r.key)
    return drop


roots_by_prefix: dict[str, list[Root]] = {}


_PLAIN = str.maketrans("âîû", "aiu")
_CASE_IDS = {"DAT", "LOC", "ABL", "INS", "EQU"}            # adverb-forming cases only
_NOT_FROZEN_IDS = {"P2SG", "P2PL", "Z1SG", "Z2SG", "Z1PL", "Z2PL", "K1SG", "K2SG", "K1PL", "K2PL"}


def _is_case_inflection(a, key) -> bool:
    """hakkında = hak+P3SG+LOC, birlikte = bir+LH+LOC: a frozen form worth decomposing.
    Not için = iç+P2SG or göre = gör+OPT (accidental homographs)."""
    return (a.root.key != key and a.root.pos in ("N", "PRON") and bool(_CASE_IDS & set(a.sfx))
            and not _NOT_FROZEN_IDS & set(a.sfx))


def mine_front_harmony(apos_chunks) -> set[str]:
    """Names whose suffixes take front vowels against their last vowel, mined from
    the corpus: Kemal'e, Kemal'in (not *Kemal'a) -> 'kemal' gets the front flag."""
    from .phonology import last_vowel
    votes: dict[str, list] = {}
    for chunk, c in apos_chunks.items():
        base, _, tail = chunk.replace("’", "'").partition("'")
        base, tail = tr_lower(base), tr_lower(tail)
        v_base, v_tail = last_vowel(base), next((ch for ch in tail if ch in "aeıioöuü"), None)
        if v_base is None or v_tail is None or v_base not in "aıou":
            continue
        votes.setdefault(base, [0, 0])[v_tail in "eiöü"] += c
    return {b for b, (back, front) in votes.items() if front >= 3 and front > 2 * back}


def mine_softening(roots, word_counts, min_count=5, ratio=3.0):
    """Corpus decides final-consonant softening for nouns Zemberek gets wrong:
    diyaloğu >> diyalogu -> add 'soften'; a 'soften' noun seen mostly hard -> drop it.
    Compares exact inflected forms (the root + a vowel-initial suffix, generated both
    ways), so unrelated words sharing a prefix (ayrıdır for ayrıt) do not count."""
    chains = [["ACC"], ["DAT"], ["P1SG"], ["P1PL"], ["GEN"], ["P3SG", "LOC"], ["P3SG", "ABL"]]
    add, remove = set(), set()
    for r in roots:
        if r.pos != "N" or len(r.form) < 4 or r.form[-1] not in "pçtkg" or "drop" in r.flags:
            continue
        base = r.flags - {"soften"}
        soft_forms = {generate(r.form, base | {"soften"}, c) for c in chains}
        hard_forms = {generate(r.form, base, c) for c in chains} - soft_forms
        n_soft = sum(word_counts.get(f, 0) for f in soft_forms - hard_forms)
        n_hard = sum(word_counts.get(f, 0) for f in hard_forms)
        if "soften" not in r.flags and n_soft >= min_count and n_soft >= ratio * n_hard:
            add.add(r.key)
        elif "soften" in r.flags and n_hard >= min_count and n_hard >= ratio * n_soft:
            remove.add(r.key)
    return add, remove


def build_lexicon(master, extra=(), proper=(), keep_proper=None, log=print,
                  front_names=frozenset(), corpus_counts=None) -> list[Root]:
    """Hand lexicon first (authoritative), then Zemberek entries.

    master/extra: common-word dictionaries; proper: proper-noun dictionaries, kept
    only if keep_proper(form) is true (e.g. attested capitalized in a corpus) and
    the form is not just an inflected common word (Evler = ev+PL).
    """
    from .morphology import Lexicon, analyze_known

    out: list[Root] = list(HAND_LEXICON)
    hand = {(r.form, r.pos) for r in HAND_LEXICON}
    hand_keys = {r.key for r in HAND_LEXICON}
    seen: dict[tuple, list[Root]] = {}
    frozen: set[str] = set()
    not_frozen: set[str] = set()                     # yeni [Adj] + yeni [Adv] -> not frozen

    def add(r: Root) -> Root | None:
        if (r.form, r.pos) in hand:
            return None
        same = seen.setdefault((r.form, r.pos), [])
        for o in same:
            if o.flags == r.flags:
                return o                             # identical behaviour: one id
        tag = f".{len(same)}" if same else ""
        r = Root(r.form, r.pos, r.flags, r.freq, tag, r.reading, r.inner)
        same.append(r)
        out.append(r)
        return r

    def variants(r: Root):
        yield r
        plain = r.form.translate(_PLAIN)              # ilân -> ilan, siyasî -> siyasi
        if plain != r.form:
            yield Root(plain, r.pos, r.flags, r.freq, "", r.reading)

    for path in [master, *extra]:
        for r0, is_frozen in read_dict(path):
            for r in variants(r0):
                added = add(r)
                if added is not None:
                    (frozen if is_frozen else not_frozen).add(added.key)
    frozen -= not_frozen

    roots_by_prefix.clear()
    for r in out:
        roots_by_prefix.setdefault(r.form[:2], []).append(r)
    if corpus_counts:
        soft_add, soft_rm = mine_softening([r for r in out if r.key not in hand_keys], corpus_counts)
        out = [Root(r.form, r.pos, (r.flags | {"soften"}) if r.key in soft_add else
                    (r.flags - {"soften"}) if r.key in soft_rm else r.flags,
                    r.freq, r.tag, r.reading, r.inner) for r in out]
        log(f"  softening from corpus: +{len(soft_add)} (diyalog -> diyaloğu), -{len(soft_rm)}")
    drop = derivable(out)
    out = [r for r in out if r.key not in drop or r.key in hand_keys]
    log(f"  derivable entries dropped: {len(drop)}")

    # frozen inflected forms (hakkında = hak+P3SG+LOC, birlikte) -> analyse instead
    common = Lexicon(out)
    inflected = set()
    for key in frozen - hand_keys:
        r = common.by_key.get(key)
        if r is None:                                # already dropped as derivable
            continue
        if any(_is_case_inflection(a, key) for a in analyze_known(r.form, common)):
            inflected.add(key)
    out = [r for r in out if r.key not in inflected]
    log(f"  frozen inflected Adv/Postp dropped: {len(inflected)}")

    common = Lexicon(out)
    n_inner = 0
    for i, r in enumerate(out):
        if r.pos == "PRON" and "cp3" not in r.flags and r.key not in hand_keys:
            inner = _strip_p3(r.form)                # biri, çoğu, diğeri, başkası: bir-, çok-, ...
            if inner and inner.split("|")[0] in common.by_form:
                out[i] = Root(r.form, r.pos, r.flags | {"cp3"}, r.freq, r.tag, r.reading, inner)
                n_inner += 1
                continue
        if "cp3" in r.flags and not r.inner:
            inner = compound_inner(r, common)
            if inner:
                out[i] = Root(r.form, r.pos, r.flags, r.freq, r.tag, r.reading, inner)
                n_inner += 1
    log(f"  compound P3SG roots with an inner stem: {n_inner}/{sum('cp3' in r.flags for r in out)}")
    common = Lexicon(out)
    n_prop = n_infl = 0

    def compound_name(r: Root) -> Root:
        if r.form in front_names:
            r = Root(r.form, r.pos, r.flags | {"front"}, r.freq, r.tag, r.reading, r.inner)
        # Beyoğlu, Karaoğlu, -oğlu surnames: compound P3SG of oğul (Beyoğlu'nda, Beyoğulları)
        if r.form.endswith("oğlu") and len(r.form) > 4:
            return Root(r.form, r.pos, r.flags | {"cp3"}, r.freq, r.tag, r.reading,
                        r.form[:-4] + "oğul|drop")
        return r
    for path in proper:
        for r, _ in read_dict(path, proper=True):
            if keep_proper is not None and not keep_proper(r.form):
                continue
            if (r.form, r.pos) in seen or (r.form, r.pos) in hand:
                continue
            if any(a.sfx for a in analyze_known(r.form, common)):
                n_infl += 1
                continue
            add(compound_name(r))
            n_prop += 1
    log(f"  proper nouns kept: {n_prop}, dropped as inflected common words: {n_infl}")
    return out
