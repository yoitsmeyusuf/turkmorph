"""Lexicon, suffix inventory, morphotactics, generator and analysis-by-synthesis."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from .phonology import (VOWELS, drop_last_vowel, n_vowels, realize, soften)


# ---------------------------------------------------------------- lexicon --
@dataclass(frozen=True)
class Root:
    form: str
    pos: str                     # N, V, PRON, PROPER, Q (question clitic), X (uninflected)
    flags: frozenset = frozenset()
    freq: int = 1
    tag: str = ""                # homograph index: same form+pos, different phonology
    reading: str = ""            # pronunciation when spelling misleads (abd -> abede)
    inner: str = ""              # compound P3SG: "stem|flags" without it (zeytinyağı -> zeytinyağ|)

    @property
    def key(self) -> str:
        return f"{self.form}#{self.pos}{self.tag}"


def _r(form, pos, *flags, freq=1):
    return Root(form, pos, frozenset(flags), freq)


# Demo lexicon. In the real system: ~30K roots, flags mined from Zemberek /
# TRmorph lexicons + corpus (e.g. is 'kitab' attested before vowels?).
HAND_LEXICON = [
    # nouns / adjectives
    _r("ev", "N", freq=50), _r("kitap", "N", "soften", freq=30),
    _r("ağız", "N", "drop"), _r("burun", "N", "drop"), _r("oğul", "N", "drop"),
    _r("şehir", "N", "drop", freq=10), _r("resim", "N", "drop"),
    _r("bağır", "N", "drop"),                      # organ; homograph of verb
    _r("hak", "N", "geminate", freq=10), _r("his", "N", "geminate"),
    _r("ret", "N", "soften", "geminate"),
    _r("saat", "N", "front", freq=10), _r("rol", "N", "front"),
    _r("alkol", "N", "front"), _r("kalp", "N", "soften", "front"),
    _r("renk", "N", "soften"), _r("ağaç", "N", "soften"),
    _r("köpek", "N", "soften"), _r("çocuk", "N", "soften", freq=20),
    _r("dolap", "N", "soften"), _r("kanat", "N", "soften"), _r("sokak", "N", "soften"),
    _r("ekmek", "N", "soften"), _r("çiçek", "N", "soften"), _r("kulak", "N", "soften"),
    _r("bıçak", "N", "soften"), _r("süt", "N"), _r("sepet", "N"),
    _r("boyun", "N", "drop"), _r("karın", "N", "drop"), _r("isim", "N", "drop"),
    _r("fikir", "N", "drop"), _r("akıl", "N", "drop"),
    _r("sır", "N", "geminate"), _r("zan", "N", "geminate"),
    _r("tıp", "N", "soften", "geminate"),                  # tıbbı
    _r("dikkat", "N", "front"), _r("kabul", "N", "front"), _r("hayal", "N", "front"),
    _r("gol", "N", "front"), _r("hal", "N", "front"),
    # compounds ending in P3SG: inner stem for plural / possessive (zeytinyağ-lar-ı)
    Root("zeytinyağı", "N", frozenset({"cp3"}), inner="zeytinyağ|"),
    Root("işbirliği", "N", frozenset({"cp3"}), inner="işbirlik|soften"),
    _r("top", "N"), _r("hukuk", "N"), _r("göz", "N"), _r("söz", "N"),
    _r("gün", "N", "temporal", "ki_harm", freq=20), _r("araba", "N"), _r("su", "N", "y_poss"),  # günkü
    _r("kale", "N"), _r("kalem", "N", freq=5), _r("okul", "N"),
    _r("masa", "N"), _r("kedi", "N"), _r("ülke", "N"), _r("dil", "N"),
    _r("yer", "N"), _r("insan", "N", freq=20), _r("el", "N"),
    _r("bilgisayar", "N"), _r("ayakkabı", "N"), _r("kahvaltı", "N"),
    _r("geçmiş", "N"), _r("anlam", "N"), _r("bakım", "N"), _r("mecaz", "N"),
    _r("mana", "N"), _r("kalıp", "N", "soften"), _r("güzel", "N"),
    _r("hasta", "N"), _r("kadar", "N"), _r("bir", "N", "num", freq=50),
    _r("hep", "N"), _r("birlikte", "N"), _r("tarih", "N"), _r("yıl", "N", "temporal"),
    _r("iki", "N", "num"), _r("üç", "N", "num"), _r("dört", "N", "num", "soften"),
    # temporal nouns: take -ki directly (dünkü, yarınki); some round it (-kü)
    _r("dün", "N", "temporal", "ki_harm"), _r("bugün", "N", "temporal", "ki_harm"),
    _r("öbür", "N", "temporal", "ki_harm"), _r("yarın", "N", "temporal"),
    _r("akşam", "N", "temporal"), _r("sabah", "N", "temporal"),
    # pronouns
    _r("ben", "PRON"), _r("sen", "PRON"), _r("o", "PRON", "pron_n"),
    _r("bu", "PRON", "pron_n", freq=30), _r("şu", "PRON", "pron_n"),
    _r("biz", "PRON"), _r("siz", "PRON"), _r("ne", "PRON", "y_poss"),
    _r("kendi", "PRON", "case_n"),                          # kendi-n-i, but kendi-leri
    # pronouns that end in P3SG: inner stem like compounds (birbir-ler-i-n-e, hep-imiz)
    Root("birbiri", "PRON", frozenset({"cp3"}), inner="birbir|"),
    Root("hepsi", "PRON", frozenset({"cp3"}), inner="hep|"),
    # copula stem i-: idi, imiş, ise, iken (no buffer y after it)
    _r("i", "Q", "no_buffer"), _r("gibi", "Q"),
    # place words with spoken locative/ablative: nerde, burdan, orda, şurdan
    _r("nere", "N", "coll"), _r("bura", "N", "coll"), _r("ora", "N", "coll"), _r("şura", "N", "coll"),
    # proper nouns
    _r("danimarka", "PROPER"), _r("türkiye", "PROPER"),
    _r("istanbul", "PROPER"), _r("ankara", "PROPER"),
    # verbs
    _r("gel", "V", "aor_hr", freq=20), _r("git", "V", "soften", freq=20),
    _r("et", "V", "soften"), _r("al", "V", "aor_hr"), _r("yap", "V", freq=20),
    _r("oku", "V"), _r("ara", "V"), _r("de", "V", "narrow_y"),
    _r("ye", "V", "narrow_y"), _r("kalk", "V"), _r("yürü", "V"),
    _r("anla", "V"), _r("başla", "V"), _r("söyle", "V"), _r("sev", "V"),
    _r("bil", "V", "aor_hr"), _r("ol", "V", "aor_hr"), _r("bul", "V", "aor_hr"),
    _r("kazan", "V"), _r("ulaş", "V"), _r("oyna", "V"), _r("tat", "V", "soften"),
    _r("bit", "V", "caus_Hr"), _r("at", "V"), _r("bağır", "V"),
    _r("bat", "V", "caus_Hr"), _r("çık", "V", "caus_Ar"), _r("piş", "V", "caus_Hr"),
    _r("yaz", "V"), _r("gör", "V", "aor_hr"), _r("bak", "V"), _r("öl", "V", "aor_hr"),
    _r("otur", "V"),
]

SUPPLETION = {
    ("ben", "DAT"): "bana", ("sen", "DAT"): "sana",
    ("ben", "GEN"): "benim", ("biz", "GEN"): "bizim",          # not *benin
    ("ben", "INS"): "benimle", ("sen", "INS"): "seninle",
    ("biz", "INS"): "bizimle", ("siz", "INS"): "sizinle",
    ("o", "INS"): "onunla", ("bu", "INS"): "bununla", ("şu", "INS"): "şununla",
    ("gel", "CAUS"): "getir", ("kalk", "CAUS"): "kaldır",
}

# (state, suffix) -> root flag required when the suffix attaches directly to
# the root: -ki straight after a noun only for temporals (dünkü, *evki).
ROOT_GATED = {("N", "REL"): "temporal", ("N", "ORD"): "num", ("N", "DIST"): "num",
              ("N", "LOC_C"): "coll", ("N", "ABL_C"): "coll"}


# --------------------------------------------------------------- suffixes --
@dataclass(frozen=True)
class Suffix:
    id: str
    tmpl: str
    nxt: str
    n_tmpl: str | None = None     # variant after pronominal n (evi-n-de)
    soften: bool = False          # final k softens before vowel (-DHk, -AcAk)
    cost: float = 1.0


_S = Suffix
SUFFIX_LIST = [
    # nominal inflection
    _S("PL", "lAr", "N_PL", n_tmpl="nlAr"),                 # o -> onlar
    _S("P1SG", "(H)m", "N_POSS"), _S("P2SG", "(H)n", "N_POSS", cost=1.2),
    _S("P3SG", "(s)H", "N_POSS3"), _S("P1PL", "(H)mHz", "N_POSS"),
    _S("P2PL", "(H)nHz", "N_POSS"), _S("P3PL", "lArH", "N_POSS3", cost=2.5),
    _S("ACC", "(y)H", "N_CASE", n_tmpl="nH"),
    _S("DAT", "(y)A", "N_CASE", n_tmpl="nA"),
    _S("LOC", "DA", "N_LOCGEN", n_tmpl="nDA"),
    _S("ABL", "DAn", "N_CASE", n_tmpl="nDAn"),
    _S("GEN", "(n)Hn", "N_LOCGEN"),
    _S("INS", "(y)lA", "N_CASE"),
    _S("EQU", "CA", "N_CASE", n_tmpl="ncA"),
    _S("REL", "ki", "N_REL"),
    # copula / predicative
    _S("COP3", "DHr", "N_COP"), _S("PCOP", "(y)DH", "V_K"),
    _S("KEN", "(y)ken", "END"),                             # no harmony: evdeyken
    _S("ECOP", "(y)mHş", "V_Z"), _S("CCOP", "(y)sA", "V_K"),
    # derivation
    _S("LH", "lH", "N"), _S("LHK", "lHk", "N", soften=True),
    _S("CH", "CH", "N"), _S("SHZ", "sHz", "N"), _S("LAS", "lAş", "V"),
    _S("SAL", "sAl", "N", cost=1.5), _S("DAS", "DAş", "N", cost=1.5),      # kurumsal, meslektaş
    _S("LA", "lA", "V", cost=1.5), _S("LAN", "lAn", "V", cost=1.5),         # temizle, değerlen
    _S("AGT", "(y)HcH", "N", cost=1.3),                                     # yönetici, okuyucu
    _S("VN_HS", "(y)Hş", "N", cost=1.8),                                    # oluş, bakış, anlayış
    _S("ORD", "(H)ncH", "N"), _S("DIST", "(ş)Ar", "N"),                     # ikinci, ikişer (numerals)
    _S("DIM", "CHk", "N", soften=True, cost=1.5),                           # ayıcık, kedicik
    _S("CASINA", "CAsHnA", "END", cost=2.0),                                # yanmışçasına, delicesine
    _S("VN_MACA", "mAcA", "N", cost=2.0),                                   # kandırmaca, saklambaç
    # voice + compound verbs
    _S("RECP", "(H)ş", "V_RECP", cost=1.5), _S("CAUS", "<CAUS>", "V_CAUS", cost=0.9),
    _S("PASS", "<PASS>", "V_PASS"), _S("HVER", "(y)Hver", "V_HVER"),
    # verbal
    _S("NEG", "mA", "V_NEG"), _S("ABIL", "(y)Abil", "V_NEG"),
    _S("IMPOT", "(y)AmA", "V_NEG"),
    _S("PROG", "(H)yor", "V_Z"), _S("PAST", "DH", "V_PAST"),
    # colloquial (spoken/informal) variants: own ids, costlier than the standard ones
    _S("PROG_C", "(H)yo", "V_ZC", cost=1.5),                               # geliyo(m), arıyo
    _S("FUT_C", "(y)HcA", "V_FUTC", cost=1.5), _S("FUT_C2", "cA", "V_FUTC", cost=1.8),  # yapıcam, yapcam
    _S("Z1SG_C", "m", "END", cost=1.2), _S("Z2SG_C", "n", "END", cost=1.2),
    _S("Z1PL_C", "z", "END", cost=1.2), _S("FUT_C3", "k", "V_Z", cost=1.2),            # yapıcak
    _S("LOC_C", "DA", "N_LOCGEN", cost=1.2), _S("ABL_C", "DAn", "N_CASE", cost=1.2),  # ner-de, bur-dan
    _S("FUT", "(y)AcAk", "V_FUT", soften=True), _S("EVID", "mHş", "V_Z"),
    _S("AOR", "<AOR>", "V_Z"), _S("COND", "sA", "V_COND"),
    _S("NECES", "mAlH", "V_Z"),
    _S("OPT", "(y)A", "V_OPT", cost=1.3), _S("OPT1PL", "lHm", "END"),   # geleyim, gelelim
    _S("IMP3SG", "sHn", "V_IMP3", cost=1.2), _S("IMP2PL", "(y)Hn", "END", cost=1.4),
    _S("IMP2PLF", "(y)HnHz", "END", cost=1.4),
    # converbs
    _S("CV_IP", "(y)Hp", "END"), _S("CV_ARAK", "(y)ArAk", "END"),
    _S("CV_INCA", "(y)HncA", "V_INCA"), _S("CV_DIKCE", "DHkçA", "END"),    # gelince(-ye)
    _S("CV_MADAN", "mAdAn", "END"), _S("CV_MAKSIZIN", "mAksHzHn", "END"),   # olmadan: negative
    _S("PROG2", "mAktA", "V_Z"),                                            # içermektedir
    _S("PART_DHK", "DHk", "N_PART", soften=True),
    _S("PART_AN", "(y)An", "N"), _S("INF", "mAk", "N"),
    _S("VN", "mA", "N", cost=2.2),
    # person (z-paradigm, k-paradigm)
    # Z1SG after negative aorist is -m and the aorist is zero: gelmem (not *gelmezim)
    # person ends the word except for -CAsInA "as if": yapmışsınızcasına, gelmişlercesine
    _S("Z1SG", "(y)Hm", "V_PERS", cost=0.9), _S("Z2SG", "sHn", "V_PERS", cost=0.9),
    _S("Z1PL", "(y)Hz", "V_PERS", cost=0.9), _S("Z2PL", "sHnHz", "V_PERS", cost=0.9),
    _S("Z3PL", "lAr", "V_Z3PL"),                            # gelmişlerdi
    _S("K1SG", "m", "END"), _S("K2SG", "n", "END"), _S("K1PL", "k", "END"),
    _S("K2PL", "nHz", "END"), _S("K3PL", "lAr", "END"),
]
SUFFIXES = {s.id: s for s in SUFFIX_LIST}

_CASES = ["ACC", "DAT", "LOC", "ABL", "GEN", "INS", "EQU"]
_POSS = ["P1SG", "P2SG", "P3SG", "P1PL", "P2PL", "P3PL"]
_COP = ["COP3", "PCOP", "ECOP", "CCOP", "KEN"]
_ZP = ["Z1SG", "Z2SG", "Z1PL", "Z2PL"]
_TENSE = ["PROG", "PROG2", "PROG_C", "FUT_C", "FUT_C2", "PAST", "FUT", "EVID", "AOR", "COND", "NECES", "OPT",
          "PART_DHK", "PART_AN", "INF", "VN",
          "IMP3SG", "IMP2PL", "IMP2PLF", "CV_IP", "CV_ARAK", "CV_INCA", "CV_DIKCE"]
_VFIN = ["NEG", "ABIL", "IMPOT", "HVER", "AGT", "VN_HS", "VN_MACA", "CV_MADAN", "CV_MAKSIZIN", *_TENSE]

STATES: dict[str, list[str]] = {
    "N": ["PL", *_POSS, *_CASES, *_COP, "LH", "LHK", "CH", "SHZ", "LAS", *_ZP, "REL",
          "SAL", "DAS", "LA", "LAN", "ORD", "DIST", "DIM", "LOC_C", "ABL_C", "CASINA"],
    "N_PL": [*_POSS[:-1], *_CASES, *_COP],
    "N_POSS": [*_CASES, *_COP, *_ZP],                       # arkadaşınız-ım
    "N_POSS3": [*_CASES, *_COP, *_ZP],                      # tanesi-y-im
    "N_CASE": [*_COP, *_ZP, "LHK"],                         # karşıya-yız, ayrıca-lık
    "N_LOCGEN": ["REL", *_COP, *_ZP, "LHK"],                # günde-lik
    "N_REL": ["PL", "P3SG", *_CASES, "COP3"],
    "N_COP": ["Z3PL"],
    "V_Z3PL": ["PCOP", "ECOP", "CCOP", "COP3", "KEN", "CASINA"],   # otururlar-ken, gelmişler-cesine
    "V_PERS": ["CASINA"],
    "N_PART": [*_POSS, "PL", "ABL"],                       # gittik-ten
    "V": ["RECP", "CAUS", "PASS", *_VFIN],
    "V_RECP": ["CAUS", "PASS", *_VFIN],
    "V_CAUS": ["CAUS", "PASS", *_VFIN],                     # yap-tır-t
    "V_PASS": ["CAUS", "PASS", *_VFIN],                     # giyin-dir, doku-n-ul, de-n-il
    "V_HVER": [s for s in _VFIN if s != "HVER"],
    "V_NEG": _TENSE,
    "V_Z": [*_ZP, "Z3PL", "PCOP", "ECOP", "CCOP", "COP3", "KEN", "LHK", "CASINA"],   # anlaşmaz-lık
    "V_INCA": ["DAT"],                                      # gelince-ye
    "V_ZC": ["Z1SG_C", "Z2SG_C", "Z1PL_C", "Z2PL", "Z3PL", "PCOP", "ECOP", "CCOP", "KEN"],  # geliyom, geliyodu
    "V_FUTC": ["Z1SG_C", "Z2SG_C", "Z1PL_C", "FUT_C3"],    # yapıca-m, yapıca-k
    "V_K": ["K1SG", "K2SG", "K1PL", "K2PL", "K3PL"],
    "V_PAST": ["K1SG", "K2SG", "K1PL", "K2PL", "K3PL", "CCOP"],   # geldiyse
    "V_COND": ["K1SG", "K2SG", "K1PL", "K2PL", "K3PL", "PCOP", "ECOP"],  # gelseydik
    "V_IMP3": ["Z3PL"],                                     # gelsinler
    "V_OPT": ["Z1SG", "Z2SG", "OPT1PL", "Z2PL", "Z3PL", "PCOP"],     # geleyim, geleydi
    "V_FUT": [*_ZP, "Z3PL", "PCOP", "ECOP", "COP3", "KEN", *_POSS, "PL", *_CASES],
    "END": [],
}
FINAL = set(STATES) - {"N_PART"}
POS_START = {"N": "N", "PROPER": "N", "PRON": "N", "V": "V", "Q": "V_Z", "X": "END"}
PRON_N_STATES = {"N_POSS3"}


# ------------------------------------------------------ marked allomorphs --
# S:ABL!dan = "ablative, surface dan". Only used where the generator is wrong
# because the reading is unknown (Bordeaux'ya, iPhone'u): the id still says ABL.
_CONTEXTS = [v + c for v in "aeıioöuü" for c in ("", "t", "d", "n")]
_MARKABLE = ["PL", *_POSS, *_CASES, "COP3", "PCOP", "ECOP"]


def _allomorphs(sf: Suffix) -> list[str]:
    out = set()
    for tmpl in {sf.tmpl, sf.n_tmpl} - {None}:
        out |= {realize(tmpl, ctx, False) for ctx in _CONTEXTS}
    return sorted(out)


MARKED = {sid: _allomorphs(SUFFIXES[sid]) for sid in _MARKABLE}
MARKED_IDS = [f"{sid}!{a}" for sid in _MARKABLE for a in MARKED[sid]]


def suffix_base(sid: str) -> str:
    """ABL!dan -> ABL; ACC|P3SG (underspecified) -> ACC, the member the generator uses."""
    return sid.split("!", 1)[0].split("|", 1)[0]


def is_suffix_id(sid: str) -> bool:
    if "|" in sid:                                       # underspecified: members are plain ids or sequences
        return all(m in SUFFIXES for mem in sid.split("|") for m in mem.split("+"))
    base, _, surf = sid.partition("!")
    return base in SUFFIXES and (not surf or surf in MARKED.get(base, ()))


_SFX_ORDER = {s.id: i for i, s in enumerate(SUFFIX_LIST)}


def _member_key(m: str):
    ids = m.split("+")
    return (len(ids), [_SFX_ORDER[i] for i in ids])


def union_id(members) -> str:
    """{'P3SG', 'ACC'} -> 'P3SG|ACC'; members may be suffix sequences ('PL+ACC').
    Canonical order (fewest morphemes, then inventory order); the first member is
    what the generator realizes."""
    return "|".join(sorted(set(members), key=_member_key))


def expand_unions(sfx_ids):
    """Chain with union ids replaced by their first member's suffixes (for generation)."""
    out = []
    for sid in sfx_ids:
        out.extend(sid.split("|", 1)[0].split("+") if "|" in sid else [sid])
    return out


def boundaries(root: Root, chain) -> tuple:
    """Surface length after the root and after each suffix."""
    return tuple(len(generate(root.form, root.flags, list(chain[:i]), root.inner))
                 for i in range(len(chain) + 1))


UNION_MARGIN = 2.3                                       # nats: within 10x of the best analysis


def underspecify(best, alts, margin: float = UNION_MARGIN):
    """`best`'s chain with the spans that near-equal alternatives (same root, same word,
    cost within `margin`) segment or label differently merged into union ids:
      kitab+ı         ACC vs P3SG              -> P3SG|ACC
      kitap+ları      P3PL vs PL+ACC, PL+P3SG  -> P3PL|PL+ACC|PL+P3SG
    The token claims only what the word itself shows; context decides the rest."""
    near = [a for a in alts if a.root.key == best.root.key and a.sfx != best.sfx
            and a.cost - best.cost <= margin and not any("!" in x for x in a.sfx)]
    if not near or any("!" in x for x in best.sfx):
        return list(best.sfx)
    chains = [list(best.sfx)] + [list(a.sfx) for a in near]
    bnds = [boundaries(best.root, c) for c in chains]
    common = set(bnds[0]).intersection(*map(set, bnds[1:]))
    segs = []                                            # per chain: list of segments (tuples)
    for c, b in zip(chains, bnds):
        cur, out = [], []
        for i, sid in enumerate(c):
            cur.append(sid)
            if b[i + 1] in common:
                out.append(tuple(cur))
                cur = []
        if cur:
            out.append(tuple(cur))
        segs.append(out)
    if len({len(x) for x in segs}) != 1:
        return list(best.sfx)
    out = []
    for k, seg in enumerate(segs[0]):
        members = {"+".join(x[k]) for x in segs}
        out.extend(seg if len(members) == 1 else [union_id(members)])
    return out


# -------------------------------------------------------------- generator --
# A compound noun that ends in P3SG (zeytin+yağ+ı) drops it for a possessive and
# re-adds it after the plural: zeytinyağ-ım, zeytinyağ-lar-ı(-n-a).
CP3_SWAP = {"PL", "P1SG", "P2SG", "P1PL", "P2PL", "P3PL"}


# de- keeps its e before these (TDK: deyip, deyince, deyiş, deyin) but not ye- (yiyip)
_DE_KEEP = {"CV_IP", "CV_INCA", "VN_HS", "IMP2PL", "IMP2PLF"}


class GenState(NamedTuple):
    """Generator state between two suffixes (lets the analyzer extend a chain by one step)."""
    s: str
    flags: frozenset
    prev_root: bool
    prev_soften: bool
    prev_id: str | None
    pron_n: bool
    vowel_from_root: bool
    neg_aor: bool
    cp3_pl: bool
    i: int


def gen_start(stem: str, flags: frozenset, first: str | None, inner: str = "") -> tuple[GenState, bool]:
    """State before the first suffix. The first suffix decides compound swaps and
    suppletion; if suppletion consumes it (ben+DAT -> bana), the state is after it."""
    cp3_pl = False
    if inner and first is not None and suffix_base(first) in CP3_SWAP:
        stem, iflags = inner.split("|")
        flags = frozenset(filter(None, iflags.split(",")))
        cp3_pl = suffix_base(first) == "PL"
    st = GenState(stem, flags, True, False, None,              # zeytinyağı-n-a, kendi-n-e
                  "pron_n" in flags or "cp3" in flags or "case_n" in flags,
                  True, False, cp3_pl, 0)
    if first is not None and (stem, first) in SUPPLETION:
        return st._replace(s=SUPPLETION[(stem, first)], prev_root=False, vowel_from_root=False,
                           prev_id=first, i=1), True
    return st, False


def gen_step(st: GenState, sid: str, nxt: str | None) -> GenState:
    """Append one suffix. nxt (the following suffix) matters only for the aorist."""
    s, flags, prev_root, prev_soften, prev_id, pron_n, vowel_from_root, neg_aor, cp3_pl, i = st
    sid, _, marked = sid.split("|", 1)[0].partition("!")
    sf = SUFFIXES[sid]
    front = vowel_from_root and "front" in flags

    if marked:
        piece = marked
    # vowel narrowing before -(H)yor: ara->arıyor, söyle->söylüyor, de->diyor
    elif sid in ("LOC_C", "ABL_C"):
        s = s[:-1]                                              # nere-de -> ner-de
        piece = realize(sf.tmpl, s, False)
    elif sid in ("PROG", "PROG_C") and s and s[-1] in "ae":
        dropped = s[-1]
        s = s[:-1]
        base = s if any(c in VOWELS for c in s) else dropped
        piece = realize("H", base, False) + ("yor" if sid == "PROG" else "yo")
    else:
        tmpl = sf.n_tmpl if (pron_n and sf.n_tmpl and (sid != "PL" or "pron_n" in flags)) else sf.tmpl
        if (prev_root and "no_buffer" in flags) or prev_id == "PROG_C":
            tmpl = tmpl.replace("(y)", "")                      # i-di, i-se; geliyo-du
        if sid == "AOR":
            tmpl = _aorist(s, prev_root, prev_id, flags)
            if tmpl == "z" and nxt in ("Z1SG", "Z1PL"):
                tmpl, neg_aor = "", True                        # gelmem, gelmeyiz
        elif sid == "Z1SG" and neg_aor:
            tmpl = "m"
        elif sid == "CAUS":
            tmpl = _causative(s, prev_root, flags)
        elif sid == "PASS":
            tmpl = _passive(s)
        elif cp3_pl and i == 0:
            tmpl = "lArH"                                       # zeytinyağ-lar-ı
        elif sid == "REL" and prev_root and "ki_harm" in flags:
            tmpl = "kü"                                         # dünkü, bugünkü
        if sid in ("P3SG", "GEN") and prev_root and "y_poss" in flags:
            tmpl = "(y)H" if sid == "P3SG" else "(y)Hn"         # su -> suyu, suyun
        piece = realize(tmpl, s, front)
        if (prev_root and "narrow_y" in flags and piece[:1] == "y" and s[-1:] == "e"
                and not (s == "de" and sid in _DE_KEEP)):
            s = s[:-1] + "i"                                    # de -> diyecek
        if piece and piece[0] in VOWELS:
            if prev_root:
                if "drop" in flags or ("drop_pass" in flags and sid == "PASS"):
                    s = drop_last_vowel(s)                  # ağız-ı -> ağzı; çevir-il -> çevril
                if "soften" in flags:
                    s = soften(s)
                if "geminate" in flags:
                    s = s + s[-1]
            elif prev_soften:
                s = soften(s)
    s += piece
    if vowel_from_root and any(c in VOWELS for c in piece):
        vowel_from_root = False
    return GenState(s, flags, False, sf.soften, sid, sf.nxt in PRON_N_STATES or (cp3_pl and i == 0),
                    vowel_from_root, neg_aor, cp3_pl, i + 1)


def generate(stem: str, flags: frozenset, sfx_ids: list[str], inner: str = "") -> str:
    """Deterministic surface realization. The single source of truth for decode."""
    if any("|" in x for x in sfx_ids):
        sfx_ids = expand_unions(sfx_ids)
    st, _ = gen_start(stem, flags, sfx_ids[0] if sfx_ids else None, inner)
    for k in range(st.i, len(sfx_ids)):
        st = gen_step(st, sfx_ids[k], sfx_ids[k + 1] if k + 1 < len(sfx_ids) else None)
    return st.s


def _causative(s, prev_root, flags):
    if prev_root and "caus_Hr" in flags:
        return "Hr"                                             # bat -> batır
    if prev_root and "caus_Ar" in flags:
        return "Ar"                                             # çık -> çıkar
    if n_vowels(s) > 1 and (s[-1] in VOWELS or s[-1] in "lr"):
        return "t"                                              # oku -> okut, otur -> oturt
    return "DHr"                                                # yap -> yaptır, de -> dedir


def _passive(s):
    if s and s[-1] in VOWELS:
        return "n"                                              # oku -> okun
    if s.endswith("l"):
        return "Hn"                                             # bil -> bilin
    return "Hl"                                                 # yaz -> yazıl


def _aorist(s, prev_root, prev_id, flags):
    if prev_id in ("NEG", "IMPOT"):
        return "z"                                              # gelmez
    if s and s[-1] in VOWELS:
        return "r"
    if prev_root:
        if "aor_ar" in flags:
            return "Ar"
        return "Hr" if ("aor_hr" in flags or n_vowels(s) > 1) else "Ar"
    return "Hr"


# -------------------------------------------------------- analysis (A-b-S) --
@dataclass(order=True)
class Analysis:
    cost: float
    tiebreak: tuple = field(compare=True)
    root: Root | None = field(compare=False, default=None)
    stem: str = field(compare=False, default="")     # for guessed roots
    sfx: tuple = field(compare=False, default=())


def _search(word, stem, flags, state, sfx, out, max_depth=12, inner=""):
    """All suffix chains from `state` that realize `word` from this stem (DFS).

    Incremental: each node keeps the generator state before and after its last
    suffix, so a child costs one gen_step (two after an aorist, whose surface
    depends on what follows: gelmez / gelmem)."""
    pre, full = None, None
    if sfx:                                              # analyze_tail starts with a marked suffix
        full, consumed = gen_start(stem, flags, sfx[0], inner)
        for k in range(full.i, len(sfx)):
            pre = full
            full = gen_step(full, sfx[k], None)
        if consumed and len(sfx) == 1:
            pre = None
    _dfs(word, stem, flags, inner, state, list(sfx), pre, full,
         full.s if full else stem, out, max_depth)


def _first_chars() -> dict[str, frozenset]:
    """Every first character a suffix's surface can have, over all contexts. A child
    whose next word character is not among them is never generated."""
    extra = {"AOR": ["z", "r", "Hr", "Ar"], "CAUS": ["Hr", "Ar", "t", "DHr"],
             "PASS": ["n", "Hn", "Hl"], "P3SG": ["(y)H"], "GEN": ["(y)Hn"], "REL": ["kü"],
             "PL": ["lArH"], "Z1SG": ["m"]}
    out = {}
    for sid, sf in SUFFIXES.items():
        tmpls = {sf.tmpl, sf.n_tmpl, *extra.get(sid, [])} - {None, "<AOR>", "<CAUS>", "<PASS>"}
        out[sid] = frozenset(realize(t, ctx, f)[:1] for t in tmpls for ctx in _CONTEXTS
                             for f in (False, True)) - {""}
    return out


FIRST_CHARS = _first_chars()
# suffixes whose step can delete the last char (aorist z -> 0) or rewrite it before
# the piece (-(H)yor narrowing): no first-char filter for them
_NO_FILTER = {"AOR", "PROG", "PROG_C"}


def _dfs(word, stem, flags, inner, state, sfx, pre, full, surf, out, max_depth):
    if surf == word and state in FINAL:
        out.append(tuple(sfx))
    if len(sfx) >= max_depth:
        return
    # what later suffixes may still rewrite: the root's last 2 chars at the first join
    # (ağız -> ağz, zeytinyağı -> zeytinyağ-), afterwards only the last char
    # (prev softening -DHk -> -DHğ, -(H)yor narrowing, aorist z -> 0 in gelmem)
    safe = surf[:max(0, len(surf) - (1 if sfx else 2))]
    if not word.startswith(safe):
        return
    last = sfx[-1] if sfx else None
    for sid in STATES[state]:
        gate = ROOT_GATED.get((state, sid))
        if gate and (sfx or gate not in flags):
            continue
        if last is None:
            st0, consumed = gen_start(stem, flags, sid, inner)
            c_pre, c_full = (None, st0) if consumed else (st0, gen_step(st0, sid, None))
        else:
            p = gen_step(pre, last, sid) if (last == "AOR" and pre is not None) else full
            if sid not in _NO_FILTER:
                k = len(p.s)
                if k >= len(word) or word[k] not in FIRST_CHARS[sid]:
                    continue
            c_pre, c_full = p, gen_step(p, sid, None)
        sfx.append(sid)
        _dfs(word, stem, flags, inner, SUFFIXES[sid].nxt, sfx, c_pre, c_full, c_full.s, out, max_depth)
        sfx.pop()


# z-person right on a noun is a nominal predicate (gözüm = "I am an eye");
# rarer than possession (gözüm = "my eye"), so it pays extra.
_PREDICATIVE_START = {"Z1SG", "Z2SG", "Z1PL", "Z2PL"}
MARKED_COST = 2.0


def suffix_cost(sfx) -> float:
    extra = 0.3 if sfx and sfx[0] in _PREDICATIVE_START else 0.0
    extra += sum(MARKED_COST for s in sfx if "!" in s)
    return extra + sum(SUFFIXES[suffix_base(s)].cost for s in sfx)


# ---------------------------------------------------------------- Lexicon --
def _joined(form: str, flags) -> str:
    """The stem a vowel-initial suffix sees: drop, soften, geminate (as generate applies them)."""
    if "drop" in flags or "drop_pass" in flags:
        form = drop_last_vowel(form)
    if "soften" in flags:
        form = soften(form)
    if "geminate" in flags:
        form = form + form[-1]
    return form


def stem_variants(root: Root) -> set[str]:
    """Every string an analysis of this root can start with (design doc: the root
    trie is expanded with alternate stems: kitab, ağz, hakk, gid, ar(ıyor), di(yor)).
    A word is only tried against roots one of whose variants is its prefix."""
    out = set()
    stems = [(root.form, root.flags)]
    if root.inner:                                       # zeytinyağ(-ları), düşkırıklığ(-ım)
        st, fl = root.inner.split("|")
        stems.append((st, frozenset(filter(None, fl.split(",")))))
    for form, flags in stems:
        out.add(form)
        out.add(_joined(form, flags))
        if root.pos == "V" and form[-1:] in ("a", "e"):
            out.add(form[:-1])                           # ara -> ar-ıyor, de -> d-iyor
        if "narrow_y" in flags:
            out.add(form[:-1] + "i")                     # de -> di-yecek
        if "coll" in flags:
            out.add(form[:-1])                           # nere -> ner-de
    out |= {v for (f, _), v in SUPPLETION.items() if f == root.form}   # bana, getir
    return {v for v in out if v}


class Lexicon:
    """Roots + an index of their stem variants (a flat trie): variant -> roots."""

    def __init__(self, roots):
        self.roots: list[Root] = list(roots)
        self.by_key: dict[str, Root] = {}
        self.by_form: dict[str, list[Root]] = {}
        self.index: dict[str, list[Root]] = {}
        for r in self.roots:
            if r.key in self.by_key:
                raise ValueError(f"duplicate root key {r.key}")
            self.by_key[r.key] = r
            self.by_form.setdefault(r.form, []).append(r)
            for k in stem_variants(r):
                self.index.setdefault(k, []).append(r)

    def __len__(self):
        return len(self.roots)

    def candidates(self, word: str) -> list[Root]:
        seen, out = set(), []
        for i in range(1, len(word) + 1):
            for root in self.index.get(word[:i], ()):
                if root.key not in seen:
                    seen.add(root.key)
                    out.append(root)
        return out

    # TSV: form, pos, tag, flags (comma-separated), freq, reading, inner
    def to_tsv(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# form\tpos\ttag\tflags\tfreq\treading\tinner\n")
            for r in self.roots:
                f.write(f"{r.form}\t{r.pos}\t{r.tag}\t{','.join(sorted(r.flags))}\t{r.freq}"
                        f"\t{r.reading}\t{r.inner}\n")

    @classmethod
    def from_tsv(cls, path) -> "Lexicon":
        roots = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            form, pos, tag, flags, freq, reading, *inner = line.split("\t")
            roots.append(Root(form, pos, frozenset(filter(None, flags.split(","))), int(freq),
                              tag, reading, inner[0] if inner else ""))
        return cls(roots)


DEFAULT_LEXICON = Lexicon(HAND_LEXICON)
# module-level names kept for the hand lexicon (demo preset)
LEXICON = HAND_LEXICON
ROOTS = DEFAULT_LEXICON.by_key
ROOTS_BY_FORM = DEFAULT_LEXICON.by_form


# ------------------------------------------------------------- cost models --
class CostModel:
    """Hand-tuned costs (prototype). Lower = preferred."""

    def analysis_cost(self, root: Root, sfx) -> float:
        return suffix_cost(sfx)


END = "$"


class LearnedCosts(CostModel):
    """-log P(analysis), P = P(root) * prod_i P(s_i | s_{i-1}) * P($ | s_n); s_0 = '^' + POS.

    Counts come from EM over a corpus (see turkmorph.em). Root probs are add-alpha
    smoothed; a suffix bigram backs off to the distribution of suffixes in the FSA
    state the previous suffix leads to, and that to the unigram. So an unseen pair
    like DIM -> ? is scored by what follows nouns (P1SG), not by verb persons (Z1SG):
    anneciğim = anne+DIM+P1SG.
    """

    def __init__(self, root_counts, bigram_counts, unigram_counts, alpha=0.1, k=2.0,
                 n_roots=30_000):
        self.root_counts = dict(root_counts)
        self.bigram_counts = {p: dict(d) for p, d in bigram_counts.items()}
        self.unigram_counts = dict(unigram_counts)
        self.alpha, self.k = alpha, k
        total = sum(self.root_counts.values())
        self._root_den = total + alpha * max(n_roots, len(self.root_counts))
        n_sfx = len(SUFFIXES) + 1
        self._uni_den = sum(self.unigram_counts.values()) + n_sfx
        self._prev_tot = {p: sum(d.values()) for p, d in self.bigram_counts.items()}
        self.state_counts: dict[str, dict[str, float]] = {}
        for prev, d in self.bigram_counts.items():
            st = _state_after(prev)
            sc = self.state_counts.setdefault(st, {})
            for x, c in d.items():
                sc[x] = sc.get(x, 0.0) + c
        self._state_tot = {st: sum(d.values()) for st, d in self.state_counts.items()}
        self._cache: dict[tuple, float] = {}

    def root_cost(self, root: Root) -> float:
        return -math.log((self.root_counts.get(root.key, 0.0) + self.alpha) / self._root_den)

    def _p_uni(self, s):
        return (self.unigram_counts.get(s, 0.0) + 1.0) / self._uni_den

    def sfx_cost(self, prev: str, s: str) -> float:
        key = (prev, s)
        c = self._cache.get(key)
        if c is None:
            st = _state_after(prev)
            sc = self.state_counts.get(st, {})
            p_state = (sc.get(s, 0.0) + self.k * self._p_uni(s)) / (self._state_tot.get(st, 0.0) + self.k)
            big = self.bigram_counts.get(prev, {})
            p = (big.get(s, 0.0) + self.k * p_state) / (self._prev_tot.get(prev, 0.0) + self.k)
            c = self._cache[key] = -math.log(p)
        return c

    # z-person right after a nominal is a predicate (anneciğim = "I am a mommy"), much
    # rarer than possession; without a prior EM drifts to it, because nothing can
    # follow a person suffix and so its end-of-word probability is ~1.
    PREDICATIVE_PENALTY = 1.5

    def analysis_cost(self, root: Root, sfx) -> float:
        c = self.root_cost(root)
        prev = "^" + root.pos
        for s in sfx:
            s = suffix_base(s)
            c += self.sfx_cost(prev, s)
            if s in _PREDICATIVE_START and _state_after(prev).startswith("N"):
                c += self.PREDICATIVE_PENALTY
            prev = s
        return c + self.sfx_cost(prev, END)

    def to_json(self, path):
        Path(path).write_text(json.dumps({
            "alpha": self.alpha, "k": self.k, "root_counts": self.root_counts,
            "bigram_counts": self.bigram_counts, "unigram_counts": self.unigram_counts,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path, n_roots=30_000) -> "LearnedCosts":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["root_counts"], d["bigram_counts"], d["unigram_counts"],
                   d["alpha"], d["k"], n_roots)


def _state_after(prev: str) -> str:
    """FSA state reached after `prev` ('^N' = start of a noun, else a suffix id)."""
    if prev.startswith("^"):
        return POS_START.get(prev[1:], prev)
    sf = SUFFIXES.get(prev)
    return sf.nxt if sf else prev


HAND_COSTS = CostModel()


# ---------------------------------------------------------------- analyzers --
def analyze_known(word: str, lex: Lexicon | None = None,
                  costs: CostModel | None = None) -> list[Analysis]:
    lex = lex or DEFAULT_LEXICON
    costs = costs or HAND_COSTS
    res = []
    for root in lex.candidates(word):
        chains: list[tuple] = []
        _search(word, root.form, root.flags, POS_START[root.pos], [], chains, inner=root.inner)
        for ch in chains:
            res.append(Analysis(costs.analysis_cost(root, ch), (-root.freq, root.key, ch),
                                root=root, sfx=ch))
    return sorted(res)


def analyze_guess(word: str, attested: set[str] | None = None) -> list[Analysis]:
    """Unknown-root guesser: strip a valid nominal suffix chain from the right.

    Accept only if the stem has >= 2 vowels AND (>= 2 suffixes OR stem is
    corpus-attested). Blocks 'avrupa' -> 'avrup'+DAT.
    """
    attested = attested or set()
    res = []
    for i in range(len(word) - 1, 1, -1):
        stem = word[:i]
        if n_vowels(stem) < 2:
            continue
        chains: list[tuple] = []
        _search(word, stem, frozenset(), "N", [], chains, max_depth=6)
        for ch in chains:
            if not ch:
                continue
            if len(ch) < 2 and stem not in attested:
                continue
            # prefer stripping more suffixes (shorter stem)
            res.append(Analysis(5.0 - len(ch) + 0.01 * len(stem), (stem, ch), stem=stem, sfx=ch))
    return sorted(res)


def analyze_tail(reading: str, tail: str, flags: frozenset = frozenset()) -> tuple | None:
    """Suffix chain after an apostrophe (proper noun, number, foreign word).

    If no regular chain fits (the reading is not the pronunciation: Bordeaux'ya),
    the first suffix may be a marked allomorph, S:DAT!ya; the rest stays regular.
    """
    word = reading + tail
    chains: list[tuple] = []
    _search(word, reading, flags, "N", [], chains, max_depth=6)
    chains = [c for c in chains if c]
    if not chains:
        for sid in STATES["N"]:
            for a in MARKED.get(sid, ()):
                if tail.startswith(a):
                    _search(word, reading, flags, SUFFIXES[sid].nxt,
                            [f"{sid}!{a}"], chains, max_depth=6)
    if not chains:
        return None
    return min(chains, key=lambda c: (suffix_cost(c), c))
