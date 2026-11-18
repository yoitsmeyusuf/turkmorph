"""Turkish morphophonology: archiphoneme realization + lexical alternations.

Archiphonemes in suffix templates:
  A -> a/e        (2-way harmony)
  H -> ı/i/u/ü    (4-way harmony)
  D -> d/t        (voicing assimilation after hard consonant)
  C -> c/ç
  G -> g/k
  (y) (n) (s) (ş) -> buffer consonant, present only after a vowel
  (H)             -> buffer vowel, present only after a consonant
  lowercase       -> literal (does not harmonize, e.g. 'yor', 'ki')

Irregularity never lives in tokens. It lives in:
  * root flags  (soften, drop, geminate, front, aor_hr, narrow_y, y_poss, pron_n)
  * suffix flags (soften: -DHk, -(y)AcAk, -lHk)
  * a small suppletion table (ben+DAT -> bana)
"""
from __future__ import annotations

VOWELS = set("aeıioöuüâîû")
BACK = set("aıouâû")
ROUND = set("oöuüû")
HARD = set("çfhkpsşt")

SOFTEN = {"p": "b", "ç": "c", "t": "d", "k": "ğ", "g": "ğ"}   # psikolog -> psikoloğu


def last_vowel(s: str):
    for ch in reversed(s):
        if ch in VOWELS:
            return ch
    return None


def n_vowels(s: str) -> int:
    return sum(ch in VOWELS for ch in s)


def soften(s: str) -> str:
    if not s:
        return s
    c = s[-1]
    if c == "k" and len(s) >= 2 and s[-2] == "n":
        return s[:-1] + "g"          # renk -> rengi
    if c in SOFTEN:
        return s[:-1] + SOFTEN[c]
    return s


def drop_last_vowel(s: str) -> str:
    for i in range(len(s) - 1, -1, -1):
        if s[i] in VOWELS:
            return s[:i] + s[i + 1:]  # ağız -> ağz
    return s


def _H(v, front_override=False):
    back = (v in BACK) if v else False
    if front_override:
        back = False
    rnd = (v in ROUND) if v else False
    if back:
        return "u" if rnd else "ı"
    return "ü" if rnd else "i"


def _A(v, front_override=False):
    back = (v in BACK) if v else False
    if front_override:
        back = False
    return "a" if back else "e"


def realize(tmpl: str, s: str, front_override: bool) -> str:
    """Turn an archiphonemic template into a surface piece appended to `s`.

    front_override: the harmony trigger is still a root vowel of a root flagged
    'front' (saat, rol, kalp) -> suffix vowels are front.
    """
    out = ""
    i = 0
    while i < len(tmpl):
        cur = s + out
        ends_v = bool(cur) and cur[-1] in VOWELS
        if tmpl[i] == "(":
            j = tmpl.index(")", i)
            inner = tmpl[i + 1:j]
            if inner == "H":
                if not ends_v:
                    out += _harm_H(s, out, front_override)
            elif ends_v:
                out += inner
            i = j + 1
            continue
        c = tmpl[i]
        if c == "A":
            v, fo = _trigger(s, out, front_override)
            out += _A(v, fo)
        elif c == "H":
            out += _harm_H(s, out, front_override)
        elif c == "D":
            out += "t" if cur and cur[-1] in HARD else "d"
        elif c == "C":
            out += "ç" if cur and cur[-1] in HARD else "c"
        elif c == "G":
            out += "k" if cur and cur[-1] in HARD else "g"
        else:
            out += c
        i += 1
    return out


def _trigger(s, out, front_override):
    v = last_vowel(out)
    if v:
        return v, False
    return last_vowel(s), front_override


def _harm_H(s, out, front_override):
    v, fo = _trigger(s, out, front_override)
    return _H(v, fo)


# --- abbreviations: a vowelless one is read letter by letter (TBMM'ye) ------
LETTER_NAMES = {"a": "a", "b": "be", "c": "ce", "ç": "çe", "d": "de", "e": "e", "f": "fe",
                "g": "ge", "ğ": "yumuşakge", "h": "he", "ı": "ı", "i": "i", "j": "je",
                "k": "ke", "l": "le", "m": "me", "n": "ne", "o": "o", "ö": "ö", "p": "pe",
                "r": "re", "s": "se", "ş": "şe", "t": "te", "u": "u", "ü": "ü", "v": "ve",
                "y": "ye", "z": "ze", "q": "kü", "w": "dabılyu", "x": "iks"}


def spelled_reading(base: str) -> str:
    """Reading used for suffix harmony after an apostrophe: the base itself, or
    its letter names when it has no vowel (tbmm -> tebemem)."""
    if n_vowels(base) or not base or any(c not in LETTER_NAMES for c in base):
        return base
    return "".join(LETTER_NAMES[c] for c in base)


# --- numerals: harmony follows pronunciation of the last number word -------
_UNITS = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def number_reading(digits: str) -> str:
    n = int(digits)
    if n == 0:
        return "sıfır"
    if n % 10:
        return _UNITS[n % 10]
    if (n // 10) % 10:
        return _TENS[(n // 10) % 10]
    if (n // 100) % 10:
        return "yüz"
    if (n // 1000) % 1000:
        return "bin"
    if (n // 10**6) % 1000:
        return "milyon"
    return "milyar"
