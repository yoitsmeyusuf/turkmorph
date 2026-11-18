import pytest

from turkmorph.phonology import drop_last_vowel, number_reading, realize, soften


@pytest.mark.parametrize("tmpl, stem, front, out", [
    ("lAr", "ev", False, "ler"), ("lAr", "kitap", False, "lar"),
    ("(H)m", "göz", False, "üm"), ("(H)m", "araba", False, "m"),
    ("DA", "kitap", False, "ta"), ("DA", "ev", False, "de"),
    ("CH", "ağaç", False, "çı"), ("(y)A", "araba", False, "ya"),
    ("(s)H", "araba", False, "sı"), ("(s)H", "ev", False, "i"),
    ("(n)Hn", "masa", False, "nın"), ("(y)DH", "hasta", False, "ydı"),
    ("H", "saat", True, "i"), ("lAr", "saat", True, "ler"),
    ("(H)yor", "gel", False, "iyor"), ("(y)ken", "ev", False, "ken"),
])
def test_realize(tmpl, stem, front, out):
    assert realize(tmpl, stem, front) == out


@pytest.mark.parametrize("s, out", [("kitap", "kitab"), ("ağaç", "ağac"), ("renk", "reng"),
                                     ("köpek", "köpeğ"), ("git", "gid"), ("ev", "ev")])
def test_soften(s, out):
    assert soften(s) == out


@pytest.mark.parametrize("s, out", [("ağız", "ağz"), ("burun", "burn"), ("şehir", "şehr")])
def test_drop_last_vowel(s, out):
    assert drop_last_vowel(s) == out


@pytest.mark.parametrize("digits, word", [
    ("0", "sıfır"), ("3", "üç"), ("10", "on"), ("40", "kırk"), ("2023", "üç"),
    ("100", "yüz"), ("1000", "bin"), ("2000000", "milyon"), ("3000000000", "milyar"),
    ("007", "yedi"),
])
def test_number_reading(digits, word):
    assert number_reading(digits) == word
