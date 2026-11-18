"""decode(encode(x)) == x is an invariant, not a hope."""
import random

import pytest

SENTENCES = [
    "Kalktığımızda hep birlikte yürüdük.",
    "Evlerimizdekiler, Danimarka'dan 2023'te geldi!",
    "The meeting was postponed; el gobierno  dijo  que no.\n\tYeni satır (parantez) \"tırnak\" ☃ emoji 🙂",
    "  baştaki boşluk ve sondaki   ",
    "İSTANBUL'DA Iğdır ısı İzmir",
    "", " ", "\n\n", "a", "İ", "I", "ıI",
    # regression: bare numbers separated by one space used to fuse (12 34 -> 1234)
    "12 34", "1 2 3", "2023'te 5 6", "3 4\n5 6",
    # odd whitespace, combining marks, apostrophes, mixed case
    "x y", "café", "rock'n'roll", "iPhone'u", "ß STRASSE", "ǅemal",
    "a''b", "'ler", "5'", "a'", " ", "😀a", "a😀", "Ankara’da", "ANKARA'DA",
    "x\r\ny", "https://example.com/a?b=1",
]


@pytest.mark.parametrize("text", SENTENCES)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text
    assert tok.decode_ids(tok.encode_ids(text)) == text


def test_numbers_do_not_fuse(tok):
    assert tok.decode(tok.encode("12 34")) == "12 34"
    assert "<sp>" in tok.encode("12 34")
    assert "<sp>" not in tok.encode("a 12")          # implicit space still used elsewhere


def test_fuzz_roundtrip(tok):
    rng = random.Random(1234)
    alpha = "abcçdefgğhıijklmnoöprsştuüvyzAİIÇĞÖŞÜ  '’.,0123456789\n\t é😀-"
    for _ in range(3000):
        s = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 25)))
        assert tok.decode(tok.encode(s)) == s, repr(s)


def test_fuzz_words(tok):
    """Concatenations of Turkish-looking syllables exercise analysis + guesser."""
    rng = random.Random(7)
    syl = ["ev", "ler", "de", "ki", "kitap", "lar", "ım", "ız", "dan", "ta", "a", "ı",
           "gel", "di", "yor", "um", "ken", "sı", "nın", "le", "miş", "cek", "ğı", "bı"]
    for _ in range(1500):
        w = "".join(rng.choice(syl) for _ in range(rng.randint(1, 5)))
        text = rng.choice([w, w.capitalize(), w.upper(), w + "'" + rng.choice(syl)])
        assert tok.decode(tok.encode(text)) == text, text


def test_cache_is_transparent(tok):
    text = "kitaplarımızdan ve Danimarka'dan"
    first = tok.encode(text)
    assert tok.encode(text) == first
    tok.cache.clear()
    assert tok.encode(text) == first


def test_deterministic(tok):
    from turkmorph.presets import demo_tokenizer
    other = demo_tokenizer()
    text = "Kalktığımızda evlerimizdekiler kitabımızı 6'dan önce getirdi."
    assert other.encode(text) == tok.encode(text)
