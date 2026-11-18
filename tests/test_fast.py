"""The Rust runtime (turkmorph.fast) must give the Python full tokenizer's tokens,
bit for bit, and decode the same way. Needs data/ and `bash scripts/build_fast.sh`."""
import random
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
pytestmark = pytest.mark.skipif(not (DATA / "lexicon.tsv").exists(),
                                reason="run scripts/build_all.sh first")
fast = pytest.importorskip("turkmorph.fast", reason="run scripts/build_fast.sh first")

from test_full import _ud_test_texts                      # noqa: E402
from test_phenomena import ROWS                           # noqa: E402
from test_roundtrip import SENTENCES                      # noqa: E402


@pytest.fixture(scope="module")
def py():
    from turkmorph.presets import full_tokenizer
    return full_tokenizer()


@pytest.fixture(scope="module")
def rs():
    return fast.fast_tokenizer()


def _fuzz_texts(seed, n):
    rng = random.Random(seed)
    alpha = "abcçdefgğhıijklmnoöprsştuüvyzAİIÇĞÖŞÜ  '’.,0123456789\n\t é😀-ßΣσ̇"
    return ["".join(rng.choice(alpha) for _ in range(rng.randint(0, 25))) for _ in range(n)]


def _fuzz_words(seed, n):
    rng = random.Random(seed)
    syl = ["ev", "ler", "de", "ki", "kitap", "lar", "ım", "ız", "dan", "ta", "a", "ı",
           "gel", "di", "yor", "um", "ken", "sı", "nın", "le", "miş", "cek", "ğı", "bı"]
    out = []
    for _ in range(n):
        w = "".join(rng.choice(syl) for _ in range(rng.randint(1, 5)))
        out.append(rng.choice([w, w.capitalize(), w.upper(), w + "'" + rng.choice(syl)]))
    return out


def test_same_vocabulary(py, rs):
    assert rs.vocab.toks == py.vocab.toks


@pytest.mark.parametrize("name, texts", [
    ("sentences", lambda: SENTENCES),
    ("phenomena", lambda: [p.values[0] for p in ROWS]),
    ("ud_test", lambda: _ud_test_texts()),
    ("fuzz", lambda: _fuzz_texts(42, 3000)),
    ("fuzz_words", lambda: _fuzz_words(43, 3000)),
])
def test_same_tokens_and_lossless(py, rs, name, texts):
    for t in texts():
        toks = rs.encode(t)
        assert toks == py.encode(t), t
        assert rs.decode(toks) == t, t
        assert rs.decode_ids(rs.encode_ids(t)) == t, t


def test_batch_equals_sequential(rs):
    texts = _ud_test_texts(60) + _fuzz_words(44, 2000)
    assert rs.encode_batch(texts, threads=6) == [rs.encode_ids(t) for t in texts]
    assert rs.encode_batch([], threads=6) == []


def test_decode_random_id_streams_like_python(py, rs):
    rng = random.Random(5)
    n = len(rs.vocab)
    for _ in range(3000):
        ids = [rng.randrange(n) for _ in range(rng.randint(0, 12))]
        assert rs.decode_ids(ids) == py.decode_ids(ids), ids


def test_decode_rejects_non_tokens(rs):
    for bad in (["S:NOPE"], ["?"]):
        with pytest.raises(ValueError):
            rs.decode(bad)
    with pytest.raises(ValueError):
        rs.decode_ids([len(rs.vocab)])
