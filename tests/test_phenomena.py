"""Phenomenon regression set: segmentation must be lossless AND match the gold
analysis. Grows with the lexicon; see the design doc, 'Hemen sonraki adım'."""
from collections import Counter
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data" / "phenomena.tsv"


def _rows():
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            cat, word, expected = line.split("\t")
            yield pytest.param(word, expected.split(), id=f"{cat}:{word}")


ROWS = list(_rows())


@pytest.mark.parametrize("word, expected", ROWS)
def test_phenomenon(tok, word, expected):
    toks = tok.encode(word)
    assert tok.decode(toks) == word
    assert toks == expected


def test_every_category_is_covered():
    cats = Counter(p.id.split(":")[0] for p in ROWS)
    assert len(ROWS) >= 200
    for cat in ["yumusama", "dusme", "ikizlesme", "uyum_istisnasi", "supletif",
                "ozel_isim", "sayi"]:
        assert cats[cat] >= 7, cat
