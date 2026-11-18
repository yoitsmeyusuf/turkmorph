"""Full tokenizer (Zemberek lexicon + EM + LEX BPE + langid). Needs data/ built by
scripts/build_all.sh; skipped otherwise."""
import random
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
pytestmark = pytest.mark.skipif(not (DATA / "lexicon.tsv").exists(),
                                reason="run scripts/build_all.sh first")


@pytest.fixture(scope="module")
def full_py():
    from turkmorph.presets import full_tokenizer
    return full_tokenizer()


@pytest.fixture(scope="module", params=["python", "rust"])
def full(request):
    """Every test on `full` runs on both runtimes (rust: turkmorph.fast)."""
    if request.param == "python":
        return request.getfixturevalue("full_py")
    fast = pytest.importorskip("turkmorph.fast", reason="run scripts/build_fast.sh first")
    return fast.fast_tokenizer()


def _ud_test_texts(n_per_file=150):
    from turkmorph.corpus import read_conllu, ud_files
    out = []
    for f in ud_files(DATA / "raw" / "ud", "test", include_noncommercial=True):
        out += [s.text for s in read_conllu(f)[:n_per_file] if s.text]
    return out


def test_roundtrip_ud_test_sentences(full):
    texts = _ud_test_texts()
    if not texts:
        pytest.skip("UD treebanks not downloaded (data/raw/ud; scripts/build_all.sh)")
    assert len(texts) > 500
    for t in texts:
        assert full.decode(full.encode(t)) == t, t


def test_fuzz_roundtrip(full):
    rng = random.Random(99)
    alpha = "abcçdefgğhıijklmnoöprsştuüvyzAİIÇĞÖŞÜ  '’.,0123456789\n\t😀-"
    for _ in range(1000):
        s = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 25)))
        assert full.decode(full.encode(s)) == s, repr(s)


@pytest.mark.parametrize("word, expected", [
    ("evlerimizdekiler", "R:ev#N S:PL S:P1PL S:LOC S:REL S:PL"),
    ("öğretmenlerimizden", "R:öğretmen#N S:PL S:P1PL S:ABL"),
    ("kalktığımızda", "R:kalk#V S:PART_DHK S:P1PL S:LOC"),
    ("gidiyorum", "R:git#V S:PROG S:Z1SG"),
    ("şehrimizde", "R:şehir#N S:P1PL S:LOC"),
    ("hakkında", "R:hak#N S:P3SG S:LOC"),
    ("için", "R:için#Q"),
    ("içindir", "R:için#Q S:COP3"),
    ("kendini", "R:kendi#PRON S:ACC"),
    ("birbirlerine", "R:birbiri#PRON S:PL S:DAT"),     # P2SG reading no longer near-equal (no DAT|P2SG+DAT union)
    ("ikinci", "R:iki#N S:ORD"),
    ("Beyoğlu'nda", "<cap> R:beyoğlu#PROPER <apos> S:LOC"),
    ("cocuklarimiz", "<ascii> R:çocuk#N S:PL S:P1PL"),
    ("tesbit", "R:tespit#N E:sub:3:b"),
    ("geliyom", "R:gel#V S:PROG_C S:Z1SG_C"),
    ("nerde", "R:nere#N S:LOC_C"),
    ("Kemal'e", "<cap> R:kemal#PROPER <apos> S:DAT"),
    ("anneciğim", "R:anne#N S:DIM S:P1SG"),
    ("ikisinide", "R:iki#N S:P3SG S:ACC <glue> R:de#X"),
    ("nolur", "R:ne#PRON <fuse> R:ol#V S:AOR"),
    ("naber", "R:ne#PRON <fuse> R:haber#N"),
    ("malesef", "R:maalesef#N E:del:2"),
    ("Danimarka'dan", "<cap> R:danimarka#PROPER <apos> S:ABL"),
    ("TBMM'ye", "<capsb> L:▁tbmm <apos> S:DAT"),
])
def test_full_segmentations(full, word, expected):
    toks = full.encode(word)
    assert full.decode(toks) == word
    if word == "TBMM'ye":                                     # LEX pieces depend on BPE
        assert toks[0] == "<capsb>" and toks[-2:] == ["<apos>", "S:DAT"]
    else:
        assert " ".join(toks) == expected


def test_every_token_has_an_id(full):
    for t in _ud_test_texts(30):
        full.encode_ids(t)


def test_vocab_budget(full_py):
    sizes = {k: b - a for k, (a, b) in full_py.vocab.ranges.items()}
    assert 25_000 < sizes["ROOT"] < 40_000
    assert sizes["LEX"] > 4_000


def test_context_tokenizer_lossless(full_py):
    pytest.importorskip("torch")
    if not (DATA / "context.pt").exists():
        pytest.skip("run scripts/train_context.py first")
    from turkmorph.context import ContextTokenizer
    ct = ContextTokenizer(full_py, DATA / "context.pt")
    for t in _ud_test_texts(40) + ["Kitabı okudum.", "Onun kitabı masada.", "Yüzü güldü."]:
        assert ct.decode(ct.encode(t)) == t, t
    toks = ct.encode("Yüzü güldü.")                          # a lowercase word never gets a name root
    assert not any(x.endswith("#PROPER") for x in toks)
