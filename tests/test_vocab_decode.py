import random

import pytest


def test_namespaces_disjoint_and_cover(tok):
    v = tok.vocab
    spans = sorted(v.ranges.values())
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b <= c
    assert len(set(v.toks)) == len(v.toks)
    assert v.namespace(v.id("<cap>")) == "SPECIAL"
    assert v.namespace(v.id("S:ABL")) == "SFX"
    assert v.namespace(v.id("R:ev#N")) == "ROOT"


def test_lex_dan_and_suffix_abl_never_share_ids(tok):
    v = tok.vocab
    toks = tok.encode("Danimarka'dan komünistlerden")
    abl = [v.id(t) for t in toks if t == "S:ABL"]
    assert len(abl) == 2 and len(set(abl)) == 1
    assert all(v.namespace(v.id(t)) != "SFX" for t in toks if t.startswith("L:"))


def test_every_encoded_token_has_an_id(tok):
    rng = random.Random(3)
    alpha = "abcçdefgğhıijklmnoöprsştuüvyzAİIÇĞÖŞÜQWX  '.,;:!?0123456789\n\t☃😀"
    for _ in range(500):
        s = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 30)))
        for t in tok.encode(s):
            tok.vocab.id(t)


def test_mixed_case_stays_in_lex(tok):
    assert not any(t.startswith("B:") for t in tok.encode("iPhone"))


@pytest.mark.parametrize("toks", [
    ["S:PL"], ["S:PL", "S:ABL"], ["L:x"], ["B:41"], ["<apos>", "S:LOC"], ["<lit>", "L:a"],
    ["R:ev#N", "<glue>", "S:PL"], ["<raw>", "B:ff"], ["L:▁ka", "B:c4"],
])
def test_decode_is_total_on_ungrammatical_input(tok, toks):
    assert isinstance(tok.decode(toks), str)


def test_decode_rejects_non_tokens(tok):
    for bad in (["S:NOPE"], ["R:yok#N"], ["?"]):
        with pytest.raises(ValueError):
            tok.decode(bad)


def test_decode_random_id_streams(tok):
    """An LM can emit any id sequence; decoding must never crash."""
    rng = random.Random(5)
    n = len(tok.vocab)
    for _ in range(2000):
        tok.decode_ids([rng.randrange(n) for _ in range(rng.randint(0, 12))])


def test_decode_rejects_out_of_range_ids(tok):
    for bad in (-1, -len(tok.vocab), len(tok.vocab)):
        with pytest.raises(ValueError):
            tok.decode_ids([bad])
