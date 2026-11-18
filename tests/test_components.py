"""Unit tests for the data-driven components (no downloaded data needed)."""
import pytest

from turkmorph.bpe import LexBPE
from turkmorph.em import collect_analyses, train
from turkmorph.langid import CharNgramLangId
from turkmorph.morphology import (DEFAULT_LEXICON, MARKED_IDS, Lexicon, Root, analyze_known,
                                  generate, is_suffix_id)
from turkmorph.zemberek import convert, parse_line


def _conv(line):
    r = convert(*parse_line(line))
    return r[0] if r else None


@pytest.mark.parametrize("line, form, pos, flags", [
    ("kitap", "kitap", "N", {"soften"}),                      # default voicing, polysyllabic
    ("top", "top", "N", set()),                               # monosyllable: no default voicing
    ("saat [A:InverseHarmony, NoVoicing]", "saat", "N", {"front"}),
    ("kalp [A:Voicing, InverseHarmony]", "kalp", "N", {"soften", "front"}),
    ("ağız [A:LastVowelDrop]", "ağız", "N", {"drop"}),
    ("hak [P:Noun; A:Doubling, InverseHarmony ; Index:1]", "hak", "N", {"geminate", "front"}),
    ("gelmek [A:Aorist_I]", "gel", "V", {"aor_hr"}),
    ("gitmek [A:Voicing]", "git", "V", {"soften"}),
    ("batmak", "bat", "V", {"caus_Hr"}),
    ("akşam [P:Noun, Time]", "akşam", "N", {"temporal"}),
    ("zeytinyağı [A:CompoundP3sg; Roots:zeytin-yağ]", "zeytinyağı", "N", {"cp3"}),
    ("değil [P:Verb]", "değil", "Q", set()),
    ("için [P:Postp,PCNom]", "için", "Q", set()),                # postposition: copula only
    ("ve [P:Conj]", "ve", "X", set()),
    ("iki [P:Num, Card]", "iki", "N", {"num"}),
])
def test_zemberek_convert(line, form, pos, flags):
    r = _conv(line)
    assert (r.form, r.pos, set(r.flags)) == (form, pos, flags)


def test_zemberek_skips_and_readings():
    assert parse_line("# comment") is None
    assert _conv(", [P:Punc]") is None
    assert _conv("TBMM [P:Abbrv]") is None                   # abbreviation without reading
    r = _conv("ABD [Pr:abede; P:Abbrv]")
    assert (r.form, r.pos, r.reading) == ("abd", "PROPER", "abede")


def test_cp3_takes_pronominal_n():
    zy = frozenset({"cp3"})
    assert generate("zeytinyağı", zy, ["DAT"]) == "zeytinyağına"
    assert generate("zeytinyağı", zy, ["LOC"]) == "zeytinyağında"


def test_lexicon_tsv_roundtrip(tmp_path):
    p = tmp_path / "lex.tsv"
    DEFAULT_LEXICON.to_tsv(p)
    back = Lexicon.from_tsv(p)
    assert [r.key for r in back.roots] == [r.key for r in DEFAULT_LEXICON.roots]
    assert back.by_key["kitap#N"].flags == DEFAULT_LEXICON.by_key["kitap#N"].flags


def test_lexicon_rejects_duplicate_keys():
    with pytest.raises(ValueError):
        Lexicon([Root("ev", "N"), Root("ev", "N")])


def test_homograph_tags_give_distinct_keys():
    lex = Lexicon([Root("hak", "N", frozenset({"geminate"})), Root("hak", "N", tag=".1")])
    assert set(lex.by_key) == {"hak#N", "hak#N.1"}
    assert {a.root.key for a in analyze_known("hakkı", lex)} == {"hak#N"}


def test_em_prefers_the_root_the_corpus_supports():
    # 'kalem' = kalem#N | kale#N+P1SG. Only kalem explains 'kalemler' (P1SG+PL is
    # not a valid order), so EM must move the mass of 'kalem' to kalem#N.
    lex = Lexicon([Root("kale", "N"), Root("kalem", "N")])
    words = {"kalem": 10, "kalemler": 10, "kalemde": 5}
    an = collect_analyses(words, lex, processes=1)
    assert len(an["kalem"]) == 2
    model = train([(w, c, None) for w, c in words.items()], an, lex, iters=5, log=None)
    best = min(analyze_known("kalem", lex, model))
    assert best.root.key == "kalem#N" and best.sfx == ()


def test_em_lemma_constraint():
    lex = Lexicon([Root("kale", "N"), Root("kalem", "N")])
    an = collect_analyses(["kalem"], lex, processes=1)
    model = train([("kalem", 10, {"kale"})], an, lex, iters=3, log=None)
    assert min(analyze_known("kalem", lex, model)).root.key == "kale#N"


def test_langid():
    tr = ["evler", "kitap", "geliyor", "güzel", "çocuk", "ağaç", "yıldız", "öğretmen",
          "şehir", "kalem", "okul", "masa", "kapı", "pencere", "deniz", "güneş"]
    fr = ["the", "which", "would", "through", "thought", "should", "where", "while",
          "while", "night", "light", "people", "school", "water", "with", "what"]
    lid = CharNgramLangId.train(tr, fr, min_count=1)
    assert lid.is_foreign("whether") and not lid.is_foreign("gözlük")
    assert not lid.is_foreign("ab")                          # too short to judge


def test_bpe_save_load(tmp_path):
    bpe = LexBPE().train(["komünist", "kapitalist", "sosyalist"] * 3, num_merges=20)
    bpe.save(tmp_path / "b.json")
    back = LexBPE.load(tmp_path / "b.json")
    assert back.merges == bpe.merges
    assert back.encode("realist") == bpe.encode("realist")


def test_marked_allomorphs():
    assert "DAT!ya" in MARKED_IDS and "ABL!dan" in MARKED_IDS
    assert is_suffix_id("DAT!ya") and is_suffix_id("DAT")
    assert not is_suffix_id("DAT!xyz") and not is_suffix_id("NOPE")
    assert generate("bordeaux", frozenset(), ["DAT!ya", "PCOP"]) == "bordeauxyaydı"


# ------------------------------------------------- compound P3SG (zeytinyağı) --
CP3 = Root("zeytinyağı", "N", frozenset({"cp3"}), inner="zeytinyağ|")
CP3_DROP = Root("âdemoğlu", "N", frozenset({"cp3"}), inner="âdemoğul|drop")


@pytest.mark.parametrize("root, chain, surface", [
    (CP3, [], "zeytinyağı"), (CP3, ["DAT"], "zeytinyağına"), (CP3, ["PL"], "zeytinyağları"),
    (CP3, ["PL", "LOC"], "zeytinyağlarında"), (CP3, ["P1SG"], "zeytinyağım"),
    (CP3, ["P1SG", "ABL"], "zeytinyağımdan"), (CP3_DROP, ["PL"], "âdemoğulları"),
    (CP3_DROP, ["P1SG"], "âdemoğlum"),
])
def test_compound_p3_generation(root, chain, surface):
    assert generate(root.form, root.flags, chain, root.inner) == surface


def test_compound_p3_analysis_via_inner_index():
    lex = Lexicon([CP3, CP3_DROP])
    for w, chain in [("zeytinyağları", ("PL",)), ("âdemoğullarından", ("PL", "ABL")),
                     ("zeytinyağımdan", ("P1SG", "ABL"))]:
        assert chain in {a.sfx for a in analyze_known(w, lex)}, w


def test_strip_p3():
    from turkmorph.zemberek import _strip_p3
    assert _strip_p3("düşkırıklığı") == "düşkırıklık|soften"
    assert _strip_p3("madensuyu") == "madensu|y_poss"
    assert _strip_p3("elyazması") == "elyazma|"


# ------------------------------------------------------------ merge layer --
def _units_from(tok, words):
    from collections import Counter

    from turkmorph.merge import units
    uc = Counter()
    for w in words:
        t = tok.encode(w)
        for i, j in units(t):
            uc[tuple(t[i:j])] += 1
    return uc


@pytest.mark.parametrize("mode", ["sfx", "full"])
def test_merge_layer_is_lossless(mode):
    from turkmorph.merge import MorphMerge
    from turkmorph.presets import demo_tokenizer
    from turkmorph.tokenizer import Vocab
    tok = demo_tokenizer()
    words = "evlerimizdekiler kitaplarımızdan evlerinde kitaplarında geldiğimizde".split() * 3
    tok.merge = MorphMerge.train(_units_from(tok, words), 20, mode)
    tok.cache.clear()
    tok.vocab = Vocab(tok)
    for text in ["Kitaplarında 2023'te geldiğimizde, evlerimizdekiler!", "evlerinde 12 34"]:
        toks = tok.encode(text)
        assert tok.decode(toks) == text
        assert tok.decode_ids(tok.encode_ids(text)) == text
        assert any(t.startswith("M:") for t in toks)
        if mode == "sfx":                                     # roots never merged
            assert not any(t.startswith("M:R:") for t in toks)


def test_merge_save_load_and_expand(tmp_path):
    from turkmorph.merge import MorphMerge
    from turkmorph.presets import demo_tokenizer
    tok = demo_tokenizer()
    m = MorphMerge.train(_units_from(tok, ["evlerimizde"] * 3), 5, "sfx")
    m.save(tmp_path / "m.json")
    back = MorphMerge.load(tmp_path / "m.json")
    base = tok.encode("evlerimizde")
    assert back.apply(base) == m.apply(base)
    assert MorphMerge.expand(m.apply(base)) == base


def test_encode_grouped_concatenates_to_encode():
    from turkmorph.presets import demo_tokenizer
    tok = demo_tokenizer()
    text = "Kitaplarında 2023'te, NATO'ya geldiğimizde  evlerimizdekiler!"
    groups = tok.encode_grouped(text)
    assert [t for g in groups for t in g] == tok.encode(text)
    assert ["<cap>", "R:kitap#N", "S:PL", "S:P3SG", "S:LOC"] in groups
    assert ["D:2", "D:0", "D:2", "D:3", "<apos>", "S:LOC"] in groups


# ----------------------------------------------------------- normalization --
def test_edit_op_roundtrip_property():
    import random

    from turkmorph.normalize import EDIT_CHARS, apply_edit, edit_op
    rng = random.Random(11)
    for _ in range(3000):
        c = "".join(rng.choice("abcçdeğıioöşuü") for _ in range(rng.randint(2, 10)))
        kind = rng.choice(["del", "ins", "sub", "swap"])
        p = rng.randrange(len(c) + (kind == "ins"))
        if kind == "swap" and p >= len(c) - 1:
            continue
        w = apply_edit(c, (kind, p, rng.choice(EDIT_CHARS)))
        if w == c:
            continue
        op = edit_op(c, w)
        assert op is not None and apply_edit(c, op) == w, (c, w, op)
    assert edit_op("kitaplar", "kitaplar") is None
    assert edit_op("nerede", "neden") is None                # distance 2: never a "typo"


def _norm_tok():
    from turkmorph.normalize import Normalizer
    from turkmorph.presets import demo_tokenizer
    from turkmorph.tokenizer import Vocab
    t = demo_tokenizer()
    counts = {"çocuklarımız": 500, "çocuk": 3000, "şehir": 900, "kitaplar": 800, "geliyorum": 700}
    t.normalizer = Normalizer(counts, set(counts), proper={"kemal"})
    t.vocab = Vocab(t)
    return t


@pytest.mark.parametrize("word, expected", [
    ("cocuklarimiz", "<ascii> R:çocuk#N S:PL S:P1PL"),
    ("Cocuk", "<cap> <ascii> R:çocuk#N"),
    ("kitablar", "R:kitap#N S:PL E:sub:4:b"),
    ("kitaplra", "R:kitap#N S:PL E:swap:6"),
    ("geliyorm", "R:gel#V S:PROG S:Z1SG E:del:7"),
    ("ların", "<bare> S:PL!lar S:GEN"),
    ("kitaplar", "R:kitap#N S:PL"),                          # correct words are never touched
])
def test_normalized_encoding(word, expected):
    t = _norm_tok()
    toks = t.encode(word)
    assert " ".join(toks) == expected
    assert t.decode(toks) == word and t.decode_ids(t.encode_ids(word)) == word


def test_normalizer_guards():
    t = _norm_tok()
    n = t.normalizer
    assert n.edit_target("kitap") is None or len("kitap") >= n.MIN_LEN
    assert n.edit_target("kit") is None                       # too short
    assert n.ascii_target("çocuk") is None                    # already Turkish
    from turkmorph.normalize import Normalizer
    blocked = Normalizer({"kemaller": 100}, {"kemaller"}, proper={"kemalar"})
    assert blocked.edit_target("kemalar") is None             # a name: leave it alone


def test_fuzz_roundtrip_with_normalizer():
    import random
    t = _norm_tok()
    rng = random.Random(3)
    alpha = "abcçdefgğhıijklmnoöprsştuüvyzAİIÇĞÖŞÜ  '’.,0123456789\n"
    for _ in range(1500):
        s = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 25)))
        assert t.decode(t.encode(s)) == s, repr(s)


def test_fuse_rule():
    from turkmorph.tokenizer import _fuse
    assert _fuse("ne", "olur") == "nolur"
    assert _fuse("ne", "yapıyon") == "napıyon"
    assert _fuse("ne", "haber") == "naber"
    assert _fuse("Ne", "oldu") == "Noldu"


def test_typo_prior_prefers_plausible_typos():
    from turkmorph.normalize import Normalizer, op_prior
    assert op_prior(("del", 3, ""), "geçiş", "geçş") > op_prior(("ins", 3, "ş"), "geç", "geçş")
    assert op_prior(("sub", 0, "s"), "şehir", "sehir") > op_prior(("sub", 0, "k"), "şehir", "kehir")
    # frequency alone would pick 'geç'; the prior picks the dropped vowel
    n = Normalizer({"geç": 1000, "geçiş": 500}, {"geç", "geçiş"})
    assert n.edit_target("geçş")[0] == "geçiş"


def test_two_letter_typo():
    from turkmorph.normalize import Normalizer, apply_edit
    n = Normalizer({"değerlendirme": 5000}, {"değerlendirme"})
    cand, ops = n.edit_target("değrlendrme")                   # two letters dropped
    assert cand == "değerlendirme" and len(ops) == 2
    s = cand
    for op in ops:
        s = apply_edit(s, op)
    assert s == "değrlendrme"


def test_mine_front_harmony():
    from turkmorph.zemberek import mine_front_harmony
    f = mine_front_harmony({"Kemal'e": 10, "Kemal'in": 8, "Kemal'a": 1, "Ahmet'e": 5,
                            "Ankara'da": 30, "Ankara'de": 1})
    assert f == {"kemal"}
