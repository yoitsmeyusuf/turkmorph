"""Ready-made tokenizers.

demo_tokenizer(): hand lexicon (~130 roots), hand costs, toy LEX BPE. Fast; used
                  by the unit tests and demo.py.
full_tokenizer(): Zemberek-derived lexicon + EM costs + LEX BPE + language id,
                  built by scripts/build_all.sh into data/.
"""
from __future__ import annotations

from pathlib import Path

from .bpe import LexBPE
from .tokenizer import TurkMorphTokenizer

DATA = Path(__file__).resolve().parent.parent / "data"

# LEX BPE training data: suffix-STRIPPED stems + foreign words (never inflected TR words).
LEX_TRAIN = """
komünist sosyalist kapitalist teknoloji biyoloji sosyoloji demokrasi danimarka
norveç almanya avrupa amerika meeting meetings government postponed policy the
was and of to in communism capitalism socialism deploy deployment shakespeare
bordeaux gobierno el la los las que comunista político política economía
computer network model training transformer tokenizer madrid barcelona
""".split() * 3


def demo_tokenizer() -> TurkMorphTokenizer:
    return TurkMorphTokenizer(LexBPE().train(LEX_TRAIN, num_merges=300),
                              attested_stems={"komünist"})


def full_tokenizer(data_dir=DATA, costs: str = "costs.json") -> TurkMorphTokenizer:
    from .morphology import LearnedCosts, Lexicon

    data_dir = Path(data_dir)
    lex = Lexicon.from_tsv(data_dir / "lexicon.tsv")
    cost_model = None
    if costs and (data_dir / costs).exists():
        cost_model = LearnedCosts.from_json(data_dir / costs, n_roots=len(lex))
    bpe_path = data_dir / "lex_bpe.json"
    bpe = LexBPE.load(bpe_path) if bpe_path.exists() else LexBPE().train(LEX_TRAIN, 300)
    foreign = None
    if (data_dir / "langid.json").exists():
        from .langid import CharNgramLangId
        foreign = CharNgramLangId.load(data_dir / "langid.json").is_foreign
    attested = set()
    if (data_dir / "attested_stems.txt").exists():
        attested = set((data_dir / "attested_stems.txt").read_text(encoding="utf-8").split())
    normalizer = None
    if (data_dir / "wordfreq.tsv").exists():
        from .normalize import Normalizer
        normalizer = Normalizer.load(data_dir / "wordfreq.tsv", data_dir / "proper_words.txt", foreign)
    tok = TurkMorphTokenizer(bpe, attested, lexicon=lex, costs=cost_model, foreign=foreign,
                             normalizer=normalizer)
    if (data_dir / "unions.txt").exists():
        tok.set_unions((data_dir / "unions.txt").read_text(encoding="utf-8").split())
    if (data_dir / "overrides.tsv").exists():
        for line in (data_dir / "overrides.tsv").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                w, key, chain, _src = line.split("\t")
                tok.overrides[w] = (key, tuple(chain.split()) if chain else ())
    return tok
