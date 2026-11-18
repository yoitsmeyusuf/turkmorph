"""UD train -> data/costs_em.json (unsupervised) + data/costs_em_lemma.json (lemma-constrained);
with BellaTurca counts also data/costs_bt.json (BellaTurca unsupervised + UD lemma-constrained)."""
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turkmorph import em                                       # noqa: E402
from turkmorph.corpus import read_conllu, ud_files             # noqa: E402
from turkmorph.morphology import Lexicon                       # noqa: E402
from turkmorph.tokenizer import tr_lower                       # noqa: E402

DATA = ROOT / "data"
WORD = re.compile(r"^[^\W\d_]+$")


def ud_items(split):
    """Counter over (lowercased word, lowercased gold lemma) for plain alphabetic tokens."""
    c = Counter()
    for f in ud_files(DATA / "raw" / "ud", split):
        for s in read_conllu(f):
            for t in s.tokens:
                if WORD.match(t.form):
                    c[(tr_lower(t.form), tr_lower(t.lemma))] += 1
    return c


UD_WEIGHT = 100                                                # gold-lemma tokens vs 118M raw tokens (x20: -0.4 root acc)
BT_MIN_COUNT = 3


def bt_words():
    p = DATA / "cache" / "bt_counts.pkl"
    if not p.exists():
        return None
    return {w: c for w, c in pickle.load(open(p, "rb"))["words"].items() if c >= BT_MIN_COUNT}


def cached_analyses(words, lex, name):
    path = DATA / "cache" / f"analyses_{name}.pkl"
    path.parent.mkdir(exist_ok=True)
    stamp = (DATA / "lexicon.tsv").stat().st_mtime
    if path.exists():
        d = pickle.loads(path.read_bytes())
        if d["stamp"] == stamp and set(words) <= d["analyses"].keys():
            return d["analyses"]
    a = em.collect_analyses(words, lex)
    path.write_bytes(pickle.dumps({"stamp": stamp, "analyses": a}))
    return a


def main():
    lex = Lexicon.from_tsv(DATA / "lexicon.tsv")
    items = ud_items("train")
    words = {w for w, _ in items}
    print(f"train: {sum(items.values())} tokens, {len(words)} types")
    analyses = cached_analyses(words, lex, "train")
    print(f"  covered types: {sum(bool(analyses[w]) for w in words)}/{len(words)}")

    by_word = Counter()
    for (w, _), c in items.items():
        by_word[w] += c
    print("unsupervised EM")
    m = em.train([(w, c, None) for w, c in by_word.items()], analyses, lex)
    m.to_json(DATA / "costs_em.json")
    print("lemma-constrained EM")
    m = em.train([(w, c, {lem}) for (w, lem), c in items.items()], analyses, lex)
    m.to_json(DATA / "costs_em_lemma.json")
    m_lemma = m

    bt = bt_words()
    if bt:
        print(f"BellaTurca: {sum(bt.values())/1e6:.1f}M tokens, {len(bt)} types (count >= {BT_MIN_COUNT})")
        an_bt = cached_analyses(set(bt), lex, "bt")
        an_bt.update(analyses)
        print(f"  covered types: {sum(bool(an_bt[w]) for w in bt)}/{len(bt)}, "
              f"tokens: {sum(c for w, c in bt.items() if an_bt[w]) / sum(bt.values()):.3f}")
        print("BellaTurca EM + UD lemma-constrained")
        m = em.train([(w, c, None) for w, c in bt.items()] +
                     [(w, c * UD_WEIGHT, {lem}) for (w, lem), c in items.items()], an_bt, lex,
                     init=m_lemma)                    # start from the lemma-supervised model
        m.to_json(DATA / "costs_bt.json")


if __name__ == "__main__":
    main()
