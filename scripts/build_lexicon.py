"""data/raw (Zemberek + UD) -> data/lexicon.tsv"""
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turkmorph.corpus import read_conllu, ud_files            # noqa: E402
from turkmorph.morphology import Lexicon                      # noqa: E402
from turkmorph.zemberek import build_lexicon, tr_lower        # noqa: E402

RAW = ROOT / "data" / "raw"
BT_PROPER_MIN = 100                                           # keeps ROOT near the ~30K+names budget


def proper_evidence() -> Counter:
    """Bases seen capitalized mid-sentence or before an apostrophe in UD train."""
    c = Counter()
    for f in ud_files(RAW / "ud", "train"):
        for s in read_conllu(f):
            for i, t in enumerate(s.tokens):
                base, *rest = re.split(r"['’]", t.form)
                if base and (rest or (i > 0 and base[0].isupper())):
                    c[tr_lower(base)] += 1
    return c


def main():
    counts = proper_evidence()
    bt = ROOT / "data" / "cache" / "bt_counts.pkl"
    front, words = set(), None
    if bt.exists():                                           # BellaTurca: frequent names only;
        btc = pickle.load(open(bt, "rb"))
        for w, c in btc["proper"].items():                    # rare ones go to LEX
            if c >= BT_PROPER_MIN:
                counts[w] += c
        from turkmorph.zemberek import mine_front_harmony
        front = mine_front_harmony({k: v for k, v in btc["chunks"].items() if "'" in k or "’" in k})
        words = {w: c for w, c in btc["words"].items() if c >= 3}
        print(f"  names with front-vowel suffixes (Kemal'e) mined from BellaTurca: {len(front)}")
    z = lambda n: RAW / f"zemberek-{n}.dict"                  # noqa: E731
    roots = build_lexicon(
        z("master-dictionary"), extra=[z("non-tdk")],
        proper=[z("proper"), z("locations-tr"), z("person-names"), z("abbreviations"),
                z("proper-from-corpus")],
        keep_proper=lambda form: counts[form] > 0, front_names=front, corpus_counts=words)
    lex = Lexicon(roots)
    out = ROOT / "data" / "lexicon.tsv"
    lex.to_tsv(out)
    print(f"{len(lex)} roots -> {out}")
    print(Counter(r.pos for r in roots))
    print(Counter(f for r in roots for f in r.flags).most_common())


if __name__ == "__main__":
    main()
