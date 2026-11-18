"""How much would a context model buy? Classify ambiguity of UD test tokens and
score the context-free ACC/P3SG choice against UD morphological features."""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_em import cached_analyses                           # noqa: E402
from turkmorph.corpus import read_conllu, ud_files             # noqa: E402
from turkmorph.morphology import LearnedCosts, Lexicon, generate  # noqa: E402
from turkmorph.tokenizer import tr_lower                       # noqa: E402

DATA = ROOT / "data"
WORD = re.compile(r"^[^\W\d_]+$")


def pieces(root, chain):
    """Surface boundaries of an analysis: lengths after each morpheme."""
    return tuple(len(generate(root.form, root.flags, list(chain[:i]))) for i in range(len(chain) + 1))


def main():
    lex = Lexicon.from_tsv(DATA / "lexicon.tsv")
    costs = LearnedCosts.from_json(DATA / "costs.json", len(lex))
    toks = [t for f in ud_files(DATA / "raw" / "ud", "test", include_noncommercial=True) for s in read_conllu(f)
            for t in s.tokens if WORD.match(t.form)]
    an = cached_analyses({tr_lower(t.form) for t in toks}, lex, "test")
    kind = Counter()
    accp3 = Counter()
    for t in toks:
        cands = an.get(tr_lower(t.form)) or []
        if not cands:
            kind["no analysis"] += 1
            continue
        scored = sorted((costs.analysis_cost(lex.by_key[k], ch), k, ch) for k, ch in cands)
        top = [(lex.by_key[k], ch) for c, k, ch in scored if c - scored[0][0] < 2.3]   # within 10x
        if len(top) == 1:
            kind["unambiguous (or clear winner)"] += 1
        else:
            segs = {(r.form, pieces(r, ch)) for r, ch in top}
            kind["label only (same boundaries)" if len(segs) == 1 else "segmentation"] += 1
        best = lex.by_key[scored[0][1]], scored[0][2]
        last = best[1][-1] if best[1] else None
        if last in ("ACC", "P3SG") and ("ACC" in [c[-1] for _, c in top if c] and
                                        "P3SG" in [c[-1] for _, c in top if c]):
            gold = "ACC" if "Case=Acc" in t.feats else \
                   "P3SG" if "Person[psor]=3" in t.feats and "Case=Nom" in t.feats else "other"
            accp3[(last, gold)] += 1
    n = sum(kind.values())
    for k, v in kind.most_common():
        print(f"{k:32s} {v:6d}  {v / n:6.1%}")
    tot = sum(accp3.values())
    right = accp3[("ACC", "ACC")] + accp3[("P3SG", "P3SG")]
    print(f"\nACC/P3SG-ambiguous tokens: {tot} ({tot / n:.1%} of all)")
    for k, v in sorted(accp3.items()):
        print(f"  chose {k[0]:5s} gold {k[1]:6s} {v}")
    print(f"  context-free choice correct: {right / max(tot, 1):.1%}  "
          f"-> at most {(tot - right) / n:.2%} of tokens get a different label with context")


if __name__ == "__main__":
    main()
