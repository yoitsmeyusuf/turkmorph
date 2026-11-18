"""Underspecified suffix tokens seen in the corpus -> data/unions.txt (one id per line).

For every BellaTurca / UD word type: best analysis + near-equal alternatives with the
same root and word -> union ids (kitab+ı: P3SG|ACC). Kept if they cover >= MIN tokens."""
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from turkmorph.morphology import Analysis, LearnedCosts, Lexicon, underspecify   # noqa: E402

DATA = ROOT / "data"
MIN = 5


def main():
    lex = Lexicon.from_tsv(DATA / "lexicon.tsv")
    costs = LearnedCosts.from_json(DATA / "costs.json", len(lex))
    counts = pickle.load(open(DATA / "cache" / "bt_counts.pkl", "rb"))["words"]
    an = pickle.load(open(DATA / "cache" / "analyses_bt.pkl", "rb"))["analyses"]
    tr = pickle.load(open(DATA / "cache" / "analyses_train.pkl", "rb"))["analyses"]
    an.update({w: a for w, a in tr.items() if w not in an})
    seen = Counter()
    for w, cands in an.items():
        if len(cands) < 2:
            continue
        objs = sorted(Analysis(costs.analysis_cost(lex.by_key[k], ch), (k, ch), root=lex.by_key[k], sfx=ch)
                      for k, ch in cands)
        objs.sort(key=lambda a: a.root.pos == "PROPER")
        for sid in underspecify(objs[0], objs[1:]):
            if "|" in sid:
                seen[sid] += max(1, counts.get(w, 1))
    keep = [u for u, c in seen.most_common() if c >= MIN]
    (DATA / "unions.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
    tot = sum(seen.values())
    print(f"unions: {len(seen)} seen, {len(keep)} kept (>= {MIN} tokens), covering "
          f"{sum(seen[u] for u in keep) / max(tot, 1):.2%} of union tokens")
    print("  top: " + ", ".join(f"{u} ({c})" for u, c in seen.most_common(12)))


if __name__ == "__main__":
    main()
