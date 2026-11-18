"""Frequency audit -> data/overrides.tsv (word -> analysis the tokenizer must prefer).

Two sources, both only among the analyzer's own candidates (so every override is
still generated and verified, i.e. lossless):
  ud    UD train+dev majority lemma differs from our choice and a candidate has it
        (UD test is never used: evaluation stays clean)
  rare  frequent corpus word whose chosen noun root is almost never seen on its own
        (ayrıcalık -> 'ayrıç'), while another candidate's root is common (ayrı)
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from errors import stems                                        # noqa: E402
from turkmorph.corpus import read_conllu, ud_files              # noqa: E402
from turkmorph.morphology import analyze_known                  # noqa: E402
from turkmorph.presets import full_tokenizer                    # noqa: E402
from turkmorph.tokenizer import tr_lower                        # noqa: E402

DATA = ROOT / "data"
WORD = re.compile(r"^[^\W\d_]+$")
TOP_TYPES = 50_000


def main():
    tok = full_tokenizer()
    tok.overrides = {}                                          # audit the raw choices
    lex, costs, counts = tok.lex, tok.costs, tok.normalizer.counts
    lemmas = defaultdict(Counter)
    for split in ("train", "dev"):
        for f in ud_files(DATA / "raw" / "ud", split):
            for s in read_conllu(f):
                for t in s.tokens:
                    if WORD.match(t.form):
                        lemmas[tr_lower(t.form)][tr_lower(t.lemma)] += 1

    out, log = {}, Counter()
    ex = defaultdict(list)
    # --- UD majority
    for w, lc in lemmas.items():
        n = sum(lc.values())
        lemma, k = lc.most_common(1)[0]
        if n < 2 or k / n < 0.6 or lemma == w:            # UD keeps frozen forms whole (özellikle,
            continue                                        # hızla): a convention, not our error
        cands = analyze_known(w, lex, costs)
        if not cands:
            continue
        best = min(cands, key=lambda a: (a.root.pos == "PROPER", a.cost))
        if lemma in stems(best.root, list(best.sfx)):
            continue
        good = [a for a in cands if lemma in stems(a.root, list(a.sfx))]
        if good:
            g = min(good, key=lambda a: a.cost)
            out[w] = (g.root.key, g.sfx, "ud")
            log["ud"] += 1
            ex["ud"].append((n, f"{w}: {best.root.key}+{'+'.join(best.sfx)} -> {g.root.key}+{'+'.join(g.sfx)}"))

    # --- rare noun roots in frequent words
    for w, c in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_TYPES]:
        if w in out or c < 50:
            continue
        cands = analyze_known(w, lex, costs)
        if len(cands) < 2:
            continue
        best = min(cands, key=lambda a: (a.root.pos == "PROPER", a.cost))
        if best.root.pos != "N" or not best.sfx:
            continue
        own = counts.get(best.root.form, 0)
        if own * 20 >= c:                                       # root is seen on its own: fine
            continue
        alts = [a for a in cands if a.root.key != best.root.key and a.root.pos != "PROPER"
                and counts.get(a.root.form, 0) >= 100 * max(own, 1)
                and len(a.sfx) <= len(best.sfx)                # no longer, odder chains (çok+GEN+lIk)
                and not {"P2SG", "P2PL"} & set(a.sfx)]
        if alts:
            g = min(alts, key=lambda a: a.cost)
            out[w] = (g.root.key, g.sfx, "rare")
            log["rare"] += 1
            ex["rare"].append((c, f"{w}: {best.root.key}+{'+'.join(best.sfx)} -> {g.root.key}+{'+'.join(g.sfx)}"))

    with open(DATA / "overrides.tsv", "w", encoding="utf-8") as f:
        f.write("# word\troot key\tsuffix chain\tsource\n")
        for w, (k, ch, src) in sorted(out.items()):
            f.write(f"{w}\t{k}\t{' '.join(ch)}\t{src}\n")
    print(f"overrides: {dict(log)} -> data/overrides.tsv")
    for src in ("ud", "rare"):
        print(f"  top {src}: " + " | ".join(e for _, e in sorted(ex[src], reverse=True)[:30]))


if __name__ == "__main__":
    main()
