"""Python side of the Rust conformance test: same output format as rust/src/bin/analyze.rs.
    python3 scripts/conformance_analyze.py < words.txt > py.txt"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from turkmorph.morphology import LearnedCosts, Lexicon, analyze_known   # noqa: E402

DATA = ROOT / "data"
LEX = COSTS = None


def _init():
    global LEX, COSTS
    LEX = Lexicon.from_tsv(DATA / "lexicon.tsv")
    COSTS = LearnedCosts.from_json(DATA / "costs.json", len(LEX))


def line(w):
    an = analyze_known(w, LEX, COSTS)
    return w + "\t" + " | ".join(f"{a.root.key} {'+'.join(a.sfx)} {a.cost:.12f}" for a in an)


if __name__ == "__main__":
    words = [w.rstrip("\n") for w in sys.stdin]
    with Pool(int(os.environ.get("TURKMORPH_PROCS") or 6), initializer=_init) as p:
        for out in p.imap(line, words, chunksize=200):
            print(out)
