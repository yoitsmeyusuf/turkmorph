"""Language id + attested stems + LEX BPE -> data/langid.json, attested_stems.txt, lex_bpe.json

LEX BPE is trained on what the morphology leaves over: guessed stems of words
with no known root (suffixes stripped), unknown words whole, proper names not in
ROOT, and foreign words. Never on inflected Turkish words."""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_em import bt_words, cached_analyses, ud_items       # noqa: E402
from turkmorph.bpe import LexBPE                               # noqa: E402
from turkmorph.corpus import read_conllu, ud_files             # noqa: E402
from turkmorph.langid import CharNgramLangId                   # noqa: E402
from turkmorph.morphology import Lexicon, analyze_guess        # noqa: E402
from turkmorph.tokenizer import tr_lower                       # noqa: E402
from turkmorph.zemberek import read_dict                       # noqa: E402
from multiprocessing import Pool                               # noqa: E402

_LID = None


def _guess_chains(w):
    if _LID.is_foreign(w):
        return w, []
    return w, [(a.stem, a.sfx) for a in analyze_guess(w, attested={w[:i] for i in range(len(w))})[:3]]

DATA = ROOT / "data"
RAW = DATA / "raw"
WORD = re.compile(r"^[^\W\d_]+$")
NUM_MERGES = 16000


def foreign_types(split):
    c = Counter()
    for f in ud_files(RAW / "ud_foreign", split, include_noncommercial=split == "test"):
        for s in read_conllu(f):
            for t in s.tokens:
                if WORD.match(t.form):
                    c[t.form.lower()] += 1
    return c


def main():
    lex = Lexicon.from_tsv(DATA / "lexicon.tsv")
    train = Counter()
    for (w, _), c in ud_items("train").items():
        train[w] += c
    dev = {w for w, _ in ud_items("dev")}

    # --- language id
    fr_train, fr_test = foreign_types("train"), foreign_types("test")
    lid = CharNgramLangId.train(list(train) + [r.form for r in lex.roots], fr_train)
    thr = lid.calibrate(dev - set(fr_train), max_tr_false_positive=0.01)
    recall = sum(lid.is_foreign(w) for w in fr_test if len(w) >= 3 and w not in train) / \
        max(1, sum(1 for w in fr_test if len(w) >= 3 and w not in train))
    print(f"langid: threshold {thr:.3f}, TR dev false-positive 1%, foreign test recall {recall:.3f}")
    lid.save(DATA / "langid.json")

    # --- attested stems: an unknown stem seen with >= 2 different suffix chains
    bt = bt_words()
    if bt:                                                     # BellaTurca: much larger residue
        for w, c in bt.items():
            train[w] += c
        analyses = cached_analyses(set(bt), lex, "bt")
        analyses.update(cached_analyses(set(train) - set(bt), lex, "train"))
    else:
        analyses = cached_analyses(set(train), lex, "train")
    unknown = [w for w in train if not analyses.get(w)]
    global _LID
    _LID = lid
    chains = defaultdict(set)
    guesses = {}
    import time
    from turkmorph.em import progress
    t0 = time.time()
    from turkmorph.em import n_procs
    with Pool(n_procs()) as pool:                              # forked workers share _LID
        for i, (w, gs) in enumerate(pool.imap_unordered(_guess_chains, unknown, chunksize=64), 1):
            guesses[w] = gs
            for stem, sfx in gs:
                chains[stem].add(sfx)
            if i % 1000 == 0 or i == len(unknown):
                progress(i, len(unknown), t0, "guess unknown stems")
    attested = {s for s, cs in chains.items() if len(cs) >= 2}
    (DATA / "attested_stems.txt").write_text("\n".join(sorted(attested)) + "\n", encoding="utf-8")

    # --- typo normalizer data: corpus word frequencies (+ known flag), names seen capitalized
    from turkmorph.normalize import Normalizer
    Normalizer(dict(train), {w for w in train if analyses.get(w)}).save(DATA / "wordfreq.tsv")
    import pickle
    btp = DATA / "cache" / "bt_counts.pkl"
    if btp.exists():
        names = sorted((w, c) for w, c in pickle.load(open(btp, "rb"))["proper"].items() if c >= 3)
        (DATA / "proper_words.txt").write_text("".join(f"{w}\t{c}\n" for w, c in names), encoding="utf-8")
    print(f"wordfreq: {len(train)} types")
    print(f"unknown types: {len(unknown)}, attested stems: {len(attested)}")

    # --- LEX BPE corpus
    corpus = []
    for w in unknown:
        # best guess under the final attested set: >= 2 suffixes, or an attested stem
        ok = [(st, sfx) for st, sfx in guesses[w] if len(sfx) >= 2 or st in attested]
        stem = ok[0][0] if ok else w
        corpus += [stem] * train[w]
    names = set()
    for f in ("proper", "locations-tr", "person-names"):
        for r, _ in read_dict(RAW / f"zemberek-{f}.dict", proper=True):
            if r.form not in lex.by_form:
                names.add(r.form)
    corpus += sorted(names)
    corpus += [w for w, c in fr_train.items() if c >= 2 for _ in range(min(c, 5))]
    print(f"LEX corpus: {len(corpus)} words ({len(names)} names, {len(fr_train)} foreign types)")
    bpe = LexBPE().train(corpus, num_merges=NUM_MERGES, min_count=2)
    bpe.save(DATA / "lex_bpe.json")
    print(f"LEX BPE: {len(bpe.merges)} merges, {len(bpe.vocab)} pieces")


if __name__ == "__main__":
    main()
