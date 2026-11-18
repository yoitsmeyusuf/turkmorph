"""Evaluate on UD Turkish test: lemma accuracy per cost model + lossless/fertility of
the full tokenizer on raw test sentences. Writes data/eval.json."""
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_em import cached_analyses                           # noqa: E402
from turkmorph import em                                       # noqa: E402
from turkmorph.corpus import read_conllu, ud_files             # noqa: E402
from turkmorph.morphology import (DEFAULT_LEXICON, HAND_COSTS, LearnedCosts,  # noqa: E402
                                  Lexicon, generate)
from turkmorph.tokenizer import tr_lower                       # noqa: E402

DATA = ROOT / "data"
WORD = re.compile(r"^[^\W\d_]+$")


def test_items():
    c = Counter()
    for f in ud_files(DATA / "raw" / "ud", "test"):
        for s in read_conllu(f):
            for t in s.tokens:
                if WORD.match(t.form):
                    low = tr_lower(t.form)
                    c[(low, tr_lower(t.lemma), t.form != low)] += 1
    return c


# derivational suffixes: UD lemmas often keep them (ilgili, temizlik), sometimes not
# (yaptırmadım -> yap). "stem" accepts the root plus any prefix of derivations.
DERIV = {"LH", "LHK", "CH", "SHZ", "LAS", "SAL", "DAS", "LA", "LAN", "AGT", "VN_HS", "DIM", "CAUS", "PASS", "RECP"}
Q_FORMS = {"mı", "mi", "mu", "mü"}                     # UD lemma of the clitic is "mi"


def _stems(root, chain):
    out = {"mi" if root.pos == "Q" and root.form in Q_FORMS else root.form}
    for i, s in enumerate(chain):
        if s not in DERIV:
            break
        out.add(generate(root.form, root.flags, list(chain[:i + 1])))
    return out


def lemma_scores(items, analyses, lex, costs):
    n = cov = ok = oracle = stem_ok = stem_oracle = 0
    for (w, lemma, cased), c in items.items():
        n += c
        cands = analyses.get(w) or []
        if not cands:
            continue
        cov += c
        roots = [lex.by_key[k] for k, _ in cands]
        root_form = lambda r: "mi" if r.pos == "Q" and r.form in Q_FORMS else r.form  # noqa: E731
        oracle += c * any(root_form(r) == lemma for r in roots)
        stem_oracle += c * any(lemma in _stems(r, ch) for r, (_, ch) in zip(roots, cands))
        best = min(range(len(cands)), key=lambda i: (
            (not cased) and roots[i].pos == "PROPER",
            costs.analysis_cost(roots[i], cands[i][1]), -roots[i].freq, roots[i].key, cands[i][1]))
        ok += c * (root_form(roots[best]) == lemma)
        stem_ok += c * (lemma in _stems(roots[best], cands[best][1]))
    return {"tokens": n, "coverage": cov / n, "root_acc": ok / n, "root_oracle": oracle / n,
            "stem_acc": stem_ok / n, "stem_oracle": stem_oracle / n}


_TOK = None


def _init():
    global _TOK
    from turkmorph.presets import full_tokenizer
    _TOK = full_tokenizer()


def _encode(text):
    t0 = time.perf_counter()
    toks = _TOK.encode(text)
    dt = time.perf_counter() - t0
    return text, toks, _TOK.decode(toks) == text, dict(_TOK.stats), dt


def tokenizer_scores():
    texts = [s.text for f in ud_files(DATA / "raw" / "ud", "test") for s in read_conllu(f) if s.text]
    from turkmorph.em import n_procs
    with Pool(n_procs(), initializer=_init) as pool:
        res = pool.map(_encode, texts, chunksize=32)
    words = sum(len(re.findall(r"\S+", t)) for t, *_ in res)
    ntok = sum(len(toks) for _, toks, *_ in res)
    chars = sum(len(t) for t, *_ in res)
    lossless = sum(ok for _, _, ok, *_ in res)
    ns = Counter()
    for _, toks, *_ in res:
        for t in toks:
            ns[t[:2] if t[1:2] == ":" else "<special>"] += 1
    return {"sentences": len(texts), "lossless": lossless / len(texts),
            "tokens_per_word": ntok / words, "chars_per_token": chars / ntok,
            "token_namespaces": {k: v / ntok for k, v in ns.most_common()}}


def baseline_bpe(vocab_size):
    """Plain BPE on whole training words, same pretokenization, same vocab budget."""
    from turkmorph.bpe import LexBPE
    from turkmorph.tokenizer import CHUNK
    words = []
    for f in ud_files(DATA / "raw" / "ud", "train"):
        for s in read_conllu(f):
            words += [m.group(1) for m in CHUNK.finditer(s.text) if m.group(1)]
    base = LexBPE()
    budget = vocab_size - 256 - len(base.vocab)
    bpe = LexBPE().train(words, num_merges=max(budget, 0), min_count=2)
    texts = [s.text for f in ud_files(DATA / "raw" / "ud", "test") for s in read_conllu(f) if s.text]
    n_tok = n_words = 0
    for t in texts:
        n_words += len(re.findall(r"\S+", t))
        for m in CHUNK.finditer(t):
            word, num, ws, other = m.groups()
            if word:
                n_tok += len(bpe.encode(word))
            elif num:
                n_tok += len(num)
            elif ws:
                n_tok += 0 if ws == " " else len(ws)
            else:
                n_tok += 1
    return {"merges": len(bpe.merges), "tokens_per_word": n_tok / n_words}


def main():
    items = test_items()
    words = {w for w, _, _ in items}
    lex = Lexicon.from_tsv(DATA / "lexicon.tsv")
    full = cached_analyses(words, lex, "test")
    demo = em.collect_analyses(words, DEFAULT_LEXICON)
    out = {"lemma": {}}
    configs = [
        ("A prototype (hand lexicon, hand costs)", demo, DEFAULT_LEXICON, HAND_COSTS),
        ("B full lexicon, hand costs", full, lex, HAND_COSTS),
        ("C full lexicon, EM unsupervised", full, lex,
         LearnedCosts.from_json(DATA / "costs_em.json", len(lex))),
        ("D full lexicon, EM lemma-constrained", full, lex,
         LearnedCosts.from_json(DATA / "costs_em_lemma.json", len(lex))),
    ]
    if (DATA / "costs_bt.json").exists():
        configs.append(("E full lexicon, EM BellaTurca + UD lemma", full, lex,
                        LearnedCosts.from_json(DATA / "costs_bt.json", len(lex))))
    for name, an, lx, costs in configs:
        r = lemma_scores(items, an, lx, costs)
        out["lemma"][name] = r
        print(f"{name:40s} cov {r['coverage']:.3f} | root {r['root_acc']:.3f} "
              f"(oracle {r['root_oracle']:.3f}) | stem {r['stem_acc']:.3f} "
              f"(oracle {r['stem_oracle']:.3f})")
    if "--no-tok" not in sys.argv:
        out["tokenizer"] = tokenizer_scores()
        print(json.dumps(out["tokenizer"], indent=1))
        from turkmorph.presets import full_tokenizer
        v = len(full_tokenizer().vocab)
        out["tokenizer"]["vocab_size"] = v
        out["baseline_bpe"] = baseline_bpe(v)
        print("vocab", v, "plain BPE, same vocab:", out["baseline_bpe"])
    (DATA / "eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
