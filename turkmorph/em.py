"""EM over morphological analyses: learn root and suffix-bigram probabilities.

The analyzer proposes every analysis of a word; EM distributes each word's
corpus count over its analyses in proportion to P(analysis), re-estimates P from
the expected counts, and repeats (Unigram-LM style). Optionally a gold lemma
constrains the analyses of a training token (lemma-supervised EM).
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

from .morphology import END, LearnedCosts, Lexicon, analyze_known

_LEX: Lexicon | None = None


def n_procs() -> int:
    """Worker processes: TURKMORPH_PROCS (e.g. 6 to keep a laptop cool), else all cores."""
    return int(os.environ.get("TURKMORPH_PROCS") or os.cpu_count())


def _init(lex):
    global _LEX
    _LEX = lex


def _analyze(word):
    return word, [(a.root.key, a.sfx) for a in analyze_known(word, _LEX)]


def progress(done, total, t0, label, every=15.0, _last=[0.0]):
    """Print 'label: done/total, elapsed, ETA' at most every `every` seconds."""
    now = time.time()
    if done < total and now - _last[0] < every:
        return
    _last[0] = now
    el = now - t0
    eta = el / max(done, 1) * (total - done)
    print(f"  [{label}] {done}/{total} ({done / max(total, 1):.0%})  "
          f"elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min", flush=True, file=sys.stdout)


def collect_analyses(words, lex: Lexicon, processes: int | None = None, chunksize=64):
    """word -> [(root_key, suffix chain), ...] (all analyses, unranked)."""
    words = sorted(set(words))
    out, t0 = {}, time.time()
    with Pool(processes or n_procs(), initializer=_init, initargs=(lex,)) as pool:
        for i, (w, a) in enumerate(pool.imap_unordered(_analyze, words, chunksize=chunksize), 1):
            out[w] = a
            if i % 1000 == 0 or i == len(words):
                progress(i, len(words), t0, f"analyse {len(words)} types")
    return out


def _counts_to_model(root_c, big_c, uni_c, lex):
    return LearnedCosts(root_c, big_c, uni_c, n_roots=len(lex))


def train(items, analyses, lex: Lexicon, iters: int = 10, log=print,
          init: LearnedCosts | None = None) -> LearnedCosts:
    """items: [(word, count, allowed_root_forms or None)].
    analyses: output of collect_analyses.
    init: model for the first E-step instead of uniform posteriors. EM only finds a
    local optimum; starting from a lemma-supervised model keeps it from settling on
    a rare homograph (ise = 'is'+DAT instead of the copula i+se)."""
    model = init
    for it in range(iters + 1):
        root_c = defaultdict(float)
        big_c = defaultdict(lambda: defaultdict(float))
        uni_c = defaultdict(float)
        ll, n = 0.0, 0
        for word, count, allowed in items:
            cands = analyses.get(word) or []
            if allowed:
                filt = [a for a in cands if lex.by_key[a[0]].form in allowed]
                cands = filt or cands
            if not cands:
                continue
            if model is None:                        # iteration 0 without init: uniform
                post = [1.0 / len(cands)] * len(cands)
            else:
                costs = [model.analysis_cost(lex.by_key[k], ch) for k, ch in cands]
                m = min(costs)
                w = [math.exp(m - c) for c in costs]
                z = sum(w)
                post = [x / z for x in w]
                ll += count * (math.log(z) - m)
                n += count
            for (key, chain), p in zip(cands, post):
                q = p * count
                root_c[key] += q
                prev = "^" + lex.by_key[key].pos
                for s in chain:
                    s = s.split("!", 1)[0]
                    big_c[prev][s] += q
                    uni_c[s] += q
                    prev = s
                big_c[prev][END] += q
                uni_c[END] += q
        model = _counts_to_model(root_c, big_c, uni_c, lex)
        if it and log:
            log(f"  EM iter {it}: avg log-lik/token {ll / max(n, 1):.4f}")
    return model
