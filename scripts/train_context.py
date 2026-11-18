"""(B) Train the context disambiguator on UD Turkish -> data/context.pt

Each word's candidates are the analyzer's own analyses; a candidate is correct when
it agrees with the gold lemma and the gold features (case, number, possessor,
polarity; known UD conventions tolerated, see scripts/errors.py). Trained on UD
train, model selected on dev, reported on test, against the deterministic choice
(EM costs + audited overrides). Must be retrained when the vocabulary changes.
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from errors import checks_for, convention, feats, stems       # noqa: E402
from turkmorph.context import ContextModel, EncoderContextModel, Featurizer   # noqa: E402
from turkmorph.corpus import read_conllu, ud_files              # noqa: E402
from turkmorph.presets import full_tokenizer                    # noqa: E402
from turkmorph.tokenizer import tr_lower                        # noqa: E402

DATA = ROOT / "data"
WORDISH = re.compile(r"^[^\W\d_]+(?:['’][^\W\d_]+)?$")         # what encode() calls a word chunk
MAX_WORDS, BATCH, EPOCHS = 60, (32 if "--encoder" in sys.argv else 64), (6 if "--encoder" in sys.argv else 9)
CONFIG = {"d": 256, "hidden": 256}


def gold_set(t, cands):
    """Indices of candidates consistent with the gold lemma (+ features if possible)."""
    lemma, f = tr_lower(t.lemma), feats(t.feats)
    lem = [k for k, a in enumerate(cands) if lemma in stems(a.root, list(a.sfx))]
    full = []
    for k in lem:
        chain = list(cands[k].sfx)
        ok = all(o == g or convention(n, o, g, cands[k].root, chain, t.upos)
                 for n, (o, g) in checks_for(t, f, chain).items())
        if ok:
            full.append(k)
    return full or lem


ENCODER = "--encoder" in sys.argv          # pretrained TokenEncoder (data/encoder.pt) instead of BiLSTM


def sentences(split, feat):
    """-> list of (forms, gold sets) with sentences split at MAX_WORDS words. The
    encoder sees punctuation and numbers too (as at inference); the BiLSTM words only."""
    out = []
    for f in ud_files(DATA / "raw" / "ud", split):
        for s in read_conllu(f):
            toks = s.tokens if ENCODER else [t for t in s.tokens if WORDISH.match(t.form)]
            for i in range(0, len(toks), MAX_WORDS):
                part = toks[i:i + MAX_WORDS]
                forms = [t.form for t in part]
                golds = [gold_set(t, feat.candidates(t.form)[0]) for t in part]
                out.append((forms, golds))
    return out


def batches(data, feat, shuffle=False, rng=None):
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    groups = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    if shuffle:
        rng.shuffle(groups)
    for g in groups:
        sents = [data[i][0] for i in g]
        x, cands = feat.tensors(sents)
        gold = torch.zeros(x[-1].shape, dtype=torch.bool)
        for b, i in enumerate(g):
            for w, gs in enumerate(data[i][1]):
                if len(cands[b][w]) > 1:
                    for k in gs:
                        gold[b, w, k] = True
        yield x, gold, cands


def evaluate(model, data, feat, dev):
    model.eval()
    n = ok_model = ok_base = 0
    with torch.no_grad():
        for x, gold, _ in batches(data, feat):
            s = model(*[t.to(dev) for t in x]).cpu()
            has = gold.any(-1)
            n += int(has.sum())
            ok_model += int(gold.gather(-1, s.argmax(-1, keepdim=True)).squeeze(-1)[has].sum())
            ok_base += int(gold[..., 0][has].sum())                 # deterministic choice = cand 0
    model.train()
    return n, ok_base / max(n, 1), ok_model / max(n, 1)


def main():
    t0 = time.time()
    rng = random.Random(0)
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = full_tokenizer()
    feat = Featurizer(tok)
    data = {sp: sentences(sp, feat) for sp in ("train", "dev", "test")}
    for sp, d in data.items():
        print(f"  {sp}: {len(d)} sentences  ({time.time() - t0:.0f}s)", flush=True)
    if ENCODER:
        ck = torch.load(DATA / "encoder.pt", weights_only=False)
        assert ck["vocab"] == len(tok.vocab), "encoder vocab differs: re-run lm/prepare.py + pretraining"
        config = {**CONFIG, "enc": ck["config"]}
        model = EncoderContextModel(len(tok.vocab), **config).to(dev)
        model.encoder.load_state_dict(ck["state"])
        enc_ids = {id(p) for p in model.encoder.parameters()}
        opt = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) in enc_ids], "lr": 1e-4},
            {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": 2e-3}],
            weight_decay=0.01)
    else:
        config = CONFIG
        model = ContextModel(len(tok.vocab), **CONFIG).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    best, best_state = -1.0, None
    for ep in range(EPOCHS):
        tot, nb = 0.0, 0
        for x, gold, _ in batches(data["train"], feat, shuffle=True, rng=rng):
            x = [t.to(dev) for t in x]
            gold = gold.to(dev)
            s = model(*x)
            has = gold.any(-1)
            if not has.any():
                continue
            s, g = s[has], gold[has]
            loss = (torch.logsumexp(s, -1) - torch.logsumexp(s.masked_fill(~g, float("-inf")), -1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        n, base, acc = evaluate(model, data["dev"], feat, dev)
        print(f"  epoch {ep + 1}/{EPOCHS} loss {tot / max(nb, 1):.3f}  dev ambiguous words {n}: "
              f"deterministic {base:.2%} -> context {acc:.2%}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    n, base, acc = evaluate(model, data["test"], feat, dev)
    print(f"TEST ambiguous words {n}: deterministic {base:.2%} -> context {acc:.2%}  "
          f"(errors {1 - base:.2%} -> {1 - acc:.2%}, {(1 - acc) / max(1 - base, 1e-9):.0%} remain)")
    torch.save({"vocab": len(tok.vocab), "config": config, "state": best_state,
                "kind": "encoder" if ENCODER else "lstm"}, DATA / "context.pt")
    (DATA / "context_eval.json").write_text(json.dumps(
        {"test_ambiguous_words": n, "deterministic": base, "context": acc, "dev_best": best}, indent=1))
    print(f"saved data/context.pt  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
