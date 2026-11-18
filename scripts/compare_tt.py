"""TurkMorph vs TurkishTokenizer (alibayram, PyPI turkish-tokenizer 1.0.4).

Run with a Python that has turkish_tokenizer installed (turkmorph itself is pure
stdlib and is imported from this repo):
    <venv>/bin/python scripts/compare_tt.py
Writes data/compare_tt.json.
"""
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import turkish_tokenizer as tt                                    # noqa: E402

from turkmorph.corpus import read_conllu, ud_files               # noqa: E402
from turkmorph.normalize import EDIT_CHARS, apply_edit, asciify  # noqa: E402
from turkmorph.presets import full_tokenizer                     # noqa: E402
from turkmorph.tokenizer import CHUNK, tr_lower                  # noqa: E402

DATA = ROOT / "data"
WORD = re.compile(r"^[^\W\d_]+$")


class TT:
    name = "TurkishTokenizer 1.0.4"

    def __init__(self):
        self.t = tt.TurkishTokenizer()

    def tokens(self, text):
        return [d["token"] for d in self.t.tokenize_text(text)]

    def ids(self, text):
        return self.t.encode(text)

    def decode(self, ids):
        return self.t.decode(ids)

    def root(self, word):
        """First ROOT token, lowercased (its lemma guess)."""
        for d in self.t.tokenize_text(word):
            if d["type"] == "ROOT" and d["token"] not in ("<uppercase>", "<space>", " "):
                return d["token"].strip().lower()
        return None


class TM:
    name = "TurkMorph"

    def __init__(self):
        self.t = full_tokenizer()

    def tokens(self, text):
        return self.t.encode(text)

    def ids(self, text):
        return self.t.encode_ids(text)

    def decode(self, ids):
        return self.t.decode_ids(ids)

    def root(self, word):
        for x in self.t.encode(word):
            if x.startswith("R:"):
                return self.t.lex.by_key[x[2:]].form
        return None


def lossless(tok, texts):
    exact = loose = 0
    for t in texts:
        d = tok.decode(tok.ids(t))
        exact += d == t
        loose += d.strip() == t.strip()
    return exact / len(texts), loose / len(texts)


def fertility(tok, texts):
    n_tok = sum(len(tok.ids(t)) for t in texts)
    n_words = sum(len(t.split()) for t in texts)
    n_chars = sum(len(t) for t in texts)
    return n_tok / n_words, n_chars / n_tok


def speed(tok, texts):
    t0 = time.perf_counter()
    for t in texts:
        tok.ids(t)
    dt = time.perf_counter() - t0
    return sum(len(t.split()) for t in texts) / dt


def word_lossless(tok, w):
    return tok.decode(tok.ids(w)).strip() == w


def main():
    rng = random.Random(7)
    tms, tts = TM(), TT()
    ud = [s for f in ud_files(DATA / "raw" / "ud", "test") for s in read_conllu(f)]
    ud_texts = [s.text for s in ud if s.text]
    bt_texts = [l.rstrip("\n") for l in open(DATA / "corpus" / "valid.txt", encoding="utf-8")]
    rng.shuffle(bt_texts)
    bt_texts = [t for t in bt_texts if t.strip()][:1500]
    out = {}

    print("== 1. Losslessness: decode(encode(x)) == x")
    for name, texts in (("UD test sentences", ud_texts), ("BellaTurca held-out docs", bt_texts)):
        for tok in (tms, tts):
            ex, lo = lossless(tok, texts)
            out.setdefault("lossless", {}).setdefault(name, {})[tok.name] = {"exact": ex, "ignoring_outer_space": lo}
            print(f"  {name:26s} {tok.name:24s} exact {ex:7.2%}   ignoring leading/trailing space {lo:7.2%}")

    print("\n== 2. Tokens per word / chars per token / speed (UD test)")
    for tok in (tms, tts):
        f, c = fertility(tok, ud_texts)
        tok_ = tms.t if tok is tms else None
        if tok_:
            tok_.cache.clear()
        cold = speed(tok, ud_texts[:1500])
        warm = speed(tok, ud_texts[:1500])
        out.setdefault("efficiency", {})[tok.name] = {"tokens_per_word": f, "chars_per_token": c,
                                                      "words_per_s_cold": cold, "words_per_s_warm": warm}
        print(f"  {tok.name:24s} tokens/word {f:.2f}  chars/token {c:.2f}  "
              f"speed {cold:,.0f} words/s (first pass), {warm:,.0f} words/s (cached)")

    print("\n== 3. Root = UD gold lemma (alphabetic UD test tokens)")
    toks = [t for s in ud for t in s.tokens if WORD.match(t.form)]
    for tok in (tms, tts):
        ok = sum(tok.root(t.form) == tr_lower(t.lemma) for t in toks)
        wl = sum(word_lossless(tok, t.form) for t in toks)
        out.setdefault("ud_words", {})[tok.name] = {"root_acc": ok / len(toks), "word_lossless": wl / len(toks)}
        print(f"  {tok.name:24s} root = lemma {ok / len(toks):7.2%}   word round-trips {wl / len(toks):7.2%}")

    print("\n== 4. Phenomenon set: does the word survive encode/decode? (per category)")
    rows = [l.rstrip("\n").split("\t") for l in open(ROOT / "tests" / "data" / "phenomena.tsv", encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    cats = {}
    for cat, w, _ in rows:
        c = cats.setdefault(cat, [0, 0, 0])
        c[0] += 1
        c[1] += word_lossless(tms, w)
        c[2] += word_lossless(tts, w)
    print(f"  {'category':20s} {'n':>3s} {'TurkMorph':>10s} {'TurkishTok':>11s}")
    for cat, (n, a, b) in cats.items():
        print(f"  {cat:20s} {n:3d} {a / n:10.0%} {b / n:11.0%}")
    out["phenomena"] = {c: {"n": n, "TurkMorph": a / n, "TurkishTokenizer": b / n} for c, (n, a, b) in cats.items()}
    fails = [w for _, w, _ in rows if not word_lossless(tts, w)][:20]
    print("  TurkishTokenizer changes e.g.: " + ", ".join(f"{w}→{tts.decode(tts.ids(w)).strip()}" for w in fails[:14]))

    print("\n== 5. Typos: is the clean word's root still found? (held-out words)")
    words = [m.group(1) for t in bt_texts for m in CHUNK.finditer(t)
             if m.group(1) and re.fullmatch(r"[a-zçğıöşü]{5,}", m.group(1))]
    rng.shuffle(words)
    res = {k: Counter() for k in ("ascii", "edit")}
    for w in words:
        if res["ascii"]["n"] >= 1000 and res["edit"]["n"] >= 1000:
            break
        kind = "ascii" if asciify(w) != w and res["ascii"]["n"] < 1000 else "edit"
        if kind == "edit":
            if res["edit"]["n"] >= 1000:
                continue
            op = rng.choice(["del", "ins", "sub", "swap"])
            p = rng.randrange(len(w) - (op == "swap") + (op == "ins"))
            noisy = apply_edit(w, (op, p, rng.choice(EDIT_CHARS)))
        else:
            noisy = asciify(w)
        if noisy == w:
            continue
        r = res[kind]
        r["n"] += 1
        for tok in (tms, tts):
            clean_root = tok.root(w)
            r[f"{tok.name} root kept"] += clean_root is not None and tok.root(noisy) == clean_root
            r[f"{tok.name} lossless"] += word_lossless(tok, noisy)
            r[f"{tok.name} tokens"] += len(tok.ids(noisy))
    for kind, r in res.items():
        n = r["n"]
        print(f"  {kind:6s} ({n} words)")
        for tok in (tms, tts):
            print(f"    {tok.name:24s} same root as clean word {r[tok.name + ' root kept'] / n:7.1%}  "
                  f"lossless {r[tok.name + ' lossless'] / n:7.1%}  tokens/word {r[tok.name + ' tokens'] / n:.2f}")
        out.setdefault("typos", {})[kind] = {k: v / n if k != "n" else v for k, v in r.items()}

    print("\n== 6. Examples")
    for s in ["Evlerimizdekiler saatlerde geldi.", "Kitabımızı Danimarka'dan 2023'te getirdiler.",
              "zeytinyağlarından", "cocuklarimiz geliyom", "Kemal'e NATO'ya"]:
        for tok in (tms, tts):
            print(f"  {tok.name:24s} {s!r:48s} -> {tok.decode(tok.ids(s))!r}")
            print(f"  {'':24s} {' '.join(tok.tokens(s))}")
    (DATA / "compare_tt.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
