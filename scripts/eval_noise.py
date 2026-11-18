"""Typo / ASCII-spelling evaluation on held-out BellaTurca text (data/corpus/valid.txt).

Clean words are corrupted synthetically (Turkish letters dropped, or one random
letter deleted / inserted / substituted / swapped). For each noisy word: is it
mapped back to the clean word's analysis? Genuine unknown words (names, foreign,
rare) measure false alarms. Every encoding is checked to be lossless.
Writes data/eval_noise.json."""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turkmorph.normalize import EDIT_CHARS, apply_edit, asciify   # noqa: E402
from turkmorph.presets import full_tokenizer                     # noqa: E402
from turkmorph.tokenizer import CHUNK                            # noqa: E402

DATA = ROOT / "data"
N_WORDS, N_UNKNOWN = 3000, 2000


def strip_markers(toks):
    return [t for t in toks if t != "<ascii>" and not t.startswith("E:")]


def path_of(toks):
    if "<ascii>" in toks:
        return "ascii"
    if any(t.startswith("E:") for t in toks):
        return "edit"
    if any(t.startswith("R:") for t in toks):
        return "known"
    if "<bare>" in toks:
        return "bare"
    return "lex/guess"


def main():
    rng = random.Random(2026)
    tok = full_tokenizer()
    raw = full_tokenizer()
    raw.normalizer = None                                      # same tokenizer, no normalization
    words = [m.group(1) for line in open(DATA / "corpus" / "valid.txt", encoding="utf-8")
             for m in CHUNK.finditer(line) if m.group(1) and m.group(1).islower()
             and re.fullmatch(r"[a-zçğıöşüâîû]+", m.group(1))]
    rng.shuffle(words)

    res = {k: Counter() for k in ("ascii", "edit")}
    ntok = {k: Counter() for k in ("ascii", "edit")}
    ex = {k: Counter() for k in ("ascii", "edit")}
    done = Counter()
    for w in words:
        if done["ascii"] >= N_WORDS and done["edit"] >= N_WORDS:
            break
        if len(w) < 5:
            continue
        clean = tok.encode(w)
        if path_of(clean) != "known":
            continue
        kinds = ["edit"] + (["ascii"] if asciify(w) != w else [])
        kind = rng.choice(kinds)
        if done[kind] >= N_WORDS:
            continue
        if kind == "ascii":
            noisy = asciify(w)
        else:
            op = rng.choice(["del", "ins", "sub", "swap"])
            p = rng.randrange(len(w) - (op == "swap") + (op == "ins"))
            noisy = apply_edit(w, (op, p, rng.choice(EDIT_CHARS)))
            if noisy == w:
                continue
        enc = tok.encode(noisy)
        assert tok.decode(enc) == noisy, (noisy, enc)
        done[kind] += 1
        pth = path_of(enc)
        if pth == "known":
            r = "real-word error (noisy form is itself a valid word: untouched)"
        elif pth in ("ascii", "edit"):
            r = "recovered" if strip_markers(enc) == clean else "mapped to another word"
        else:
            r = "not recovered (LEX / guess)"
        res[kind][r] += 1
        ntok[kind]["clean"] += len(clean)
        ntok[kind]["noisy, normalizer"] += len(enc)
        ntok[kind]["noisy, no normalizer"] += len(raw.encode(noisy))
        if r != "recovered":
            ex[kind][f"{w} -> {noisy}: {' '.join(enc)}"] += 1

    # false alarms: genuine words without a known analysis
    fa, fa_ex, seen = Counter(), Counter(), 0
    for w in words:
        if seen >= N_UNKNOWN:
            break
        enc = raw.encode(w)
        if path_of(enc) == "known":
            continue
        seen += 1
        e2 = tok.encode(w)
        assert tok.decode(e2) == w
        p = path_of(e2)
        fa[p] += 1
        if p in ("ascii", "edit"):
            fa_ex[f"{w}: {' '.join(e2)}"] += 1

    out = {"synthetic": {}, "false_alarms": {}}
    print("SYNTHETIC NOISE on held-out BellaTurca words (all encodings lossless)")
    for kind in ("ascii", "edit"):
        n = sum(res[kind].values())
        print(f"\n  {kind} ({n} words)")
        for r, c in res[kind].most_common():
            print(f"    {r:66s} {c:5d}  {c / n:6.1%}")
        t = ntok[kind]
        print(f"    tokens/word: clean {t['clean'] / n:.2f}, noisy with normalizer "
              f"{t['noisy, normalizer'] / n:.2f}, noisy without {t['noisy, no normalizer'] / n:.2f}")
        print("    examples of misses: " + " | ".join(e for e, _ in ex[kind].most_common(8)))
        out["synthetic"][kind] = {"n": n, **dict(res[kind]), **{f"tokens_{k}": v / n for k, v in t.items()}}
    n = sum(fa.values())
    print(f"\nFALSE ALARMS: {n} genuine words without a known analysis (names, foreign, rare)")
    for p, c in fa.most_common():
        print(f"    {p:12s} {c:5d}  {c / n:6.1%}")
    print("    normalized anyway: " + " | ".join(e for e, _ in fa_ex.most_common(25)))
    out["false_alarms"] = {"n": n, **dict(fa), "examples": fa_ex.most_common(50)}
    (DATA / "eval_noise.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
