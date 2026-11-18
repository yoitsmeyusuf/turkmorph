"""Where does the full tokenizer segment wrong? Error analysis on UD Turkish test.

Every alphabetic test token goes through the real tokenizer; its root and suffix
chain are compared with the gold lemma and gold morphological features.
Writes data/errors.json and prints the breakdown with examples."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turkmorph.corpus import read_conllu, ud_files             # noqa: E402
from turkmorph.morphology import analyze_known, generate        # noqa: E402
from turkmorph.presets import full_tokenizer                    # noqa: E402
from turkmorph.tokenizer import tr_lower                        # noqa: E402

DATA = ROOT / "data"
WORD = re.compile(r"^[^\W\d_]+(?:['’][^\W\d_]+)?$")
DERIV = {"LH", "LHK", "CH", "SHZ", "LAS", "SAL", "DAS", "LA", "LAN", "AGT", "VN_HS", "DIM", "CAUS", "PASS", "RECP"}
Q_FORMS = {"mı", "mi", "mu", "mü"}
CASES = {"ACC": "Acc", "DAT": "Dat", "LOC": "Loc", "ABL": "Abl", "GEN": "Gen", "INS": "Ins", "EQU": "Equ"}
POSS = {"P1SG": ("1", "Sing"), "P2SG": ("2", "Sing"), "P3SG": ("3", "Sing"),
        "P1PL": ("1", "Plur"), "P2PL": ("2", "Plur"), "P3PL": ("3", "Plur")}


def feats(s):
    return dict(kv.split("=", 1) for kv in s.split("|") if "=" in kv)


def parse(tokens, lex):
    """-> (path, root, chain): first R:/L: word in the token list."""
    root, chain, lex_word = None, [], False
    for t in tokens:
        if t.startswith("R:") and root is None and not lex_word:
            root = lex.by_key[t[2:]]
        elif t.startswith("L:"):
            lex_word = True
        elif t.startswith("S:"):
            chain.append(t[2:].split("!")[0])               # unions (P3SG|ACC) kept whole
    if root is not None:
        return ("apos" if any(t in ("<apos>", "<apos2>") for t in tokens) else "known"), root, chain
    return ("guess" if chain else "lex"), None, chain


def expansions(chain):
    """All plain chains an underspecified chain stands for."""
    outs = [[]]
    for sid in chain:
        mems = [m.split("+") for m in sid.split("|")]
        outs = [o + m for o in outs for m in mems]
    return outs


def stems(root, chain):
    chain = expansions(chain)[0]
    out = {"mi" if root.pos == "Q" and root.form in Q_FORMS else root.form}
    for i, s in enumerate(chain):
        if s not in DERIV:
            break
        out.add(generate(root.form, root.flags, chain[:i + 1], root.inner))
    return out


def lexicalized(root, chain, lemma):
    """UD keeps a frozen form as lemma (sadece = sade+EQU, gerçekten = gerçek+ABL)."""
    return any(generate(root.form, root.flags, chain[:i], root.inner) == lemma
               for i in range(1, len(chain) + 1))


def nominal_check(chain, f):
    """Compare Case / Number / possessor of our chain with UD features."""
    out = {}
    if "Case" in f:
        ours = "Nom"
        for s in chain:
            if s in CASES:
                ours = CASES[s]
        out["Case"] = (ours, f["Case"])
    if "Number" in f and "Case" in f:
        out["Number"] = ("Plur" if "PL" in chain else "Sing", f["Number"])
    if "Case" in f:
        p = [POSS[s] for s in chain if s in POSS]
        ours = p[-1] if p else None
        gold = (f["Person[psor]"], f.get("Number[psor]", "Sing")) if "Person[psor]" in f else None
        out["Possessor"] = (ours, gold)
    return out


def verbal_check(chain, f):
    out = {}
    out["Polarity"] = ("Neg" if {"NEG", "IMPOT", "CV_MADAN", "CV_MAKSIZIN"} & set(chain) else "Pos",
                       f.get("Polarity", "Pos"))
    if "Voice" in f or {"PASS", "CAUS"} & set(chain):
        v = ("Cau" if "CAUS" in chain else "") + ("Pass" if "PASS" in chain else "")
        out["Voice"] = (v or None, f.get("Voice"))
    if f.get("Aspect") == "Prog" or {"PROG", "PROG2"} & set(chain):
        out["Prog"] = (bool({"PROG", "PROG2"} & set(chain)), f.get("Aspect") == "Prog")
    return out


def convention(name, ours, gold, root, chain, upos):
    """Known UD-vs-us labeling differences that are not segmentation errors."""
    if upos == "PRON" and name in ("Number", "Possessor") and ours in ("Sing", None):
        return True                                  # biz = Plur, kendi = Psor 3: lexical
    if name == "Polarity" and root.form == "değil":
        return True                                  # değil is negative by itself
    if name == "Prog" and gold and ({"CV_ARAK", "KEN"} & set(chain) or chain[-2:] == ["INF", "LOC"]
                                    or chain[-3:-1] == ["INF", "LOC"]):
        return True                                  # UD: -ArAk, -ken, -mAktA are Aspect=Prog
    if name == "Voice" and (gold is None or (ours == "Cau" + (gold or "") and gold != "Cau")):
        return True                                  # lexicalized voice (ilgilen, getir, çıkar-ıl) unmarked in UD
    return False


def checks_for(t, f, chain):
    c = {}
    if t.upos in ("NOUN", "PROPN", "PRON", "ADJ") or "Case" in f:
        c.update(nominal_check(chain, f))
    if t.upos in ("VERB", "AUX"):
        c.update(verbal_check(chain, f))
    return c


def main():
    tok = full_tokenizer()
    lex = tok.lex
    sents = [s for f in ud_files(DATA / "raw" / "ud", "test", include_noncommercial=True) for s in read_conllu(f)]
    toks = [t for s in sents for t in s.tokens if WORD.match(t.form)]
    forced = {}                                       # id(token) -> encoding chosen in context
    if "--context" in sys.argv:                       # (B): analyses picked by reading the sentence
        from turkmorph.context import ContextTokenizer
        ct = ContextTokenizer(tok, DATA / "context.pt")
        wordish = re.compile(r"^[^\W\d_]+(?:['’][^\W\d_]+)?$")
        for s in sents:
            ws = [t for t in s.tokens if wordish.match(t.form)]
            for t, p in zip(ws, ct.choose([t.form for t in ws], with_probs=True)):
                if p is not None:
                    forced[id(t)] = tok._encode_forced(t.form, p[0].root.key, ct.chain_for(*p))
    n = len(toks)
    cls = Counter()
    ex = defaultdict(Counter)
    upos_err = defaultdict(Counter)
    feat_tot, feat_err = Counter(), Counter()
    feat_kind = defaultdict(Counter)                  # feature -> {context, convention, error}
    feat_ex = defaultdict(Counter)
    for t in toks:
        enc = forced.get(id(t)) or tok.encode(t.form)
        path, root, chain = parse(enc, lex)
        lemma = tr_lower(t.lemma)
        f = feats(t.feats)
        seg = f"{root.key if root else ''} {'+'.join(chain)}".strip()
        if root is None:
            low = tr_lower(t.form)
            if tok.foreign and tok.foreign(low):
                c = "no-root: foreign"
            elif t.form.isupper() and len(t.form) > 1:
                c = "no-root: abbreviation"
            elif t.upos == "PROPN" or t.form[:1].isupper():
                c = "no-root: proper noun"
            else:
                c = "no-root: Turkish word (missing root/suffix)"
            cls[c] += 1
            ex[c][f"{t.form} ({' '.join(enc)})"] += 1
            upos_err[t.upos]["no-root"] += 1
            continue
        if lemma in stems(root, chain):
            cls["root OK"] += 1
            exps = expansions(chain)
            chain = exps[0]
            checks = checks_for(t, f, chain)
            bad = False
            alts = None
            for name, (ours, gold) in checks.items():
                feat_tot[name] += 1
                if ours == gold:
                    continue
                feat_err[name] += 1
                if len(exps) > 1 and any(checks_for(t, f, e).get(name, (None, None))[0] == gold for e in exps):
                    kind = "underspecified"                     # the token left it open (P3SG|ACC)
                    feat_kind[name][kind] += 1
                    continue
                if convention(name, ours, gold, root, chain, t.upos):
                    kind = "convention"
                else:
                    if alts is None:
                        alts = [list(a.sfx) for a in analyze_known(tr_lower(t.form), lex)
                                if a.root.key == root.key]
                    ok_alt = any(checks_for(t, f, a).get(name, (None, None))[0] == gold for a in alts)
                    kind = "context" if ok_alt else "error"
                    bad = bad or kind == "error"
                feat_kind[name][kind] += 1
                feat_ex[f"{name} [{kind}]"][f"{t.form}: {seg}  (ours {ours}, gold {gold})"] += 1
            if bad:
                upos_err[t.upos]["suffix error"] += 1
            continue
        alts = analyze_known(tr_lower(t.form), lex)
        if any(lemma in stems(a.root, list(a.sfx)) for a in alts):
            c = "wrong root: gold was a candidate (disambiguation)"
        elif lexicalized(root, chain, lemma):
            c = "wrong root: UD keeps frozen form as lemma (convention)"
        else:
            c = "wrong root: gold not among candidates"
        cls[c] += 1
        ex[c][f"{t.form}: {seg}  (gold lemma {t.lemma})"] += 1
        upos_err[t.upos]["root"] += 1

    print(f"UD test, {n} alphabetic tokens\n")
    print("ROOT LEVEL")
    for c, v in sorted(cls.items(), key=lambda kv: -kv[1]):
        print(f"  {c:58s} {v:6d}  {v / n:6.2%}")
    print("\nSUFFIX LEVEL (tokens with correct root vs UD features)")
    print(f"  {'feature':10s} {'checked':>8s} {'mismatch':>9s} | {'open':>6s} {'context':>8s} {'convention':>10s} {'ERROR':>7s}")
    for name in feat_tot:
        k = feat_kind[name]
        print(f"  {name:10s} {feat_tot[name]:8d} {feat_err[name]:9d} | {k['underspecified']:6d} {k['context']:8d} "
              f"{k['convention']:10d} {k['error']:7d} ({k['error'] / feat_tot[name]:.2%})")
    print("\nERRORS BY PART OF SPEECH (share of that POS's tokens)")
    upos_n = Counter(t.upos for t in toks)
    for u, c in sorted(upos_err.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        print(f"  {u:6s} {tot:5d}/{upos_n[u]:5d} ({tot / upos_n[u]:5.1%})  "
              + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    for title, d in (("ROOT-LEVEL EXAMPLES", ex), ("SUFFIX-LEVEL EXAMPLES", feat_ex)):
        print(f"\n{title}")
        for c, cnt in d.items():
            print(f"  -- {c}")
            for e, k in cnt.most_common(12):
                print(f"     {k:4d}  {e}")
    out = {"tokens": n, "root_level": dict(cls),
           "suffix_level": {k: {"checked": feat_tot[k], "mismatch": feat_err[k], **feat_kind[k]}
                            for k in feat_tot},
           "by_upos": {u: dict(c) for u, c in upos_err.items()},
           "examples": {c: d.most_common(30) for c, d in list(ex.items()) + list(feat_ex.items())}}
    name = "errors_context.json" if "--context" in sys.argv else "errors.json"
    (DATA / name).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
