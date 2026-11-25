"""Tokenizers for the ablation ladder between byte-level BPE and turkmorph.

  bpe        byte-level BPE (lm/prepare.py)
  bpe_norm   the same BPE trained after turkmorph's cheap normalisations: casing marks
             (<cap>/<caps> + Turkish lowercase), one token per digit, apostrophes isolated
  morph_sfc  turkmorph analyses, but every suffix id is split by its surface allomorph
             (S:PL@ler vs S:PL@lar), i.e. morpheme boundaries without archiphonemes
  morph      turkmorph

bpe_norm vs bpe measures the normalisations; morph_sfc vs bpe_norm the morpheme split;
morph vs morph_sfc the archiphonemic suffix ids. All are lossless (checked on use).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turkmorph.tokenizer import tr_cap, tr_lower, tr_upper  # noqa: E402

CAP, CAPS = "", ""                     # private-use marks; never in real text here
_WORD = re.compile(r"([ \t]?)([^\W\d_]+)")
_MARKED = re.compile(f"([{CAP}{CAPS}])([ \\t]?)([^\\W\\d_]+)")


def normalise(text: str) -> str:
    """Lowercase words that are Capitalised or ALL CAPS and put a mark before the word's
    leading space, so ' Istanbul' -> CAP + ' istanbul' keeps BPE's space+word pieces.
    Mixed-case words (iPhone) stay as they are, which keeps the mapping invertible."""
    if CAP in text or CAPS in text:
        raise ValueError("text contains the private-use case marks")

    def sub(m):
        ws, w = m.groups()
        low = tr_lower(w)
        if w == low:
            return m.group(0)
        if w == tr_cap(low):
            return CAP + ws + low
        if len(w) > 1 and w == tr_upper(low):
            return CAPS + ws + low
        return m.group(0)
    return _WORD.sub(sub, text)


def denormalise(text: str) -> str:
    return _MARKED.sub(lambda m: m.group(2) + (tr_cap(m.group(3)) if m.group(1) == CAP else tr_upper(m.group(3))), text)


def train_norm_bpe(train_path: Path, vocab: int, out_path: Path):
    """BPE on normalised text; digits split, apostrophes isolated, marks as special tokens."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    norm = out_path.with_suffix(".normalised.txt")
    with open(train_path, encoding="utf-8") as src, open(norm, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(normalise(line))
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex("['’]"), behavior="isolated"),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([CAP, CAPS])
    tr = trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             special_tokens=[CAP, CAPS], show_progress=False)
    tk.train([str(norm)], tr)
    norm.unlink()
    tk.save(str(out_path))
    return tk


class NormBPE:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(path))
        self.vocab_size = self.tk.get_vocab_size()

    def encode_ids(self, text):
        return self.tk.encode(normalise(text)).ids

    def decode_ids(self, ids):
        return denormalise(self.tk.decode(list(ids), skip_special_tokens=False))


def _word_open(tok, prev):
    """turkmorph.tokenizer.group_flags rule for one token."""
    from turkmorph.tokenizer import MERGE_PREFIX, WORD_PREFIX
    cont = (tok.startswith(("S:", "B:", "E:", MERGE_PREFIX)) or (tok.startswith("L:") and not tok.startswith("L:▁"))
            or tok in ("<apos>", "<apos2>", "<lit>") or (tok.startswith("D:") and prev.startswith("D:")))
    return not (cont or prev in WORD_PREFIX)


class SurfaceMorph:
    """turkmorph with suffix ids split by surface form. The label of suffix k in a word is
    the text it adds: the last (len(decode(word[:k])) - len(decode(word[:k-1]))) chars,
    or '∅' when it adds none (vowel-drop stems etc.). Decoding drops the label, so it is
    lossless by construction; unseen variants fall back to the plain suffix id."""

    def __init__(self, variants=None):
        from turkmorph.fast import fast_tokenizer
        self.base = fast_tokenizer()
        self.toks = list(self.base.vocab.toks)
        self.n_base = len(self.toks)
        self.variants = list(variants or [])
        self.index = {t: i for i, t in enumerate(self.toks)}
        for v in self.variants:
            self.index[v] = len(self.toks)
            self.toks.append(v)
        self.vocab_size = len(self.toks)

    def labelled(self, text):
        toks = self.base.encode(text)
        out, word, prev = [], [], ""
        for t in toks:
            if _word_open(t, prev):
                word = []
            word.append(t)
            prev = t
            if t.startswith("S:"):
                before = len(self.base.decode(word[:-1]))
                full = self.base.decode(word)
                added = len(full) - before
                out.append(f"{t}@{full[-added:] if added > 0 else '∅'}")
            else:
                out.append(t)
        return out

    def encode_ids(self, text):
        return [self.index.get(t, self.index.get(t.split("@", 1)[0])) for t in self.labelled(text)]

    def decode_ids(self, ids):
        return self.base.decode_ids([i if i < self.n_base else self.index[self.toks[i].split("@", 1)[0]] for i in ids])

    @classmethod
    def fit(cls, lines, min_count=1):
        """Collect suffix variants from training text."""
        probe = cls()
        c = Counter(t for line in lines for t in probe.labelled(line) if t.startswith("S:") and "@" in t)
        return cls(sorted(v for v, n in c.items() if n >= min_count)), c


if __name__ == "__main__":
    sm = SurfaceMorph()
    for w in ["evlerden", "kitaplardan", "kitabı", "burnu", "gidemeyeceklermiş", "İstanbul'dakiler"]:
        print(w, sm.labelled(w))
    for s in ["İstanbul'da 2023 yılında NATO zirvesi.", "iPhone ve COVID-19, Ankara'ya"]:
        n = normalise(s)
        print(json.dumps(n, ensure_ascii=False), denormalise(n) == s)
