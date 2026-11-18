"""CoNLL-U reader for UD Turkish: surface tokens with gold lemma / UPOS."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Token:
    form: str
    lemma: str
    upos: str
    feats: str


@dataclass
class Sentence:
    text: str
    tokens: list[Token]


def read_conllu(path) -> list[Sentence]:
    """Multiword tokens (Turkish copula clitics: 'gelecekmiş' = 2 syntactic words)
    are collapsed to one surface token carrying the first word's lemma/UPOS."""
    out, text, toks, mwt_end, mwt_form = [], "", [], -1, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = unicodedata.normalize("NFC", line)
        if line.startswith("# text ="):
            text = line.split("=", 1)[1].strip()
        elif not line.strip():
            if toks:
                out.append(Sentence(text, toks))
            text, toks, mwt_end = "", [], -1
        elif not line.startswith("#"):
            cols = line.split("\t")
            idx = cols[0]
            if "-" in idx:
                a, b = idx.split("-")
                mwt_end, mwt_form = int(b), cols[1]
                toks.append(None)                    # filled by first sub-word
                continue
            if "." in idx:
                continue
            i = int(idx)
            if toks and toks[-1] is None:
                toks[-1] = Token(mwt_form, cols[2], cols[3], cols[5])
            elif i <= mwt_end:
                continue
            else:
                toks.append(Token(cols[1], cols[2], cols[3], cols[5]))
    if toks:
        out.append(Sentence(text, toks))
    return out


# Treebanks under non-commercial licenses (CC BY-NC-SA). Nothing that ships in data/
# is learned from them; evaluation scripts may still score on their test splits.
NONCOMMERCIAL_TREEBANKS = frozenset({"tr_imst", "it_isdt"})


def ud_files(root, split: str, include_noncommercial: bool = False):
    files = sorted(Path(root).glob(f"*-ud-{split}.conllu"))
    if include_noncommercial:
        return files
    return [f for f in files if f.name.split("-ud-")[0] not in NONCOMMERCIAL_TREEBANKS]
