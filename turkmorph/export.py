"""Export the morphology tables for the Rust runtime -> data/morph_tables.json

The Rust crate (rust/) implements the generator and analyzer *rules* (aorist,
causative, narrowing, softening, ...); every *table* it needs comes from here, so
the two implementations cannot drift apart. Re-run after changing morphology.py:
    python3 -m turkmorph.export
"""
from __future__ import annotations

import json
from pathlib import Path

from . import morphology as M
from .phonology import LETTER_NAMES

OUT = Path(__file__).resolve().parent.parent / "data" / "morph_tables.json"


def tables() -> dict:
    return {
        "suffixes": [{"id": s.id, "tmpl": s.tmpl, "nxt": s.nxt, "n_tmpl": s.n_tmpl,
                      "soften": s.soften, "cost": s.cost} for s in M.SUFFIX_LIST],
        "states": M.STATES,                                   # ordered: DFS order
        "final": sorted(M.FINAL),
        "pos_start": M.POS_START,
        "pron_n_states": sorted(M.PRON_N_STATES),
        "root_gated": [[st, sid, flag] for (st, sid), flag in M.ROOT_GATED.items()],
        "suppletion": [[stem, sid, surf] for (stem, sid), surf in M.SUPPLETION.items()],
        "markable": list(M._MARKABLE),
        "marked": {sid: M.MARKED[sid] for sid in M._MARKABLE},
        "first_chars": {sid: "".join(sorted(cs)) for sid, cs in M.FIRST_CHARS.items()},
        "no_filter": sorted(M._NO_FILTER),
        "cp3_swap": sorted(M.CP3_SWAP),
        "de_keep": sorted(M._DE_KEEP),
        "predicative_start": sorted(M._PREDICATIVE_START),
        "marked_cost": M.MARKED_COST,
        "union_margin": M.UNION_MARGIN,
        "predicative_penalty": M.LearnedCosts.PREDICATIVE_PENALTY,
        "letter_names": LETTER_NAMES,
    }


def _ranges(pred) -> list[list[int]]:
    """Code point ranges [lo, hi] where pred(chr(cp)) holds (surrogates skipped)."""
    out, start = [], None
    for cp in range(0x110000):
        ok = not (0xD800 <= cp <= 0xDFFF) and pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append([start, cp - 1])
            start = None
    if start is not None:
        out.append([start, 0x10FFFF])
    return out


def tokenizer_tables() -> dict:
    """What the Rust tokenizer must agree on with Python, Unicode included (Python's
    regex classes and str.lower/upper, not Rust's own, which follow another version)."""
    import re

    from . import normalize as N
    from . import tokenizer as T
    letter = re.compile(r"[^\W\d_]")
    space = re.compile(r"\s")
    chars = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    return {
        "letter_ranges": _ranges(lambda c: bool(letter.match(c))),
        "space_ranges": _ranges(lambda c: bool(space.match(c))),
        # final-sigma context, derived from CPython's own str.lower: x is cased if "AΣx"
        # keeps σ; case-ignorable if only "AΣxA" does (it is skipped over)
        "cased_ranges": _ranges(lambda c: ("AΣ" + c).lower()[1] == "σ"),
        "ignorable_ranges": _ranges(lambda c: ("AΣ" + c).lower()[1] != "σ" and ("AΣ" + c + "A").lower()[1] == "σ"),
        "lower": {c: c.lower() for c in chars if c.lower() != c},
        "upper": {c: c.upper() for c in chars if c.upper() != c},
        "specials": T.SPECIALS,
        "word_prefix": sorted(T.WORD_PREFIX),
        "punct": sorted(T.PUNCT),
        "glued_clitics": list(T.GLUED_CLITICS),
        "glued_conj": sorted(T.GLUED_CONJ),
        "bare_first": sorted(T.BARE_FIRST),
        "edit_chars": N.EDIT_CHARS,
        "max_pos": N.MAX_POS,
        "normalizer": {"min_len": N.Normalizer.MIN_LEN, "ratio": N.Normalizer.RATIO,
                       "short_ratio": N.Normalizer.SHORT_RATIO, "ascii_ratio": N.Normalizer.ASCII_RATIO,
                       "dominant": N.Normalizer.DOMINANT, "ed2_min_len": N.Normalizer.ED2_MIN_LEN,
                       "ed2_ratio": N.Normalizer.ED2_RATIO, "ed2_index_min_count": N.Normalizer.ED2_INDEX_MIN_COUNT,
                       "foreign_override": N.Normalizer.FOREIGN_OVERRIDE},
    }


def main():
    OUT.write_text(json.dumps(tables(), ensure_ascii=False, indent=1), encoding="utf-8")
    tok_out = OUT.with_name("tokenizer_tables.json")
    tok_out.write_text(json.dumps(tokenizer_tables(), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}, {tok_out}")


if __name__ == "__main__":
    main()
