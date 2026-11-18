"""Train comparison tokenizers and encode the LM corpus with each one.

  python3 lm/prepare.py --chars 150e6 --vocab 64000

Outputs data/lm/<name>/{train,valid}.bin (uint32 ids), meta.json; for the composer
also groups.bin (uint8, 1 = token starts a word group). Every tokenizer is checked
for losslessness on the validation text; BPC is only comparable if that holds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turkmorph.merge import MorphMerge, units                   # noqa: E402
from turkmorph.tokenizer import CHUNK, Vocab                    # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "lm"


def read_chars(path, n_chars):
    lines, n = [], 0
    for line in open(path, encoding="utf-8"):
        lines.append(line)
        n += len(line)
        if n >= n_chars:
            break
    return lines


# ------------------------------------------------------------ our tokenizer --
_TOK = None


def _init_full():
    global _TOK
    from turkmorph.presets import full_tokenizer
    _TOK = full_tokenizer()


def _encode_chunk_types(chunks):
    return [(c, tuple(_TOK._encode_word(c))) for c in chunks]


def chunk_counts(lines):
    types = Counter()
    for line in lines:
        for m in CHUNK.finditer(line):
            if m.group(1):
                types[m.group(1)] += 1
    return types


def prefill_cache(items, processes):
    """Analyse every distinct word chunk once, in parallel -> {chunk: base tokens}."""
    items = list(items)
    batches = [items[i:i + 2000] for i in range(0, len(items), 2000)]
    cache = {}
    with Pool(processes, initializer=_init_full) as pool:
        for res in pool.imap_unordered(_encode_chunk_types, batches):
            cache.update(res)
    return cache


def _encode_lines(lines):
    # encode("\n" + line)[1:] == the tokens this line gets inside the full text
    # (the <glue> a word needs after the previous line's newline is kept)
    out = []
    for line in lines:
        out.extend(_TOK.encode("\n" + line)[1:])
    return np.fromiter((_TOK.vocab.id(t) for t in out), dtype=np.uint32, count=len(out))


def morph_tokenizer(cache, merge=None):
    from turkmorph.presets import full_tokenizer
    tok = full_tokenizer()
    tok.merge = merge
    tok.cache = {k: tuple(merge.apply(list(v))) if merge else v for k, v in cache.items()}
    tok.cache_size = 1 << 62
    tok.vocab = Vocab(tok)
    return tok


def encode_with(tok, lines, processes):
    global _TOK
    _TOK = tok                                           # forked workers inherit the cache
    batches = [lines[i:i + 500] for i in range(0, len(lines), 500)]
    with Pool(processes) as pool:
        return np.concatenate(list(pool.imap(_encode_lines, batches)))


def group_starts(tokens):
    """1 where a token opens a new word group (see TurkMorphTokenizer.encode_grouped)."""
    from turkmorph.tokenizer import group_flags
    return group_flags(tokens)


# --------------------------------------------------------------- baselines --
def train_bpe(train_path, vocab):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    tk.train([str(train_path)], tr)
    return tk


def train_unigram(train_path, vocab, prefix):
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(
        input=str(train_path), model_prefix=str(prefix), vocab_size=vocab, model_type="unigram",
        byte_fallback=True, normalization_rule_name="identity", remove_extra_whitespaces=False,
        add_dummy_prefix=False, split_digits=True, character_coverage=0.9999,
        input_sentence_size=2_000_000, shuffle_input_sentence=True, max_sentence_length=1 << 16,
        train_extremely_large_corpus=False, num_threads=os.cpu_count(), minloglevel=2)
    return spm.SentencePieceProcessor(model_file=str(prefix) + ".model")


def save(name, train_ids, valid_ids, vocab_size, valid_text, extra=None, groups=None):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.asarray(train_ids, dtype=np.uint32).tofile(d / "train.bin")
    np.asarray(valid_ids, dtype=np.uint32).tofile(d / "valid.bin")
    if groups is not None:
        for split, g in groups.items():
            np.asarray(g, dtype=np.uint8).tofile(d / f"groups_{split}.bin")
    meta = {"vocab_size": vocab_size, "train_tokens": len(train_ids), "valid_tokens": len(valid_ids),
            "valid_chars": len(valid_text), **(extra or {})}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {name}: vocab {vocab_size}, train {len(train_ids)/1e6:.1f}M tok, "
          f"valid {len(valid_ids)} tok, chars/token {len(valid_text)/len(valid_ids):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=float, default=150e6)
    ap.add_argument("--valid-chars", type=float, default=2e6)
    ap.add_argument("--vocab", type=int, default=64000)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    procs = int(os.environ.get("TURKMORPH_PROCS") or 6)       # 6 workers: keeps a laptop CPU cooler
    OUT.mkdir(parents=True, exist_ok=True)
    train_lines = read_chars(DATA / "corpus" / "train.txt", args.chars)
    valid_lines = read_chars(DATA / "corpus" / "valid.txt", args.valid_chars)
    valid_text = "".join(valid_lines)
    tok_train = OUT / "tokenizer_train.txt"                 # tokenizers learn from 300M chars
    only = set(filter(None, args.only.split(",")))
    if not tok_train.exists() and (not only or only & {"bpe", "unigram"}):
        tok_train.write_text("".join(read_chars(DATA / "corpus" / "train.txt", 3e8)), encoding="utf-8")
    only = set(filter(None, args.only.split(",")))
    want = lambda n: not only or n in only                   # noqa: E731
    report = {}

    if want("bpe"):
        t0 = time.time()
        bpe = train_bpe(tok_train, args.vocab)
        assert bpe.decode(bpe.encode(valid_text).ids) == valid_text, "BPE not lossless"
        enc = lambda ls: [i for e in bpe.encode_batch(ls) for i in e.ids]   # noqa: E731
        save("bpe", enc(train_lines), enc(valid_lines), bpe.get_vocab_size(), valid_text,
             {"build_s": time.time() - t0})

    if want("unigram"):
        t0 = time.time()
        sp = train_unigram(tok_train, args.vocab, OUT / "unigram")
        ok = sp.decode(sp.encode(valid_text)) == valid_text
        report["unigram_lossless"] = ok
        print("  unigram lossless on valid:", ok)
        enc = lambda ls: [i for ids in sp.encode(ls) for i in ids]          # noqa: E731
        save("unigram", enc(train_lines), enc(valid_lines), sp.get_piece_size(), valid_text,
             {"build_s": time.time() - t0, "lossless": ok})

    morph_names = [n for n in ("morph", "morph_sfx", "morph_full") if want(n)]
    if morph_names or want("composer"):
        t0 = time.time()
        types = chunk_counts(train_lines)                    # merge statistics: train only
        cache = prefill_cache(set(types) | set(chunk_counts(valid_lines)), procs)
        print(f"  analysed {len(cache)} chunk types in {time.time() - t0:.0f}s", flush=True)
        base_vocab = len(morph_tokenizer({}).vocab)
        budget = args.vocab - base_vocab
        for name in ("morph", "morph_sfx", "morph_full"):
            if name not in morph_names and not (name == "morph" and want("composer")):
                continue
            merge = None
            if name != "morph":
                mode = name.split("_")[1]
                uc = Counter()
                for chunk, toks in cache.items():
                    toks = list(toks)
                    for i, j in units(toks):
                        if j - i > 1:
                            uc[tuple(toks[i:j])] += types[chunk]
                merge = MorphMerge.train(uc, max(budget, 0), mode)
                merge.save(DATA / f"merge_{mode}.json")
            tok = morph_tokenizer(cache, merge)
            tr_ids = encode_with(tok, train_lines, procs)
            va_ids = encode_with(tok, valid_lines, procs)
            assert tok.decode_ids(va_ids.tolist()) == valid_text, f"{name} not lossless"
            groups = None
            if name == "morph" and want("composer"):
                groups = {s: group_starts([tok.vocab.tok(i) for i in ids])
                          for s, ids in (("train", tr_ids), ("valid", va_ids))}
            save(name, tr_ids, va_ids, len(tok.vocab), valid_text,
                 {"merges": len(merge.merges) if merge else 0}, groups)
    (OUT / "prepare_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
