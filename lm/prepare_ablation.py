"""Identical text segments for the ablation ladder: bpe, bpe_norm, morph_sfc, morph.

  python lm/prepare_ablation.py --output-dir data/lm/ablation-30m [--train-chars 30000000]

Text and the plain BPE come from a prepared BPE dir (default data/lm/bpe-v2: 300M chars of
BellaTurca). bpe_norm is trained on the same 300M characters, with the same vocabulary size.
morph_sfc's suffix variants are collected on the LM training text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAYON_NUM_THREADS", "8")

from lm.ablation_tokenizers import NormBPE, SurfaceMorph, train_norm_bpe  # noqa: E402
from lm.experiment import sha256  # noqa: E402
from lm.prepare_equal_text import write_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bpe-dir", type=Path, default=ROOT / "data" / "lm" / "bpe-v2")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data" / "lm" / "ablation-30m")
    ap.add_argument("--train-chars", type=int, default=30_000_000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--chunk-chars", type=int, default=1024)
    args = ap.parse_args()
    if args.output_dir.exists():
        ap.error("Output directory already exists; refusing to overwrite it")
    from tokenizers import Tokenizer
    from turkmorph.fast import fast_tokenizer
    started = time.monotonic()
    out = args.output_dir.with_name(args.output_dir.name + f".preparing-{os.getpid()}")
    out.mkdir(parents=True)
    bpe_meta = json.loads((args.bpe_dir / "meta.json").read_text())
    vocab = bpe_meta["vocab_size"]

    norm_path = args.bpe_dir.parent / (args.bpe_dir.name + "-norm.json")
    if not norm_path.exists():
        print(f"training bpe_norm on {args.bpe_dir / 'reference_train.txt'} ...", flush=True)
        train_norm_bpe(args.bpe_dir / "reference_train.txt", vocab, norm_path)
    norm = NormBPE(norm_path)
    if norm.vocab_size != vocab:
        raise ValueError(f"bpe_norm vocabulary {norm.vocab_size} != {vocab}")
    print(f"bpe_norm ready ({time.monotonic() - started:.0f}s)", flush=True)

    with (args.bpe_dir / "reference_train.txt").open(encoding="utf-8") as f:
        train_text = f.read(args.train_chars)
    valid_text = (args.bpe_dir / "reference_valid.txt").read_text(encoding="utf-8")
    sfc, counts = SurfaceMorph.fit(train_text.splitlines(keepends=True))
    print(f"morph_sfc: {len(sfc.variants)} suffix variants from {sum(counts.values()):,} suffix tokens "
          f"({time.monotonic() - started:.0f}s)", flush=True)
    (out / "morph_sfc_variants.json").write_text(json.dumps(sfc.variants, ensure_ascii=False), encoding="utf-8")

    morph = fast_tokenizer()
    bpe = Tokenizer.from_file(str(args.bpe_dir / "tokenizer.json"))
    encode = {"bpe": lambda t: bpe.encode(t).ids, "bpe_norm": norm.encode_ids,
              "morph_sfc": sfc.encode_ids, "morph": morph.encode_ids}
    decode = {"bpe": bpe.decode, "bpe_norm": norm.decode_ids, "morph_sfc": sfc.decode_ids, "morph": morph.decode_ids}
    sizes = {"bpe": bpe.get_vocab_size(), "bpe_norm": norm.vocab_size, "morph_sfc": sfc.vocab_size,
             "morph": len(morph.vocab)}
    for name in encode:
        (out / name).mkdir()
    splits = {}
    for split, text in (("train", train_text), ("valid", valid_text)):
        splits[split] = write_split(text, split, out, encode, decode, args.max_tokens, args.chunk_chars)
        print(split, json.dumps(splits[split]), f"({time.monotonic() - started:.0f}s)", flush=True)
    meta = {"protocol": "ablation ladder; paired text segments, one shuffled pass, identical order per seed",
            "base_vocab_size": sizes["morph"], "vocab_size": sizes["morph"] + 1, "bos_id": sizes["morph"],
            "tokenizers": {n: {"base_vocab_size": v, "vocab_size": v + 1, "bos_id": v} for n, v in sizes.items()},
            "max_tokens": args.max_tokens, "chunk_chars": args.chunk_chars, "splits": splits,
            "source": str(args.bpe_dir), "bpe_norm_sha256": sha256(norm_path),
            "tokenizer_sha256": {"bpe": sha256(args.bpe_dir / "tokenizer.json"), "bpe_norm": sha256(norm_path),
                                 "morph_vocab": hashlib.sha256(json.dumps(morph.vocab.toks, ensure_ascii=False)
                                                               .encode()).hexdigest()},
            "prepare_code_sha256": sha256(Path(__file__)), "build_s": time.monotonic() - started}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out.rename(args.output_dir)
    print("Prepared", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
