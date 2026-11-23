"""Build a BPE baseline whose vocabulary matches the current morph tokenizer, on a larger reference text.

  python3 lm/prepare_bpe_v2.py --train-chars 300e6 --output-dir data/lm/bpe-v2

reference_train.txt = whole lines from the start of data/corpus/train.txt;
reference_valid.txt = the validation reference used by every earlier experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAYON_NUM_THREADS", "8")

from lm.experiment import sha256
from lm.prepare import read_chars, train_bpe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-chars", type=float, default=300e6)
    ap.add_argument("--train-source", type=Path, default=ROOT / "data" / "corpus" / "train.txt")
    ap.add_argument("--valid-reference", type=Path, default=ROOT / "data" / "lm" / "bpe" / "reference_valid.txt")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data" / "lm" / "bpe-v2")
    args = ap.parse_args()
    if args.output_dir.exists():
        ap.error("Output directory already exists; refusing to overwrite it")
    from turkmorph.fast import fast_tokenizer
    vocab = len(fast_tokenizer().vocab)
    staging = args.output_dir.with_name(args.output_dir.name + f".preparing-{os.getpid()}")
    staging.mkdir(parents=True)
    started = time.monotonic()
    train_text = "".join(read_chars(args.train_source, args.train_chars))
    (staging / "reference_train.txt").write_text(train_text, encoding="utf-8")
    shutil.copyfile(args.valid_reference, staging / "reference_valid.txt")
    valid_text = (staging / "reference_valid.txt").read_text(encoding="utf-8")
    print(f"reference: train {len(train_text):,} chars, valid {len(valid_text):,} chars", flush=True)
    bpe = train_bpe(staging / "reference_train.txt", vocab)
    if bpe.get_vocab_size() != vocab:
        raise ValueError(f"BPE vocabulary {bpe.get_vocab_size()} != morph vocabulary {vocab}")
    if bpe.decode(bpe.encode(valid_text).ids) != valid_text:
        raise ValueError("BPE is not lossless on the validation text")
    bpe.save(str(staging / "tokenizer.json"))
    meta = {"vocab_size": vocab, "train_chars": len(train_text), "valid_chars": len(valid_text),
            "reference_text": {s: {"chars": len(t), "sha256": sha256(staging / f"reference_{s}.txt")}
                               for s, t in (("train", train_text), ("valid", valid_text))},
            "reference_dataset": f"{args.train_source} prefix", "lossless": True,
            "build_s": time.monotonic() - started}
    (staging / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    staging.rename(args.output_dir)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
