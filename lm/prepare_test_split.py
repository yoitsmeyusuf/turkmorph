"""Prepare a held-out test split from validation-file text no LM run has trained or been scored on."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAYON_NUM_THREADS", "4")

from lm.experiment import sha256
from lm.prepare_equal_text import write_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "lm")
    ap.add_argument("--matched-root", type=Path, default=ROOT / "data" / "lm" / "equal-text-30m")
    ap.add_argument("--source", type=Path, default=ROOT / "data" / "corpus" / "valid.txt")
    ap.add_argument("--bpe-dir", type=Path, default=None, help="default: <data-root>/bpe")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data" / "lm" / "equal-text-test-2m")
    ap.add_argument("--test-chars", type=int, default=2_000_000)
    args = ap.parse_args()
    if args.output_dir.exists():
        ap.error("Output directory already exists; refusing to overwrite it")
    from tokenizers import Tokenizer
    from turkmorph.fast import fast_tokenizer

    matched = json.loads((args.matched_root / "meta.json").read_text())
    used = matched["splits"]["valid"]["chars"]
    full = args.source.read_text(encoding="utf-8")
    if hashlib.sha256(full[:used].encode("utf-8")).hexdigest() != matched["splits"]["valid"]["sha256"]:
        raise ValueError("Source does not start with the validation text; cannot prove disjointness")
    start = full.rfind("\n", 0, len(full) - args.test_chars) + 1      # whole lines from the end of the file
    if start <= used:
        raise ValueError("Not enough unused text after the validation split")
    text = full[start:]
    morph = fast_tokenizer()
    bpe_dir = args.bpe_dir or args.data_root / "bpe"
    bpe = Tokenizer.from_file(str(bpe_dir / "tokenizer.json"))
    tokenizer_hashes = {"bpe": sha256(bpe_dir / "tokenizer.json"),
                        "morph_vocab": hashlib.sha256(json.dumps(morph.vocab.toks,
                            ensure_ascii=False).encode()).hexdigest()}
    if tokenizer_hashes != matched["tokenizer_sha256"]:
        raise ValueError("Tokenizers differ from the ones used for the matched training data")
    output = args.output_dir.with_name(args.output_dir.name + f".preparing-{os.getpid()}")
    output.mkdir(parents=True)
    for name in ("morph", "bpe"):
        (output / name).mkdir()
    split = write_split(text, "test", output,
                        {"morph": morph.encode_ids, "bpe": lambda piece: bpe.encode(piece).ids},
                        {"morph": morph.decode_ids, "bpe": bpe.decode},
                        matched["max_tokens"], matched["chunk_chars"])
    artifacts = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    meta = {"protocol": "held-out test; identical segments for both tokenizers; never trained or validated on",
            "source": str(args.source), "source_sha256": sha256(args.source),
            "source_char_range": [start, len(full)], "validation_char_range": [0, used],
            "base_vocab_size": matched["base_vocab_size"], "vocab_size": matched["vocab_size"],
            "bos_id": matched["bos_id"], "max_tokens": matched["max_tokens"],
            "chunk_chars": matched["chunk_chars"], "tokenizer_sha256": tokenizer_hashes,
            "splits": {"test": split}, "artifacts": artifacts}
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    output.rename(args.output_dir)
    print("test", json.dumps(split), flush=True)
    print(f"Prepared held-out test: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
