"""Build a BPE baseline from the exact text encoded in the existing morph dataset."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAYON_NUM_THREADS", "4")

from lm.prepare import train_bpe
from turkmorph.fast import fast_tokenizer

DATA = ROOT / "data"


def recover_text(tok, ids_path, source_path, output_path, batch_tokens=250_000):
    """Split after newline tokens so independently decoded chunks preserve word state."""
    ids = np.memmap(ids_path, dtype=np.uint32, mode="r")
    newline = tok.vocab.id("<nl>")
    digest = hashlib.sha256()
    chars = 0
    start = 0
    with source_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as out:
        while start < len(ids):
            stop = min(start + batch_tokens, len(ids))
            if stop < len(ids):
                boundaries = np.flatnonzero(ids[start:stop] == newline)
                if not len(boundaries):
                    raise ValueError("No safe newline boundary; increase batch_tokens")
                stop = start + int(boundaries[-1]) + 1
            text = tok.decode_ids(ids[start:stop].tolist())
            if source.read(len(text)) != text:
                raise ValueError(f"Decoded reference differs from {source_path} at/after character {chars}")
            out.write(text)
            digest.update(text.encode("utf-8"))
            chars += len(text)
            start = stop
    return {"chars": chars, "sha256": digest.hexdigest()}


def encode_file(tok, source_path, output_path):
    count = 0
    with source_path.open(encoding="utf-8") as source, output_path.open("wb") as out:
        while lines := list(itertools.islice(source, 256)):
            encodings = tok.encode_batch(lines)
            if tok.decode_batch([e.ids for e in encodings]) != lines:
                raise ValueError(f"BPE roundtrip failed for {source_path}")
            ids = np.asarray([i for e in encodings for i in e.ids], dtype=np.uint32)
            ids.tofile(out)
            count += len(ids)
    return count


def main():
    destination = DATA / "lm" / "bpe"
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite {destination}")
    staging = DATA / "lm" / f".bpe-prepare-{os.getpid()}"
    staging.mkdir()
    meta = json.loads((DATA / "lm" / "morph" / "meta.json").read_text())
    tok = fast_tokenizer()
    if len(tok.vocab) != meta["vocab_size"]:
        raise ValueError("Current morph vocabulary does not match the prepared dataset")
    references = {}
    started = time.monotonic()
    for split in ("train", "valid"):
        references[split] = recover_text(
            tok, DATA / "lm" / "morph" / f"{split}.bin",
            DATA / "corpus" / f"{split}.txt", staging / f"reference_{split}.txt")
        print(f"Verified {split}: {references[split]['chars']:,} characters", flush=True)
    if references["valid"]["chars"] != meta["valid_chars"]:
        raise ValueError("Validation text length does not match the reference metadata")
    print(f"Training byte-level BPE: vocabulary {meta['vocab_size']}", flush=True)
    bpe = train_bpe(staging / "reference_train.txt", meta["vocab_size"])
    bpe.save(str(staging / "tokenizer.json"))
    counts = {}
    for split in ("train", "valid"):
        counts[split] = encode_file(bpe, staging / f"reference_{split}.txt", staging / f"{split}.bin")
        print(f"Encoded {split}: {counts[split]:,} tokens; lossless", flush=True)
    result = {"vocab_size": bpe.get_vocab_size(), "train_tokens": counts["train"],
              "valid_tokens": counts["valid"], "valid_chars": references["valid"]["chars"],
              "train_chars": references["train"]["chars"], "reference_text": references,
              "reference_dataset": "morph", "lossless": True,
              "build_s": round(time.monotonic() - started, 2)}
    if result["vocab_size"] != meta["vocab_size"]:
        raise ValueError("BPE vocabulary budget was not reached")
    (staging / "meta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    staging.rename(destination)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
