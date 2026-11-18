"""Prepare identical text segments for both tokenizers, without truncating either."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAYON_NUM_THREADS", "4")

from lm.experiment import sha256


def matched_segments(text, encode, max_tokens=512, chunk_chars=1024):
    """Yield contiguous text with all encodings fitting the same model context."""
    def fit(piece):
        ids = {name: fn(piece) for name, fn in encode.items()}
        if all(0 < len(value) <= max_tokens for value in ids.values()):
            yield piece, ids
            return
        if len(piece) < 2:
            raise ValueError("A single character cannot fit in the token context")
        middle = len(piece) // 2
        cut = max(piece.rfind(" ", 0, middle), piece.rfind("\n", 0, middle)) + 1
        cut = cut if cut else middle
        yield from fit(piece[:cut])
        yield from fit(piece[cut:])

    start = 0
    while start < len(text):
        stop = min(start + chunk_chars, len(text))
        if stop < len(text):
            boundary = max(text.rfind(" ", start, stop), text.rfind("\n", start, stop),
                           text.rfind("\t", start, stop))
            if boundary >= start:
                stop = boundary + 1
        yield from fit(text[start:stop])
        start = stop


def write_split(text, split, output, encode, decode, max_tokens, chunk_chars):
    offsets = {name: [0] for name in encode}
    chars = []
    digest = hashlib.sha256()
    handles = {name: (output / name / f"{split}.bin").open("wb") for name in encode}
    try:
        with (output / f"{split}_segments.jsonl").open("w", encoding="utf-8") as manifest:
            position = 0
            for piece, encodings in matched_segments(text, encode, max_tokens, chunk_chars):
                record = {"start": position, "end": position + len(piece),
                          "sha256": hashlib.sha256(piece.encode("utf-8")).hexdigest(), "tokens": {}}
                for name, ids in encodings.items():
                    if decode[name](ids) != piece:
                        raise ValueError(f"{name} cannot decode the segment at character {position}")
                    np.asarray(ids, dtype="<u4").tofile(handles[name])
                    offsets[name].append(offsets[name][-1] + len(ids))
                    record["tokens"][name] = len(ids)
                manifest.write(json.dumps(record) + "\n")
                chars.append(len(piece))
                digest.update(piece.encode("utf-8"))
                position += len(piece)
    finally:
        for handle in handles.values():
            handle.close()
    if sum(chars) != len(text) or digest.hexdigest() != hashlib.sha256(text.encode("utf-8")).hexdigest():
        raise ValueError("Segmenting dropped or duplicated text")
    np.save(output / f"{split}_chars.npy", np.asarray(chars, dtype="<i8"))
    for name in encode:
        np.save(output / name / f"{split}_offsets.npy", np.asarray(offsets[name], dtype="<i8"))
    return {"chars": len(text), "segments": len(chars), "sha256": digest.hexdigest(),
            "tokens": {name: value[-1] for name, value in offsets.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "lm")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data" / "lm" / "equal-text-30m")
    ap.add_argument("--train-chars", type=int, default=30_000_000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--chunk-chars", type=int, default=1024)
    args = ap.parse_args()
    if min(args.train_chars, args.max_tokens, args.chunk_chars) < 1:
        ap.error("Character and token limits must be positive")
    if args.output_dir.exists():
        ap.error("Output directory already exists; refusing to overwrite it")
    from tokenizers import Tokenizer
    from turkmorph.fast import fast_tokenizer

    bpe_dir = args.data_root / "bpe"
    bpe_meta = json.loads((bpe_dir / "meta.json").read_text())
    for split in ("train", "valid"):
        if sha256(bpe_dir / f"reference_{split}.txt") != bpe_meta["reference_text"][split]["sha256"]:
            raise ValueError(f"Reference {split} text has changed")
    morph = fast_tokenizer()
    bpe = Tokenizer.from_file(str(bpe_dir / "tokenizer.json"))
    if len(morph.vocab) != bpe.get_vocab_size() or len(morph.vocab) != bpe_meta["vocab_size"]:
        raise ValueError("Tokenizer vocabulary sizes differ")
    output = args.output_dir.with_name(args.output_dir.name + f".preparing-{os.getpid()}")
    output.mkdir(parents=True)
    for name in ("morph", "bpe"):
        (output / name).mkdir()
    encode = {"morph": morph.encode_ids, "bpe": lambda text: bpe.encode(text).ids}
    decode = {"morph": morph.decode_ids, "bpe": bpe.decode}
    started = time.monotonic()
    splits = {}
    for split in ("train", "valid"):
        with (bpe_dir / f"reference_{split}.txt").open(encoding="utf-8") as source:
            text = source.read(args.train_chars) if split == "train" else source.read()
        if split == "train" and len(text) != args.train_chars:
            raise ValueError("Not enough training characters")
        splits[split] = write_split(text, split, output, encode, decode, args.max_tokens, args.chunk_chars)
        print(split, json.dumps(splits[split]), flush=True)
    artifacts = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    meta = {"protocol": "paired text segments, one shuffled pass, identical order per seed",
            "source_reference": bpe_meta["reference_text"], "base_vocab_size": len(morph.vocab),
            "vocab_size": len(morph.vocab) + 1, "bos_id": len(morph.vocab),
            "max_tokens": args.max_tokens, "chunk_chars": args.chunk_chars,
            "splits": splits, "artifacts": artifacts,
            "tokenizer_sha256": {"bpe": sha256(bpe_dir / "tokenizer.json"),
                                 "morph_vocab": hashlib.sha256(json.dumps(morph.vocab.toks,
                                     ensure_ascii=False).encode()).hexdigest()},
            "prepare_code_sha256": sha256(Path(__file__)), "build_s": time.monotonic() - started}
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    output.rename(args.output_dir)
    print(f"Prepared matched text: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
