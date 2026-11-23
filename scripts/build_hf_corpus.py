"""Build data/hf/corpus/{train,valid,test,valid_test}.txt from the downloaded HF parquet files.

train: 300M chars Wikipedia-tr + 700M chars FineWeb-2 tur_Latn, documents shuffled together
valid: 1M + 1M chars from documents not used in train
test:  first ~2M chars of the FineWeb-2 tur_Latn *test* shard (official held-out split)
Documents are separated by a blank line. Cleaning: drop \\r, collapse 3+ newlines, drop docs < 200 chars.
"""
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "hf"
OUT = HF / "corpus"
WIKI = sorted((HF / "wikipedia").rglob("*.parquet"))
WEB = sorted((HF / "fineweb-2" / "data" / "tur_Latn" / "train").glob("*.parquet"))
WEB_TEST = sorted((HF / "fineweb-2" / "data" / "tur_Latn" / "test").glob("*.parquet"))
PLAN = {"wiki": (WIKI, 300_000_000, 0.40), "web": (WEB, 700_000_000, 0.08)}   # files, train chars, keep prob
VALID_EACH, TEST_CHARS = 1_000_000, 2_000_000


def clean(t):
    t = re.sub(r"\n{3,}", "\n\n", t.replace("\r", "")).strip()
    return t if len(t) >= 200 else None


def docs(files, keep, rng):
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for t in pf.read_row_group(g, columns=["text"]).column("text").to_pylist():
                if rng.random() < keep and (t := clean(t)):
                    yield t


def take(it, n):
    out, size = [], 0
    for d in it:
        out.append(d)
        size += len(d) + 2
        if size >= n:
            return out
    raise SystemExit(f"not enough text: {size:,} < {n:,}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261123)
    train, valid, stats = [], [], {}
    for name, (files, n, keep) in PLAN.items():
        it = docs(files, keep, rng)
        v = take(it, VALID_EACH)
        t = take(it, n)
        valid += v
        train += t
        stats[name] = {"train_docs": len(t), "train_chars": sum(map(len, t)), "valid_docs": len(v)}
        print(name, stats[name], flush=True)
    rng.shuffle(train)
    rng.shuffle(valid)
    test = take(docs(WEB_TEST, 1.0, rng), TEST_CHARS)
    texts = {"train": "\n\n".join(train) + "\n", "valid": "\n\n".join(valid) + "\n", "test": "\n\n".join(test) + "\n"}
    for k, v in texts.items():
        (OUT / f"{k}.txt").write_text(v, encoding="utf-8")
    (OUT / "valid_test.txt").write_text(texts["valid"] + texts["test"], encoding="utf-8")
    # no document of valid/test may appear in train
    seen = {hashlib.sha1(d.encode()).digest() for d in train}
    overlap = {k: sum(hashlib.sha1(d.encode()).digest() in seen for d in ds) for k, ds in (("valid", valid), ("test", test))}
    stats.update({k: len(v) for k, v in texts.items()}, overlap=overlap,
                 sources={"wikipedia": "wikimedia/wikipedia 20231101.tr (CC BY-SA 4.0)",
                          "fineweb2": "HuggingFaceFW/fineweb-2 tur_Latn (ODC-By 1.0)"})
    (OUT / "meta.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
