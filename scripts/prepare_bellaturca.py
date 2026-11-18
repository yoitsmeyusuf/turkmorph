"""BellaTurca sample -> data/corpus/{train,valid}.txt + data/cache/bt_counts.pkl

Documents are split by a hash of their text (1% held out): the held-out part is
never seen by any tokenizer or LM training and is used only for BPC."""
import glob
import hashlib
import json
import os
import pickle
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from turkmorph.tokenizer import CHUNK, tr_lower                # noqa: E402

DATA = ROOT / "data"
SENT_END = re.compile(r"[.!?…:]\s*$")


def _count(path):
    words, proper, chunks = Counter(), Counter(), Counter()
    train, valid = [], []
    for line in open(path, encoding="utf-8"):
        text = json.loads(line)["text"]
        held = int(hashlib.md5(text.encode()).hexdigest(), 16) % 100 == 0
        (valid if held else train).append(text)
        if held:
            continue
        prev_end = True
        for m in CHUNK.finditer(text):
            w = m.group(1)
            if w:
                chunks[w] += 1
                base, *rest = re.split(r"['’]", w)
                low = tr_lower(base)
                if not rest:
                    words[low] += 1
                if base[:1].isupper() and (rest or not prev_end):
                    proper[low] += 1
                prev_end = False
            elif m.group(4):
                prev_end = m.group(4) in ".!?…:"
    return words, proper, chunks, train, valid


def main():
    files = sorted(glob.glob(str(DATA / "raw" / "bellaturca" / "*.jsonl")))
    (DATA / "corpus").mkdir(exist_ok=True)
    (DATA / "cache").mkdir(exist_ok=True)
    words, proper, chunks = Counter(), Counter(), Counter()
    with open(DATA / "corpus" / "train.txt", "w", encoding="utf-8") as ft, \
            open(DATA / "corpus" / "valid.txt", "w", encoding="utf-8") as fv, \
            Pool(int(os.environ.get("TURKMORPH_PROCS") or os.cpu_count())) as pool:
        for w, p, c, tr, va in pool.imap(_count, files):
            words.update(w)
            proper.update(p)
            chunks.update(c)
            for t in tr:
                ft.write(t.replace("\n", " ") + "\n")
            for t in va:
                fv.write(t.replace("\n", " ") + "\n")
    pickle.dump({"words": words, "proper": proper, "chunks": chunks},
                open(DATA / "cache" / "bt_counts.pkl", "wb"))
    print(f"tokens {sum(words.values())/1e6:.1f}M, types {len(words)}, "
          f"types>=3 {sum(c >= 3 for c in words.values())}, chunk types {len(chunks)}")
    for f in ("train", "valid"):
        print(f, os.path.getsize(DATA / "corpus" / f"{f}.txt") // 10**6, "MB")


if __name__ == "__main__":
    main()
