"""Python side of the Rust tokenizer conformance test (same format as rust/src/bin/encode.rs).
    python3 scripts/conformance_encode.py < texts.jsonl > py.txt"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TOK = None


def _init():
    global TOK
    from turkmorph.presets import full_tokenizer
    TOK = full_tokenizer()


def line(text):
    toks = TOK.encode(text)
    return "\x1f".join(toks) + "\t" + ("ok" if TOK.decode(toks) == text else "BAD")


if __name__ == "__main__":
    texts = [json.loads(x) for x in sys.stdin]
    with Pool(int(os.environ.get("TURKMORPH_PROCS") or 6), initializer=_init) as p:
        for out in p.imap(line, texts, chunksize=50):
            print(out)
