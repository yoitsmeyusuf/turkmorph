"""Throughput from Python: turkmorph.fast (Rust) vs Python TurkMorph vs TurkishTokenizer.
    .venv-tt/bin/python scripts/bench_fast.py TEXTS.jsonl     (one JSON string per line)
Each runs twice: cold (empty cache) and warm."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turkmorph.fast import fast_tokenizer  # noqa: E402

texts = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
words = sum(len(t.split()) for t in texts)
print(f"{len(texts)} texts, {words} words")


def run(name, fn, data=texts):
    w = sum(len(t.split()) for t in data)
    for p in ("cold", "warm"):
        t0 = time.perf_counter()
        fn(data)
        dt = time.perf_counter() - t0
        print(f"{name:34s} {p}: {dt:7.2f}s  {w / dt:>12,.0f} words/s", flush=True)


rs = fast_tokenizer()
run("turkmorph.fast 1 thread", lambda d: [rs.encode_ids(t) for t in d])
rs = fast_tokenizer()
run("turkmorph.fast batch 6 threads", lambda d: rs.encode_batch(d, 6))
try:
    import turkish_tokenizer as tt
    T = tt.TurkishTokenizer()
    run("TurkishTokenizer 1.0.4", lambda d: [T.encode(t) for t in d])
except ImportError:
    print("turkish_tokenizer not installed: skipped")
from turkmorph.presets import full_tokenizer  # noqa: E402
py = full_tokenizer()
run("turkmorph Python (first 300 texts)", lambda d: [py.encode_ids(t) for t in d], texts[:300])
