"""Run equal-text LM experiments on Modal GPUs.

  modal volume put turkmorph-lm data/lm/equal-text-300m /data/equal-text-300m      # once
  modal run lm/modal_run.py --plan bench          # throughput + cost estimate, a few minutes
  modal run lm/modal_run.py --plan M              # morph vs BPE, d768 L12
  modal run lm/modal_run.py --runs M-bpe-b32-lr6e-4-s0,...   # any spec: SIZE-TOK-bBATCH-lrLR-sSEED

Results (result.json, test_result.json, config.json, train.log) are copied to
data/lm/experiments/<experiment>/<run>/; checkpoints stay on the volume.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = "H100"
SIZES = {"S": (384, 6), "M": (768, 12), "L": (1024, 24)}
HG = ["--arch", "hourglass", "--n-pre", "4", "--n-mid", "16", "--n-post", "4"]
ARCH = {"HL": ("L", HG), "HLW": ("L", HG + ["--d-mid", "1408"])}       # hourglass at L width
# long-context experiment on the HF corpus (Wikipedia-tr + FineWeb-2, 1B chars, <=2048 tokens/segment)
DATASETS = {"bellaturca-300m": ("equal-text-300m", "equal-text-300m-test", "scaling-300m", 512),
            # data axis: same 1B-char data, train on a 300M-char prefix of the seed order
            "hf-300m-long": ("equal-text-hf-1b-long", "equal-text-hf-test-long", "hf-300m-long", 2048, 300_000_000),
            "hf-1b-long": ("equal-text-hf-1b-long", "equal-text-hf-test-long", "hf-1b-long", 2048)}
PLANS = {
    "bench": ["M-morph-b16-lr6e-4-s0", "M-bpe-b16-lr6e-4-s0", "L-morph-b16-lr3e-4-s0"],
    "M": ["M-morph-b16-lr6e-4-s0", "M-bpe-b16-lr6e-4-s0"],
    "L": ["L-morph-b16-lr3e-4-s0", "L-bpe-b16-lr3e-4-s0"],
    "phase1": ["L-morph-b16-lr3e-4-s0", "L-bpe-b16-lr3e-4-s0",
               "M-morph-b16-lr6e-4-s0", "M-bpe-b16-lr6e-4-s0",
               "M-morph-b32-lr1e-3-s0", "M-bpe-b32-lr1e-3-s0"],
    # phase 2: is BPE still undertuned at M (b64)? L with the better (b32) setting; seeds for M best.
    "phase2": ["L-morph-b32-lr6e-4-s0", "L-bpe-b32-lr6e-4-s0",
               "M-morph-b64-lr1.5e-3-s0", "M-bpe-b64-lr1.5e-3-s0",
               "M-morph-b32-lr1e-3-s1", "M-bpe-b32-lr1e-3-s1",
               "M-morph-b32-lr1e-3-s2", "M-bpe-b32-lr1e-3-s2"],
    # phase 3: where does the batch trend saturate?
    "hg-bench": ["L-morph-b16-lr8e-4-s0", "HL-morph-b16-lr8e-4-s0", "HLW-morph-b16-lr8e-4-s0"],
    "hg": ["L-morph-b16-lr8e-4-s0", "HL-morph-b16-lr8e-4-s0", "HLW-morph-b16-lr8e-4-s0"],
    "b128": ["M-morph-b128-lr2e-3-s0", "M-bpe-b128-lr2e-3-s0",
             "M-morph-b128-lr3e-3-s0", "M-bpe-b128-lr3e-3-s0"],
}

app = modal.App("turkmorph-lm")
volume = modal.Volume.from_name("turkmorph-lm", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.14.1", "numpy")
         .add_local_dir(ROOT / "lm", "/root/proj/lm", ignore=["__pycache__", "*.pyc"]))


def parse(spec):
    head, seed = spec.rsplit("-s", 1)                       # lr may contain "-" (6e-4)
    size, tok, batch, lr = head.split("-", 3)
    return {"name": spec, "size": size, "tokenizer": tok, "batch": int(batch[1:]),
            "lr": float(lr[2:]), "seed": int(seed)}


@app.function(image=image, gpu=GPU, volumes={"/vol": volume}, timeout=8 * 3600)
def train(spec: str, benchmark_steps: int = 0, dataset: str = "bellaturca-300m") -> dict:
    run = parse(spec)
    data, test_data, experiment, ctx, *limit = DATASETS[dataset]
    size, extra = ARCH.get(run["size"], (run["size"], []))
    d, layers = SIZES[size]
    out = Path("/vol/runs") / experiment / (spec + ("-bench" if benchmark_steps else ""))
    if not benchmark_steps and (out / "result.json").exists() and (out / "test_result.json").exists():
        print(f"{spec}: already done", flush=True)
    else:
        if out.exists():                                   # partial run from a preempted container
            subprocess.run(["rm", "-rf", str(out)], check=True)
        out.mkdir(parents=True)
        cmd = [sys.executable, "-u", "/root/proj/lm/train_equal_text.py", "--data-root", f"/vol/data/{data}",
               "--tokenizer", run["tokenizer"], "--output-dir", str(out), "--seed", str(run["seed"]),
               "--d", str(d), "--layers", str(layers), "--batch", str(run["batch"]), "--lr", str(run["lr"]),
               "--duty-cycle", "1", "--benchmark-steps", str(benchmark_steps),
               "--ctx", str(ctx), *extra,
               *(["--train-chars", str(limit[0])] if limit else []),
               "--accum", str(max(1, run["batch"] * ctx // 32768))]   # micro-batches of <=32K token slots (80 GB)
        started = time.monotonic()
        with (out / "train.log").open("w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log.write(line)
                if "step" not in line or "/50 " in line or line.startswith("step 1/") or " 0.0 min" in line:
                    print(f"[{spec}] {line}", end="", flush=True)
                elif line.startswith("step ") and int(line.split()[1].split("/")[0]) % 2000 == 1:
                    print(f"[{spec}] {line}", end="", flush=True)
            if proc.wait():
                volume.commit()
                raise RuntimeError(f"{spec} failed; see {out}/train.log")
        if not benchmark_steps:
            sys.path.insert(0, "/root/proj")
            from lm.eval_equal_text import evaluate
            test_root = Path(f"/vol/data/{test_data}")
            meta = json.loads((test_root / "meta.json").read_text())
            test = evaluate(out, test_root, meta, "cuda", run["batch"])
            test["valid_bpc"] = json.loads((out / "result.json").read_text())["valid_bpc"]
            (out / "test_result.json").write_text(json.dumps(test, indent=2))
        print(f"{spec}: finished in {(time.monotonic() - started) / 60:.1f} min", flush=True)
        volume.commit()
    files = {p.name: p.read_text() for p in out.iterdir() if p.suffix in (".json", ".log")}
    return {"spec": spec, "files": files}


@app.local_entrypoint()
def main(plan: str = "", runs: str = "", benchmark_steps: int = 0, dataset: str = ""):
    specs = PLANS[plan] if plan else [s for s in runs.split(",") if s]
    dataset = dataset or ("hf-1b-long" if plan.startswith("hg") else "bellaturca-300m")
    if plan.endswith("bench") and not benchmark_steps:
        benchmark_steps = 300
    started = time.monotonic()
    print(f"{len(specs)} runs on {GPU}, data {dataset}: {', '.join(specs)}", flush=True)
    for res in train.map(specs, kwargs={"benchmark_steps": benchmark_steps, "dataset": dataset},
                         return_exceptions=True):
        if isinstance(res, Exception):
            print("FAILED:", res, flush=True)
            continue
        name = res["spec"] + ("-bench" if benchmark_steps else "")
        local = ROOT / "data" / "lm" / "experiments" / DATASETS[dataset][2] / name
        local.mkdir(parents=True, exist_ok=True)
        for fname, text in res["files"].items():
            (local / fname).write_text(text)
        if benchmark_steps:
            line = [l for l in res["files"].get("train.log", "").splitlines() if "benchmark_steps" in l]
            print(name, line[-1] if line else "no benchmark line", flush=True)
        else:
            test = json.loads(res["files"]["test_result.json"])
            print(f"{name}: valid {test['valid_bpc']:.4f}  test {test['test_bpc']:.4f}", flush=True)
    print(f"all done in {(time.monotonic() - started) / 60:.1f} min", flush=True)
