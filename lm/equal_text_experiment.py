"""Run the three-seed comparison on identical text, boundaries, and sample order."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lm.experiment import sha256, run_training, summarize


def make_plan(args):
    meta = json.loads((args.data_root / "meta.json").read_text())
    if args.ctx < meta["max_tokens"]:
        raise ValueError("Model context is smaller than the prepared segment limit")
    for filename, expected in meta["artifacts"].items():
        if sha256(args.data_root / filename) != expected:
            raise ValueError(f"Dataset artifact changed: {filename}")
    common = {"data_root": str(args.data_root.resolve()), "d": args.d, "layers": args.layers,
              "ctx": args.ctx, "batch": args.batch, "lr": args.lr, "device": args.device,
              "duty_cycle": args.duty_cycle}
    runs = []
    for index, seed in enumerate(args.seeds):
        for name in (("morph", "bpe") if index % 2 == 0 else ("bpe", "morph")):
            runs.append({"name": f"{name}-seed{seed}", "dataset": name,
                         "config": {**common, "tokenizer": name, "seed": seed}})
    fingerprints = {name: sha256(ROOT / "lm" / name) for name in (
        "train_equal_text.py", "equal_text_experiment.py", "model.py", "train.py", "experiment.py")}
    return {"protocol": meta["protocol"], "seeds": args.seeds, "runs": runs,
            "data_meta": meta, "data_meta_sha256": sha256(args.data_root / "meta.json"),
            "code_sha256": fingerprints, "pause_seconds": args.pause_seconds,
            "note": "Identical text, segment boundaries, one-pass sample order and optimizer steps. "
                    "Token counts and compute differ. Shared BOS and per-character loss normalization "
                    "make this a new protocol; BPC is not directly comparable with earlier runs."}


def read_completed(run, directory, meta):
    if not (directory / "result.json").exists():
        return None
    config = json.loads((directory / "config.json").read_text())
    if config.get("data_meta") != meta or any(config.get(k) != v for k, v in run["config"].items()):
        raise ValueError(f"Saved run settings differ: {directory}")
    result = json.loads((directory / "result.json").read_text())
    seed = run["config"]["seed"]
    order = np.random.default_rng(seed).permutation(meta["splits"]["train"]["segments"])
    expected = {"data": run["dataset"], "seed": seed,
                "steps": (len(order) + run["config"]["batch"] - 1) // run["config"]["batch"],
                "train_chars_seen": meta["splits"]["train"]["chars"],
                "train_tokens_seen": meta["splits"]["train"]["tokens"][run["dataset"]],
                "train_order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
                "train_text_sha256": meta["splits"]["train"]["sha256"],
                "valid_chars": meta["splits"]["valid"]["chars"],
                "valid_tokens_scored": meta["splits"]["valid"]["tokens"][run["dataset"]],
                "valid_text_sha256": meta["splits"]["valid"]["sha256"]}
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Text coverage, seed or sample order differs: {directory}")
    if not math.isfinite(result["valid_bpc"]) or not (directory / "model.pt").exists():
        raise ValueError(f"Invalid result or missing checkpoint: {directory}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "lm" / "equal-text-30m")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=.001)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--duty-cycle", type=float, default=.7)
    ap.add_argument("--pause-seconds", type=float, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if (len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2
            or args.batch < 1 or args.d < 64 or args.d % 64 or args.layers < 1
            or not 0 < args.duty_cycle <= 1 or args.pause_seconds < 0 or args.lr <= 0):
        ap.error("Invalid seeds, model, batch, learning rate or pacing")
    args.output_dir = args.output_dir.resolve()
    plan = make_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Experiment plan changed; choose a new output directory")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    commands = {}
    results = {}
    for run in plan["runs"]:
        directory = args.output_dir / run["name"]
        command = [sys.executable, "-u", str(ROOT / "lm" / "train_equal_text.py")]
        for key, value in run["config"].items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        commands[run["name"]] = command + ["--output-dir", str(directory)]
        result = read_completed(run, directory, plan["data_meta"])
        if result:
            results[run["name"]] = result
    train_info = plan["data_meta"]["splits"]["train"]
    print(f"{len(plan['runs'])}-run matched-text comparison: {train_info['chars']:,} characters, "
          f"{train_info['segments']:,} identical segments per run; {args.output_dir}", flush=True)
    if args.dry_run:
        for name, command in commands.items():
            print(name, json.dumps(command), flush=True)
        return
    if args.device == "cuda" and len(results) < len(plan["runs"]):
        import torch
        if not torch.cuda.is_available():
            ap.error("CUDA unavailable here; run this command in the GPU-enabled WSL terminal")
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        x.square().mean().backward()
        torch.cuda.synchronize()
        print(f"CUDA forward/backward passed: {torch.cuda.get_device_name(0)}", flush=True)
    trained = False
    for run in plan["runs"]:
        if run["name"] in results:
            print(f"Already complete: {run['name']}", flush=True)
            continue
        directory = args.output_dir / run["name"]
        if (directory / "model.pt").exists():
            raise RuntimeError(f"Checkpoint without evaluation result: {directory}; preserve it and choose a new experiment directory")
        if trained and args.pause_seconds:
            print(f"Idle pause: {args.pause_seconds:g} seconds", flush=True)
            time.sleep(args.pause_seconds)
        directory.mkdir(exist_ok=True)
        print(f"Starting {run['name']}", flush=True)
        run_training(commands[run["name"]], directory / "train.log")
        result = read_completed(run, directory, plan["data_meta"])
        if result is None:
            raise RuntimeError(f"Training produced no result: {directory}")
        results[run["name"]] = result
        (args.output_dir / "progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        trained = True
    summarize(plan, results, args.output_dir)


if __name__ == "__main__":
    main()
