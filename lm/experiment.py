"""Run the paired, three-seed GPT tokenizer comparison sequentially."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_plan(args):
    datasets, fingerprints = {}, {}
    for name in ("morph", "bpe"):
        directory = args.data_root / name
        meta = json.loads((directory / "meta.json").read_text())
        for split in ("train", "valid"):
            path = directory / f"{split}.bin"
            if path.stat().st_size != meta[f"{split}_tokens"] * 4:
                raise ValueError(f"Token count differs from metadata: {path}")
            fingerprints[f"{name}/{split}.bin"] = sha256(path)
        fingerprints[f"{name}/meta.json"] = sha256(directory / "meta.json")
        if meta["train_tokens"] < args.max_tokens:
            raise ValueError(f"{name} has fewer tokens than the requested budget")
        if meta["train_tokens"] <= args.ctx + 1 or meta["valid_tokens"] < 2:
            raise ValueError(f"{name}: not enough data for the requested context")
        datasets[name] = meta
    if datasets["morph"]["vocab_size"] != datasets["bpe"]["vocab_size"]:
        raise ValueError("Vocabulary sizes differ")
    if datasets["morph"]["valid_chars"] != datasets["bpe"]["valid_chars"]:
        raise ValueError("Validation character counts differ")
    if not datasets["bpe"].get("lossless") or datasets["bpe"].get("reference_dataset") != "morph":
        raise ValueError("BPE data must come from prepare_bpe.py's verified morph reference")
    for split in ("train", "valid"):
        path = args.data_root / "bpe" / f"reference_{split}.txt"
        if sha256(path) != datasets["bpe"]["reference_text"][split]["sha256"]:
            raise ValueError(f"BPE reference text has changed: {split}")
    for name in ("train.py", "model.py", "experiment.py"):
        fingerprints[f"code/{name}"] = sha256(ROOT / "lm" / name)
    common = {"model": "gpt", "d": args.d, "layers": args.layers, "ctx": args.ctx,
              "batch": args.batch, "lr": args.lr, "max_tokens": args.max_tokens,
              "device": args.device, "duty_cycle": args.duty_cycle, "tag": "paired-token-budget",
              "results_file": str(args.output_dir / "results.jsonl")}
    runs = []
    for i, seed in enumerate(args.seeds):
        # Alternate order so one tokenizer does not always follow a cold GPU.
        for name in (("morph", "bpe") if i % 2 == 0 else ("bpe", "morph")):
            runs.append({"name": f"{name}-seed{seed}", "dataset": name,
                         "config": {**common, "data": str((args.data_root / name).resolve()),
                                    "seed": seed}})
    return {"protocol": "equal nominal training tokens; random windows with replacement",
            "note": "Different tokenizers see different amounts of text; this is not an equal-text test.",
            "seeds": args.seeds, "datasets": datasets, "fingerprints": fingerprints,
            "pause_seconds": args.pause_seconds, "runs": runs}


def command_for(run, output):
    command = [sys.executable, "-u", str(ROOT / "lm" / "train.py")]
    for key, value in run["config"].items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return command + ["--output-dir", str(output)]


def load_completed(run, directory):
    result_path = directory / "result.json"
    if not result_path.exists():
        return None
    config = json.loads((directory / "config.json").read_text())
    for key, value in run["config"].items():
        if config.get(key) != value:
            raise ValueError(f"Completed run has a different {key}: {directory}")
    result = json.loads(result_path.read_text())
    expected = run["config"]["max_tokens"] // (
        run["config"]["batch"] * run["config"]["ctx"]) * run["config"]["batch"] * run["config"]["ctx"]
    if result["seed"] != run["config"]["seed"] or result["train_tokens_seen"] != expected:
        raise ValueError(f"Completed run has a different seed or training budget: {directory}")
    if not math.isfinite(result["valid_bpc"]) or not (directory / "model.pt").exists():
        raise ValueError(f"Incomplete or invalid training artifacts: {directory}")
    return result


def run_training(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                raise RuntimeError(f"Training exited with code {code}; see {log_path}")


def summarize(plan, results, directory):
    values = {name: [results[f"{name}-seed{seed}"]["valid_bpc"] for seed in plan["seeds"]]
              for name in ("morph", "bpe")}
    if len({r["params"] for r in results.values()}) != 1:
        raise ValueError("Trained parameter counts differ")
    stats = {name: {"mean_bpc": statistics.mean(v), "sample_sd": statistics.stdev(v), "bpc": v}
             for name, v in values.items()}
    deltas = [b - m for m, b in zip(values["morph"], values["bpe"])]
    report = {"protocol": plan["protocol"], "seeds": plan["seeds"], "stats": stats,
              "paired_bpe_minus_morph": deltas,
              "mean_paired_difference": statistics.mean(deltas),
              "paired_difference_sample_sd": statistics.stdev(deltas),
              "morph_lower_bpc_seeds": sum(d > 0 for d in deltas),
              "morph_relative_bpc_reduction_percent": (
                  (stats["bpe"]["mean_bpc"] - stats["morph"]["mean_bpc"]) /
                  stats["bpe"]["mean_bpc"] * 100),
              "note": "Three seeds describe variability; this alone does not establish statistical significance.",
              "runs": results}
    (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["| Seed | Morph BPC ↓ | BPE BPC ↓ | BPE − Morph |",
             "| --- | ---: | ---: | ---: |"]
    for seed, m, b in zip(plan["seeds"], values["morph"], values["bpe"]):
        lines.append(f"| {seed} | {m:.6f} | {b:.6f} | {b-m:+.6f} |")
    lines.extend(["", *(f"{name}: mean {s['mean_bpc']:.6f}, sample SD {s['sample_sd']:.6f}"
                           for name, s in stats.items()), "", report["note"], plan["note"]])
    text = "\n".join(lines) + "\n"
    (directory / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "lm")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-tokens", type=int, default=10_000_000)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--duty-cycle", type=float, default=0.7)
    ap.add_argument("--pause-seconds", type=float, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        ap.error("Provide at least two distinct seeds")
    if (args.max_tokens < args.batch * args.ctx or args.batch < 1 or args.ctx < 1
            or args.d < 64 or args.d % 64 or args.layers < 1
            or not 0 < args.duty_cycle <= 1 or args.pause_seconds < 0):
        ap.error("Invalid training budget, dimensions, duty cycle or pause")
    args.output_dir = args.output_dir.resolve()
    plan = prepare_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Saved experiment plan differs; use a new output directory")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Experiment: {args.output_dir}; {len(plan['runs'])} runs", flush=True)
    if args.dry_run:
        for run in plan["runs"]:
            print(run["name"], json.dumps(command_for(run, args.output_dir / run["name"])))
        return
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            ap.error("CUDA is unavailable here; launch this command in the GPU-enabled WSL terminal")
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        x.square().mean().backward()
        torch.cuda.synchronize()
        print(f"CUDA forward/backward passed: {torch.cuda.get_device_name(0)}", flush=True)
    results = {}
    trained = False
    for run in plan["runs"]:
        directory = args.output_dir / run["name"]
        result = load_completed(run, directory)
        if result is None:
            if (directory / "model.pt").exists():
                raise RuntimeError(f"Checkpoint exists without a result: {directory}; preserve it and use a new experiment directory")
            if trained and args.pause_seconds:
                print(f"Idle pause: {args.pause_seconds:g} seconds", flush=True)
                time.sleep(args.pause_seconds)
            directory.mkdir(exist_ok=True)
            print(f"Starting {run['name']}", flush=True)
            run_training(command_for(run, directory), directory / "train.log")
            result = load_completed(run, directory)
            if result is None:
                raise RuntimeError(f"Training produced no result: {directory}")
            trained = True
        else:
            print(f"Already complete: {run['name']}", flush=True)
        results[run["name"]] = result
        (args.output_dir / "progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summarize(plan, results, args.output_dir)


if __name__ == "__main__":
    main()
