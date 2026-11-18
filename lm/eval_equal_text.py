"""Score finished equal-text checkpoints on the held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lm.train_equal_text import load_split, text_batch        # sets the CUDA allocator option before torch
import torch
from lm.model import GPT, bpc


def evaluate(run_dir, test_root, meta, device, batch):
    saved = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    config = saved["config"]
    trained = config["data_meta"]
    if (trained["tokenizer_sha256"] != meta["tokenizer_sha256"] or trained["vocab_size"] != meta["vocab_size"]
            or trained["bos_id"] != meta["bos_id"]):
        raise ValueError(f"Checkpoint and test split use different tokenizers: {run_dir}")
    split = load_split(test_root, config["tokenizer"], "test")
    if int(np.diff(split[1]).max()) > config["ctx"]:
        raise ValueError("A test segment exceeds model context")
    model = GPT(meta["vocab_size"], config["d"], config["layers"], config["d"] // 64, config["ctx"])
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    amp = device == "cuda" and torch.cuda.is_bf16_supported()
    total, chars_seen, tokens_seen = 0., 0, 0
    with torch.no_grad():
        for start in range(0, len(split[2]), batch):
            indices = np.arange(start, min(start + batch, len(split[2])))
            x, y, chars, tokens = text_batch(split, indices, meta["bos_id"])
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                total += model(x.to(device), y.to(device)).item()
            chars_seen += chars
            tokens_seen += tokens
    expected = meta["splits"]["test"]
    if chars_seen != expected["chars"] or tokens_seen != expected["tokens"][config["tokenizer"]]:
        raise ValueError("Test text coverage mismatch")
    return {"data": config["tokenizer"], "seed": config["seed"], "test_bpc": bpc(total, chars_seen),
            "test_chars": chars_seen, "test_tokens_scored": tokens_seen,
            "test_text_sha256": expected["sha256"],
            "train_chars_seen": saved["train_chars_seen"], "train_tokens_seen": saved["train_tokens_seen"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, default=ROOT / "data" / "lm" / "equal-text-test-2m")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    meta = json.loads((args.test_root / "meta.json").read_text())
    results = {}
    for run_dir in sorted(p.parent for p in args.experiment_dir.glob("*/result.json")):
        path = run_dir / "test_result.json"
        if path.exists():
            results[run_dir.name] = json.loads(path.read_text())
        else:
            results[run_dir.name] = evaluate(run_dir, args.test_root, meta, args.device, args.batch)
            valid = json.loads((run_dir / "result.json").read_text())["valid_bpc"]
            results[run_dir.name]["valid_bpc"] = valid
            path.write_text(json.dumps(results[run_dir.name], indent=2), encoding="utf-8")
        print(run_dir.name, f"test BPC {results[run_dir.name]['test_bpc']:.6f}", flush=True)
    lines = ["| Run | Train chars | Train tokens | Valid BPC ↓ | Test BPC ↓ |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['train_chars_seen']:,} | {r['train_tokens_seen']:,} | "
                     f"{r['valid_bpc']:.6f} | {r['test_bpc']:.6f} |")
    lines.append("")
    for name in sorted({r["data"] for r in results.values()}):
        values = [r["test_bpc"] for r in results.values() if r["data"] == name]
        spread = f", sample SD {statistics.stdev(values):.6f}" if len(values) > 1 else ""
        lines.append(f"{name}: test mean {statistics.mean(values):.6f}{spread} ({len(values)} runs)")
    (args.experiment_dir / "test_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.experiment_dir / "test_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
