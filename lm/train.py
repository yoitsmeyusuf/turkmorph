"""Train one LM on one tokenizer's encoding and report held-out BPC.

  python3 lm/train.py --data morph --model gpt
  python3 lm/train.py --data morph --model composer
  python3 lm/train.py --data morph --model gpt --max-tokens 13e6   # nominal token budget

Training samples random windows with replacement. By default the nominal token
budget is the size of train.bin; this is not an exact pass over the corpus.
--max-tokens caps that nominal budget, not FLOPs or wall time. Composer uses
average group length to estimate its steps. BPC = summed nats on valid /
valid characters / ln 2.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, Composer, bpc, n_params                  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "lm"


def learning_rate_scale(step, steps):
    warm = max(1, min(300, steps // 10))
    return min(1, (step + 1) / warm) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)))


def pace_step(started, duty_cycle, device):
    """Insert idle time after completed work; this is not a hardware power cap."""
    if duty_cycle == 1:
        return
    if device == "cuda":
        torch.cuda.synchronize()
    active = max(0, time.perf_counter() - started)
    time.sleep(active * (1 / duty_cycle - 1))


def word_spans(groups, k_max):
    """Start offsets of word groups; groups longer than k_max are split."""
    starts = np.flatnonzero(groups).astype(np.int64)
    ends = np.append(starts[1:], len(groups))
    out = []
    for s, e in zip(starts, ends):
        out.extend(range(s, e, k_max))
    starts = np.asarray(out, dtype=np.int64)
    return starts, np.append(starts[1:], len(groups))


def word_batch(ids, starts, ends, offsets, w, k_max):
    b = len(offsets)
    out = np.full((b, w, k_max), -1, dtype=np.int64)
    for i, o in enumerate(offsets):
        s, e = starts[o:o + w], ends[o:o + w]
        lens = e - s
        idx = s[:, None] + np.arange(k_max)[None, :]
        m = np.arange(k_max)[None, :] < lens[:, None]
        out[i][m] = ids[idx[m]]
    return torch.from_numpy(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="gpt", choices=["gpt", "composer"])
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=512)            # tokens (gpt) / words (composer: ctx // 2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-tokens", type=float, default=0)     # nominal token budget, not equal compute
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--results-file", type=Path,
                    help="aggregate JSONL path; defaults to data/lm/results.jsonl")
    ap.add_argument("--duty-cycle", type=float, default=1.0,
                    help="active fraction of each work/idle cycle, in (0, 1]; 0.7 adds idle time")
    args = ap.parse_args()
    if not 0 < args.duty_cycle <= 1:
        ap.error("--duty-cycle must be greater than 0 and at most 1")
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is unavailable in this process. Run nvidia-smi in the terminal "
                 "that will launch training; use --device cpu only for a small smoke test.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    d = DATA / args.data
    run_dir = args.output_dir or DATA / "runs" / (
        f"{d.name}-{args.model}-seed{args.seed}-{time.time_ns()}")
    if (run_dir / "model.pt").exists():
        ap.error(f"checkpoint already exists: {run_dir / 'model.pt'}; choose a new --output-dir")
    meta = json.loads((d / "meta.json").read_text())
    tr = np.fromfile(d / "train.bin", dtype=np.uint32).astype(np.int64)
    va = np.fromfile(d / "valid.bin", dtype=np.uint32).astype(np.int64)
    V = meta["vocab_size"]
    if args.model == "gpt":
        model = GPT(V, args.d, args.layers, args.d // 64, args.ctx).to(dev)
        n_train = len(tr) if not args.max_tokens else min(len(tr), int(args.max_tokens))
        per_step = args.batch * args.ctx
    else:
        W = args.ctx // 2
        model = Composer(V, args.d, args.layers, args.d // 64, W, args.k_max).to(dev)
        g_tr = np.fromfile(d / "groups_train.bin", dtype=np.uint8)
        g_va = np.fromfile(d / "groups_valid.bin", dtype=np.uint8)
        st_tr, en_tr = word_spans(g_tr, args.k_max)
        st_va, en_va = word_spans(g_va, args.k_max)
        n_train = len(tr) if not args.max_tokens else min(len(tr), int(args.max_tokens))
        tok_per_word = len(tr) / len(st_tr)
        per_step = int(args.batch * W * tok_per_word)
    steps = max(1, n_train // per_step)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
                            fused=dev == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: learning_rate_scale(s, steps))
    amp_enabled = dev == "cuda" and torch.cuda.is_bf16_supported()
    def autocast():
        return torch.autocast(dev, dtype=torch.bfloat16, enabled=amp_enabled)

    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "output_dir": str(run_dir), "data_meta": meta,
              "results_file": str(args.results_file) if args.results_file else None,
              "torch": str(torch.__version__), "device_name": (
                  torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"),
              "precision": "bfloat16" if amp_enabled else "float32"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"device: {config['device_name']}; precision: {config['precision']}; "
          f"duty cycle: {args.duty_cycle:.0%}; output: {run_dir}", flush=True)
    print(f"{args.data}/{args.model}: vocab {V}, non-emb params {n_params(model)/1e6:.1f}M, "
          f"steps {steps}, ~{per_step} tokens/step", flush=True)
    t0 = time.time()
    model.train()
    train_tokens_seen = 0
    for step in range(steps):
        step_started = time.perf_counter()
        if args.model == "gpt":
            o = np.random.randint(0, len(tr) - args.ctx - 1, args.batch)
            x = torch.from_numpy(np.stack([tr[i:i + args.ctx] for i in o])).to(dev)
            y = torch.from_numpy(np.stack([tr[i + 1:i + args.ctx + 1] for i in o])).to(dev)
            with autocast():
                loss = model(x, y)
            n = y.numel()
        else:
            o = np.random.randint(0, len(st_tr) - W, args.batch)
            words = word_batch(tr, st_tr, en_tr, o, W, args.k_max).to(dev)
            with autocast():
                loss = model(words)
            n = int((words >= 0).sum()) + words.shape[0] * words.shape[1]
        opt.zero_grad(set_to_none=True)
        (loss / n).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        train_tokens_seen += y.numel() if args.model == "gpt" else int((words >= 0).sum())
        pace_step(step_started, args.duty_cycle, dev)
        if step % 50 == 0 or step == steps - 1:
            el = time.time() - t0
            remaining = el / (step + 1) * (steps - step - 1)
            print(f"  step {step + 1}/{steps} loss/token {loss.item() / n:.3f}  "
                  f"{el:.0f}s; training ETA {remaining / 60:.1f} min (validation extra)", flush=True)
    train_s = time.time() - t0
    torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "scheduler_state": sched.state_dict(), "config": config,
                "steps": steps, "train_tokens_seen": train_tokens_seen}, run_dir / "model.pt")
    print(f"saved {run_dir / 'model.pt'}; evaluating", flush=True)

    model.eval()
    total = 0.0
    with torch.no_grad(), autocast():
        if args.model == "gpt":
            for i in range(0, len(va) - 1, args.ctx):
                step_started = time.perf_counter()
                x = torch.from_numpy(va[i:i + args.ctx][None]).to(dev)
                y = torch.from_numpy(va[i + 1:i + args.ctx + 1][None]).to(dev)
                x = x[:, :y.shape[1]]
                total += model(x, y).item()
                pace_step(step_started, args.duty_cycle, dev)
        else:
            for o in range(0, len(st_va), W):
                step_started = time.perf_counter()
                w = min(W, len(st_va) - o)
                words = word_batch(va, st_va, en_va, [o], w, args.k_max).to(dev)
                total += model(words).item()
                pace_step(step_started, args.duty_cycle, dev)
    res = {"data": args.data, "model": args.model, "tag": args.tag, "vocab": V,
           "non_emb_params": n_params(model), "params": n_params(model, False),
           "train_tokens_seen": train_tokens_seen, "train_tokens_total": len(tr), "steps": steps,
           "seed": args.seed, "device": dev, "output_dir": str(run_dir),
           "train_s": round(train_s), "valid_bpc": bpc(total, meta["valid_chars"]),
           "valid_chars_per_token": meta["valid_chars"] / meta["valid_tokens"]}
    print(json.dumps(res), flush=True)
    (run_dir / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    results_file = args.results_file or DATA / "results.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
