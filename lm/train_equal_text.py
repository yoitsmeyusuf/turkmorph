"""Train a GPT on every shared text segment once, in a seed-paired order."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Batches have many different lengths; the default allocator keeps a block per size and runs out of 8 GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lm.model import GPT, Hourglass, bpc, n_params
from lm.train import learning_rate_scale, pace_step


def load_split(root, name, split):
    ids = np.memmap(root / name / f"{split}.bin", mode="r", dtype="<u4")
    offsets = np.load(root / name / f"{split}_offsets.npy")
    chars = np.load(root / f"{split}_chars.npy")
    if (len(offsets) != len(chars) + 1 or offsets[0] != 0 or offsets[-1] != len(ids)
            or np.any(np.diff(offsets) <= 0) or np.any(chars <= 0)):
        raise ValueError(f"Invalid segment boundaries: {name}/{split}")
    return ids, offsets, chars


def build_model(config, meta, data_root):
    """GPT or Hourglass from a run config (dict); used by training and evaluation."""
    if config.get("arch", "gpt") == "gpt":
        return GPT(meta["vocab_size"], config["d"], config["layers"], config["d"] // 64, config["ctx"])
    wclass_path = Path(data_root) / "morph" / "word_classes.npy"
    wclass = np.zeros(meta["vocab_size"], dtype=np.uint8)
    if wclass_path.exists():
        loaded = np.load(wclass_path)
        wclass[:len(loaded)] = loaded
    elif config["tokenizer"] != "morph":
        raise ValueError("Hourglass word groups are only defined for morph tokens")
    return Hourglass(meta["vocab_size"], config["d"], config["n_pre"], config["n_mid"], config["n_post"],
                     config["d"] // 64, config["ctx"], config.get("d_mid") or None, wclass,
                     config.get("pool", "last_mean"), config.get("in_word_pos", True))


def text_batch(split, indices, bos):
    ids, offsets, chars = split
    lengths = offsets[indices + 1] - offsets[indices]
    x = np.zeros((len(indices), int(lengths.max())), dtype=np.int64)
    y = np.full_like(x, -100)
    for row, index in enumerate(indices):
        tokens = ids[offsets[index]:offsets[index + 1]]
        x[row, 0] = bos
        x[row, 1:len(tokens)] = tokens[:-1]
        y[row, :len(tokens)] = tokens
    return torch.from_numpy(x), torch.from_numpy(y), int(chars[indices].sum()), int(lengths.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--tokenizer", choices=["morph", "bpe"], required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--duty-cycle", type=float, default=.7)
    ap.add_argument("--arch", choices=["gpt", "hourglass"], default="gpt")
    ap.add_argument("--n-pre", type=int, default=4, help="hourglass: token blocks before the word stack")
    ap.add_argument("--n-mid", type=int, default=16, help="hourglass: word-level blocks")
    ap.add_argument("--n-post", type=int, default=4, help="hourglass: token blocks after the word stack")
    ap.add_argument("--d-mid", type=int, default=0, help="hourglass: word-stack width (0 = d)")
    ap.add_argument("--pool", choices=["sum", "last_mean"], default="last_mean")
    ap.add_argument("--no-in-word-pos", dest="in_word_pos", action="store_false")
    ap.add_argument("--accum", type=int, default=1,
                    help="split each batch into this many micro-batches (same update, less memory)")
    ap.add_argument("--train-chars", type=int, default=0,
                    help="train on the shortest prefix of the seed order holding >= this many characters "
                         "(same segments for every tokenizer); 0 = all")
    ap.add_argument("--benchmark-steps", type=int, default=0,
                    help="time this many training steps, print throughput, exit without saving")
    args = ap.parse_args()
    if (args.batch < 1 or args.accum < 1 or args.d < 64 or args.d % 64 or args.layers < 1
            or args.ctx < 1 or not 0 < args.duty_cycle <= 1):
        ap.error("Invalid model, batch or duty-cycle settings")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is unavailable here; use the GPU-enabled WSL terminal")
    if (args.output_dir / "model.pt").exists() and not args.benchmark_steps:
        ap.error("Refusing to overwrite an existing checkpoint")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    meta = json.loads((args.data_root / "meta.json").read_text())
    train = load_split(args.data_root, args.tokenizer, "train")
    valid = load_split(args.data_root, args.tokenizer, "valid")
    if any(int(np.diff(split[1]).max()) > args.ctx for split in (train, valid)):
        ap.error("A segment exceeds model context; truncation is forbidden")
    for name, split in (("train", train), ("valid", valid)):
        if int(split[2].sum()) != meta["splits"][name]["chars"]:
            raise ValueError(f"Character count mismatch: {name}")
        if int(split[0].max()) >= meta["base_vocab_size"]:
            raise ValueError(f"Token id outside vocabulary: {name}")
    order = np.random.default_rng(args.seed).permutation(len(train[2]))
    if args.train_chars:
        cut = int(np.searchsorted(np.cumsum(train[2][order]), args.train_chars)) + 1
        order = order[:cut]
    planned_chars = int(train[2][order].sum())
    order_hash = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
    steps = (len(order) + args.batch - 1) // args.batch
    if args.arch == "hourglass" and (args.tokenizer != "morph" or not (args.data_root / "morph" / "word_classes.npy").exists()):
        ap.error("Hourglass needs morph data with morph/word_classes.npy (lm/add_word_classes.py)")
    model = build_model(vars(args), meta, args.data_root).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.1,
                            fused=args.device == "cuda")
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: learning_rate_scale(s, steps))
    amp = args.device == "cuda" and torch.cuda.is_bf16_supported()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(data_meta=meta, train_order_sha256=order_hash,
                  torch=str(torch.__version__), precision="bfloat16" if amp else "float32",
                  device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
                  loss_normalization="summed token NLL / batch characters",
                  evaluation="identical text segments; all tokens scored from shared BOS; right padding ignored")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"{args.tokenizer}, seed {args.seed}: {steps} steps, {planned_chars:,} of "
          f"{meta['splits']['train']['chars']:,} characters; output {args.output_dir}", flush=True)
    seen_chars = seen_tokens = 0
    started = time.monotonic()
    model.train()
    for step, start in enumerate(range(0, len(order), args.batch)):
        tick = time.perf_counter()
        batch_indices = order[start:start + args.batch]
        chars = int(train[2][batch_indices].sum())          # normaliser for the whole batch
        tokens, loss = 0, 0.
        opt.zero_grad(set_to_none=True)
        for micro in np.array_split(batch_indices, min(args.accum, len(batch_indices))):
            x, y, _, micro_tokens = text_batch(train, micro, meta["bos_id"])
            x, y = x.to(args.device), y.to(args.device)
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=amp):
                micro_loss = model(x, y)
            (micro_loss / chars).backward()
            loss += micro_loss.detach()
            tokens += micro_tokens
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        schedule.step()
        seen_chars += chars
        seen_tokens += tokens
        pace_step(tick, args.duty_cycle, args.device)
        if args.benchmark_steps and step + 1 == args.benchmark_steps:
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            print(json.dumps({"benchmark_steps": step + 1, "seconds": round(elapsed, 2),
                              "chars_per_s": round(seen_chars / elapsed), "tokens_per_s": round(seen_tokens / elapsed),
                              "full_run_estimate_min": round(elapsed / (step + 1) * steps / 60, 1),
                              "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                              if args.device == "cuda" else None}), flush=True)
            return
        if step % 50 == 0 or step + 1 == steps:
            elapsed = time.monotonic() - started
            eta = elapsed / (step + 1) * (steps - step - 1) / 60
            print(f"step {step+1}/{steps} nats/char {loss.item()/chars:.4f}; "
                  f"seen {seen_chars:,} chars; training ETA {eta:.1f} min (validation extra)", flush=True)
    train_s = time.monotonic() - started
    if seen_chars != planned_chars:
        raise ValueError("Training did not cover exactly the planned text")
    torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "scheduler_state": schedule.state_dict(), "config": config,
                "train_chars_seen": seen_chars, "train_tokens_seen": seen_tokens}, args.output_dir / "model.pt")
    print("Saved checkpoint; evaluating identical validation segments", flush=True)
    model.eval()
    total = 0.
    valid_chars = valid_tokens = 0
    with torch.no_grad():
        for start in range(0, len(valid[2]), args.batch):
            tick = time.perf_counter()
            indices = np.arange(start, min(start + args.batch, len(valid[2])))
            x, y, chars, tokens = text_batch(valid, indices, meta["bos_id"])
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=amp):
                total += model(x.to(args.device), y.to(args.device)).item()
            valid_chars += chars
            valid_tokens += tokens
            pace_step(tick, args.duty_cycle, args.device)
    if valid_chars != meta["splits"]["valid"]["chars"]:
        raise ValueError("Validation text coverage mismatch")
    result = {"data": args.tokenizer, "seed": args.seed, "model": "gpt", "params": n_params(model, False),
              "non_emb_params": n_params(model), "steps": steps, "train_s": round(train_s, 2),
              "train_chars_seen": seen_chars, "train_tokens_seen": seen_tokens,
              "train_order_sha256": order_hash, "train_text_sha256": meta["splits"]["train"]["sha256"],
              "valid_chars": valid_chars, "valid_tokens_scored": valid_tokens,
              "valid_text_sha256": meta["splits"]["valid"]["sha256"],
              "valid_bpc": bpc(total, valid_chars), "output_dir": str(args.output_dir)}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
