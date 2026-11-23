"""Sample from training checkpoints (flat GPT or Hourglass) without converting them.

  python lm/sample_native.py RUN_DIR [RUN_DIR ...] --prompt "..." [--prompt ...] [-n 2] [--tokens 120]

Each run gets the same prompts and seeds, so outputs are side by side. No KV cache: the
whole prefix is recomputed every step, which the hourglass needs anyway (a finished word
only reaches the word-level stack once its last morpheme is known).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lm"))

from train_equal_text import build_model  # noqa: E402


def load(run_dir, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    meta = cfg["data_meta"]
    model = build_model(cfg, meta, run_dir / "_no_data")          # word classes come from the state dict
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)["model_state"])
    return model.to(device).eval(), meta["bos_id"], cfg.get("arch", "gpt")


def next_logits(model, idx):
    if not hasattr(model, "hidden"):          # flat GPT
        x = model.emb(idx) + model.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in model.blocks:
            x = b(x)
        h = model.ln(x[:, -1])
    else:
        h = model.hidden(idx)[:, -1]
    return (h @ model.emb.weight.T).float()


@torch.no_grad()
def sample(model, ids, tokens, banned, temperature=0.8, top_p=0.95, rep=1.1):
    for _ in range(tokens):
        with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.device.type == "cuda"):
            logits = next_logits(model, ids)
        seen = torch.gather(logits, 1, ids)                       # repetition penalty (HF rule)
        logits.scatter_(1, ids, torch.where(seen > 0, seen / rep, seen * rep))
        logits[:, banned] = -float("inf")
        probs = torch.softmax(logits / temperature, -1)
        p, order = probs.sort(-1, descending=True)
        p[(p.cumsum(-1) - p) > top_p] = 0
        nxt = order.gather(1, torch.multinomial(p, 1))
        ids = torch.cat([ids, nxt], 1)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-tokens", action="store_true")
    args = ap.parse_args()
    from turkmorph.fast import fast_tokenizer
    tok = fast_tokenizer()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in args.runs:
        model, bos, arch = load(run, dev)
        banned = [bos] + [tok.vocab.id(t) for t in ("<pad>", "<bos>", "<eos>")]
        print(f"===== {run.name} ({arch})", flush=True)
        for pi, prompt in enumerate(args.prompt):
            torch.manual_seed(args.seed + pi)
            ids = torch.tensor([[bos] + tok.encode_ids(prompt.strip())] * args.n, device=dev)
            t0 = time.perf_counter()
            out = sample(model, ids, args.tokens, banned)
            dt = time.perf_counter() - t0
            for i, o in enumerate(out):
                print(f"--- [{pi + 1}.{i + 1}] {tok.decode_ids(o[1:].tolist())}")
                if args.show_tokens:
                    print("   ", " ".join(tok.vocab.tok(j) for j in o[ids.shape[1]:ids.shape[1] + 30].tolist()), "...")
            print(f"    ({args.n}x{args.tokens} tokens in {dt:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
