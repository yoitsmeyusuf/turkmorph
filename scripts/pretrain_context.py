"""Pretrain the TokenEncoder on BellaTurca with whole-word masking -> data/encoder.pt

Input: data/lm/morph/{train,valid}.bin + groups_*.bin (python3 lm/prepare.py --only morph,composer).
A word = a token group (encode_grouped); 15% of words are masked as a whole (80% [MASK],
10% random, 10% kept) and their tokens predicted from context."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from turkmorph.context import TokenEncoder                       # noqa: E402

D = ROOT / "data" / "lm" / "morph"
CFG = {"d": 384, "layers": 6, "heads": 6, "ctx": 512}
SEQ, BATCH, LR = 256, 64, 4e-4
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0     # time budget


def batch(ids, gstart, wid, rng, V, mask_id):
    o = rng.integers(0, len(ids) - SEQ - 1, BATCH)
    x = np.stack([ids[i:i + SEQ] for i in o]).astype(np.int64)
    w = np.stack([wid[i:i + SEQ] for i in o])
    w = w - w[:, :1]                                             # word index within the window
    pick = rng.random((BATCH, w.max() + 1)) < 0.15
    m = np.take_along_axis(pick, w, 1)
    y = np.where(m, x, -100)
    r = rng.random(x.shape)
    x = np.where(m & (r < 0.8), mask_id, x)
    x = np.where(m & (r >= 0.8) & (r < 0.9), rng.integers(0, V, x.shape), x)
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    meta = json.loads((D / "meta.json").read_text())
    V = meta["vocab_size"]
    tr = np.fromfile(D / "train.bin", dtype=np.uint32)
    va = np.fromfile(D / "valid.bin", dtype=np.uint32)
    wtr = np.cumsum(np.fromfile(D / "groups_train.bin", dtype=np.uint8)).astype(np.int64)
    wva = np.cumsum(np.fromfile(D / "groups_valid.bin", dtype=np.uint8)).astype(np.int64)
    dev = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = TokenEncoder(V, **CFG).to(dev)
    print(f"encoder: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"train {len(tr) / 1e6:.1f}M tokens, budget {MINUTES:.0f} min", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=0.01, fused=True)
    t0, step, est_steps = time.time(), 0, None
    while True:
        el = (time.time() - t0) / 60
        if el >= MINUTES:
            break
        if step == 200:                                          # steps that fit the budget -> schedule
            est_steps = int(200 / el * MINUTES)
        total = est_steps or 10 ** 9
        lr = LR * min(1, (step + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / total, 1))))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(tr, None, wtr, rng, V, model.mask_id)
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(x)
            sel = y != -100
            loss = torch.nn.functional.cross_entropy(model.mlm_logits(h[sel]).float(), y[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vx, vy = batch(va, None, wva, np.random.default_rng(1), V, model.mask_id)
                vx, vy = vx.to(dev), vy.to(dev)
                vh = model(vx)
                vs = vy != -100
                vl = torch.nn.functional.cross_entropy(model.mlm_logits(vh[vs]).float(), vy[vs]).item()
                acc = (model.mlm_logits(vh[vs]).argmax(-1) == vy[vs]).float().mean().item()
            model.train()
            print(f"  step {step}{'/' + str(est_steps) if est_steps else ''}  train {loss.item():.3f}  "
                  f"valid {vl:.3f} (masked-token acc {acc:.1%})  {el:.1f}/{MINUTES:.0f} min", flush=True)
    torch.save({"vocab": V, "config": CFG, "state": model.state_dict()}, ROOT / "data" / "encoder.pt")
    print(f"saved data/encoder.pt after {step} steps ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
