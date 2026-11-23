"""Sample Turkish text from the turkmorph hourglass LM (standalone; needs torch,
safetensors, turkmorph).

  python generate.py MODEL_DIR_OR_HUB_ID "İstanbul'un en eski semtlerinden" [-n 2] [--tokens 120]

No KV cache: the prefix is recomputed every step, because a word only reaches the
word-level stack once its last morpheme is known.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _files(model_dir):
    d = Path(model_dir)
    if not (d / "config.json").exists():                          # a Hub id
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(model_dir))
    return d


def load(model_dir, device=None):
    from safetensors.torch import load_file
    d = _files(model_dir)
    sys.path.insert(0, str(d))
    from model import GPT, Hourglass
    cfg = json.loads((d / "config.json").read_text())
    if cfg["arch"] == "hourglass":
        model = Hourglass(cfg["vocab_size"], cfg["d"], cfg["n_pre"], cfg["n_mid"], cfg["n_post"], cfg["heads"],
                          cfg["ctx"], cfg.get("d_mid") or None, None, cfg["pool"], cfg["in_word_pos"])
    else:
        model = GPT(cfg["vocab_size"], cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(load_file(d / "model.safetensors"))      # includes the word-class table
    model = model.to(device, torch.bfloat16 if device == "cuda" else torch.float32).eval()
    return model, cfg


def next_logits(model, idx):
    if hasattr(model, "hidden"):
        h = model.hidden(idx)[:, -1]
    else:
        x = model.emb(idx) + model.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in model.blocks:
            x = b(x)
        h = model.ln(x[:, -1])
    return (h @ model.emb.weight.T).float()


@torch.no_grad()
def sample(model, ids, tokens, banned, temperature=0.8, top_p=0.95, rep=1.1):
    for _ in range(tokens):
        logits = next_logits(model, ids[:, -model.pos.num_embeddings:])
        seen = torch.gather(logits, 1, ids)
        logits.scatter_(1, ids, torch.where(seen > 0, seen / rep, seen * rep))
        logits[:, banned] = -float("inf")
        p, order = torch.softmax(logits / temperature, -1).sort(-1, descending=True)
        p[(p.cumsum(-1) - p) > top_p] = 0
        ids = torch.cat([ids, order.gather(1, torch.multinomial(p, 1))], 1)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--show-tokens", action="store_true")
    args = ap.parse_args()
    from turkmorph.fast import fast_tokenizer
    model, cfg = load(args.model_dir)
    tok = fast_tokenizer()
    vocab = json.loads((_files(args.model_dir) / "turkmorph_vocab.json").read_text(encoding="utf-8"))
    if list(tok.vocab.toks) != vocab["tokens"]:
        raise SystemExit("installed turkmorph vocabulary differs from the one the model was trained with")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    bos = cfg["bos_id"]
    dev = next(model.parameters()).device
    ids = torch.tensor([[bos] + tok.encode_ids(args.prompt.strip())] * args.n, device=dev)
    banned = [bos] + [tok.vocab.id(t) for t in ("<pad>", "<bos>", "<eos>")]
    out = sample(model, ids, args.tokens, banned, args.temperature, args.top_p)
    for i, o in enumerate(out):
        print(f"--- {i + 1}\n{tok.decode_ids(o[1:].tolist())}")
        if args.show_tokens:
            print("   ", " ".join(tok.vocab.tok(j) for j in o[ids.shape[1]:ids.shape[1] + 40].tolist()), "...")


if __name__ == "__main__":
    main()
