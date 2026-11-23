"""Sample Turkish text from an exported turkmorph LM (see lm/export_hf.py).

  python lm/generate.py MODEL_DIR "İstanbul'un en eski semtlerinden" [-n 3] [--tokens 120]

The prompt is encoded with turkmorph; the model's ids are decoded back with turkmorph,
which accepts any id sequence. --show-tokens prints the morpheme stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo checkout; pip install turkmorph also works


def load(model_dir, device=None):
    from transformers import GPT2LMHeadModel
    from turkmorph.fast import fast_tokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_dir, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    tok = fast_tokenizer()
    vocab_file = Path(model_dir) / "turkmorph_vocab.json"
    if not vocab_file.exists():                                   # a Hub id, not a local folder
        from huggingface_hub import hf_hub_download
        vocab_file = hf_hub_download(model_dir, "turkmorph_vocab.json")
    info = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
    if list(tok.vocab.toks) != info["tokens"]:
        raise SystemExit("installed turkmorph vocabulary differs from the one the model was trained with")
    return model.to(device).eval(), tok, info["bos_id"]


def generate(model, tok, bos, prompt, n=1, tokens=120, temperature=0.8, top_p=0.95, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    ids = torch.tensor([[bos] + tok.encode_ids(prompt.strip())] * n, device=model.device)
    banned = [[bos]] + [[tok.vocab.id(t)] for t in ("<pad>", "<bos>", "<eos>")]
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=tokens, do_sample=True,
                             temperature=temperature, top_p=top_p, repetition_penalty=1.1,
                             bad_words_ids=banned, pad_token_id=bos)
    return [(tok.decode_ids(o[1:].tolist()), [tok.vocab.tok(i) for i in o[ids.shape[1]:].tolist()]) for o in out]


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
    model, tok, bos = load(args.model_dir)
    for i, (text, toks) in enumerate(generate(model, tok, bos, args.prompt, args.n, args.tokens,
                                               args.temperature, args.top_p, args.seed)):
        print(f"--- {i + 1}\n{text}")
        if args.show_tokens:
            print("   ", " ".join(toks[:40]), "...")


if __name__ == "__main__":
    main()
