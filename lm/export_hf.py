"""Export a flat GPT checkpoint (lm/model.py) to a Hugging Face GPT2LMHeadModel.

The architectures match exactly once the missing biases are zero: pre-LN blocks,
exact GELU, learned positions, tied output embedding, eps 1e-5. The text side stays
turkmorph (morphological, lossless); the HF folder carries its vocabulary.

  python lm/export_hf.py RUN_DIR OUT_DIR [--dtype bfloat16] [--verify-data DATA_DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lm"))

from model import GPT  # noqa: E402


def to_hf(sd, layers):
    t = lambda w: w.t().contiguous()  # noqa: E731  (Linear (out,in) -> Conv1D (in,out))
    out = {"transformer.wte.weight": sd["emb.weight"], "transformer.wpe.weight": sd["pos.weight"],
           "transformer.ln_f.weight": sd["ln.weight"], "transformer.ln_f.bias": sd["ln.bias"]}
    for i in range(layers):
        p, h = f"blocks.{i}.", f"transformer.h.{i}."
        for a, b in (("ln1", "ln_1"), ("ln2", "ln_2")):
            out[h + b + ".weight"], out[h + b + ".bias"] = sd[p + a + ".weight"], sd[p + a + ".bias"]
        for name, src in (("attn.c_attn", "qkv"), ("attn.c_proj", "proj"), ("mlp.c_fc", "mlp.0"), ("mlp.c_proj", "mlp.2")):
            w = t(sd[p + src + ".weight"])
            out[h + name + ".weight"], out[h + name + ".bias"] = w, torch.zeros(w.shape[1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--verify-data", type=Path, help="equal-text data dir: rescore its valid split")
    args = ap.parse_args()
    from transformers import GPT2Config, GPT2LMHeadModel, GenerationConfig
    from turkmorph.fast import fast_tokenizer

    cfg = json.loads((args.run_dir / "config.json").read_text())
    meta = cfg["data_meta"]
    if cfg["arch"] != "gpt":
        sys.exit("only the flat GPT architecture maps onto GPT-2")
    sd = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=False)["model_state"]
    vocab, d, layers, ctx = meta["vocab_size"], cfg["d"], cfg["layers"], cfg["ctx"]
    hf_cfg = GPT2Config(vocab_size=vocab, n_positions=ctx, n_embd=d, n_layer=layers, n_head=d // 64,
                        activation_function="gelu", layer_norm_epsilon=1e-5, resid_pdrop=0, embd_pdrop=0,
                        attn_pdrop=0, bos_token_id=meta["bos_id"], eos_token_id=None, pad_token_id=None,
                        tie_word_embeddings=True, architectures=["GPT2LMHeadModel"])
    model = GPT2LMHeadModel(hf_cfg)
    missing, unexpected = model.load_state_dict(to_hf(sd, layers), strict=False)
    missing = [k for k in missing if k != "lm_head.weight" and not k.endswith(".attn.bias")]
    if missing or unexpected:
        sys.exit(f"state dict mismatch: missing {missing} unexpected {unexpected}")
    model.tie_weights()
    model.eval()

    # 1) logits identical to the training model (fp32)
    tok = fast_tokenizer()
    ref = GPT(vocab, d, layers, d // 64, ctx)
    ref.load_state_dict(sd)
    ref.eval()
    ids = torch.tensor([[meta["bos_id"]] + tok.encode_ids("Ankara'dan İzmir'e giden otobüsler bu sabah gecikmeli kalktı.")])
    with torch.no_grad():
        x = ref.emb(ids) + ref.pos(torch.arange(ids.shape[1]))
        for b in ref.blocks:
            x = b(x)
        a = ref.ln(x) @ ref.emb.weight.T
        bl = model(ids).logits
    diff = (a - bl).abs().max().item()
    print(f"max |logit diff| fp32: {diff:.2e}", flush=True)
    if diff > 1e-3:
        sys.exit("logits differ")
    del ref

    model = model.to(getattr(torch, args.dtype))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    GenerationConfig(bos_token_id=meta["bos_id"], do_sample=True, temperature=0.8, top_p=0.95,
                     max_new_tokens=200, repetition_penalty=1.1).save_pretrained(args.out_dir)
    toks = list(tok.vocab.toks)
    (args.out_dir / "turkmorph_vocab.json").write_text(json.dumps(
        {"tokens": toks, "bos_id": meta["bos_id"], "morph_vocab_sha256": meta["tokenizer_sha256"]["morph_vocab"]},
        ensure_ascii=False), encoding="utf-8")
    results = {k: json.loads((args.run_dir / f).read_text()) for k, f in
               (("valid", "result.json"), ("test", "test_result.json")) if (args.run_dir / f).exists()}

    # 2) rescore the valid split with the exported weights
    if args.verify_data:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = GPT2LMHeadModel.from_pretrained(args.out_dir, dtype=getattr(torch, args.dtype)).to(dev).eval()
        data = np.fromfile(args.verify_data / "morph" / "valid.bin", dtype="<u4")
        off = np.load(args.verify_data / "morph" / "valid_offsets.npy")
        chars = np.load(args.verify_data / "valid_chars.npy")
        nats = 0.0
        with torch.no_grad():
            for i in range(len(off) - 1):
                seg = torch.tensor(np.concatenate([[meta["bos_id"]], data[off[i]:off[i + 1]]]).astype(np.int64), device=dev)
                logits = m(seg[None, :-1]).logits[0].float()
                nats += torch.nn.functional.cross_entropy(logits, seg[1:], reduction="sum").item()
        bpc = nats / int(chars.sum()) / math.log(2)
        results["valid_rescored_" + args.dtype] = {"bpc": bpc, "segments": len(off) - 1, "chars": int(chars.sum())}
        print(f"valid BPC with exported {args.dtype} weights: {bpc:.4f} "
              f"(training eval {results.get('valid', {}).get('valid_bpc', 'n/a')})", flush=True)
    (args.out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
