"""Export a non-GPT-2 checkpoint (e.g. Hourglass) as a self-contained HF repo folder:
bf16 safetensors (no optimizer state) + model.py + standalone generate.py + vocabulary.

  python lm/export_native.py RUN_DIR OUT_DIR [--verify-data DATA_DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lm"))

from train_equal_text import build_model  # noqa: E402

KEEP = ("arch", "tokenizer", "d", "layers", "ctx", "n_pre", "n_mid", "n_post", "d_mid", "pool", "in_word_pos")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--verify-data", type=Path)
    args = ap.parse_args()
    from safetensors.torch import save_file
    from turkmorph.fast import fast_tokenizer

    cfg = json.loads((args.run_dir / "config.json").read_text())
    meta = cfg["data_meta"]
    sd = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=False)["model_state"]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in sd.items()},
              out / "model.safetensors", metadata={"format": "pt"})
    model_cfg = {k: cfg[k] for k in KEEP if k in cfg} | {
        "vocab_size": meta["vocab_size"], "bos_id": meta["bos_id"], "heads": cfg["d"] // 64}
    (out / "config.json").write_text(json.dumps(model_cfg, indent=2))
    shutil.copy(ROOT / "lm" / "model.py", out / "model.py")
    shutil.copy(ROOT / "lm" / "hourglass_generate.py", out / "generate.py")
    tok = fast_tokenizer()
    (out / "turkmorph_vocab.json").write_text(json.dumps(
        {"tokens": list(tok.vocab.toks), "bos_id": meta["bos_id"],
         "morph_vocab_sha256": meta["tokenizer_sha256"]["morph_vocab"]}, ensure_ascii=False), encoding="utf-8")
    results = {k: json.loads((args.run_dir / f).read_text()) for k, f in
               (("valid", "result.json"), ("test", "test_result.json")) if (args.run_dir / f).exists()}

    if args.verify_data:                      # rescore valid with the exported bf16 weights
        sys.path.insert(0, str(out))
        from generate import load
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load(out, dev)
        data = np.fromfile(args.verify_data / "morph" / "valid.bin", dtype="<u4")
        off = np.load(args.verify_data / "morph" / "valid_offsets.npy")
        chars = int(np.load(args.verify_data / "valid_chars.npy").sum())
        nats = 0.0
        with torch.no_grad():
            for i in range(len(off) - 1):
                seg = torch.tensor(np.concatenate([[meta["bos_id"]], data[off[i]:off[i + 1]]]).astype(np.int64), device=dev)
                nats += model(seg[None, :-1], seg[None, 1:]).item()
        bpc = nats / chars / math.log(2)
        results["valid_rescored_bfloat16"] = {"bpc": bpc, "segments": len(off) - 1, "chars": chars}
        print(f"valid BPC with exported bf16 weights: {bpc:.4f} (training eval {results.get('valid', {}).get('valid_bpc')})")
    (out / "eval_results.json").write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
