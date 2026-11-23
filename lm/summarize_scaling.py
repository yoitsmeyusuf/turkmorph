"""Summarise data/lm/experiments/scaling-300m into summary.md / summary.json."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "lm" / "experiments" / (sys.argv[1] if len(sys.argv) > 1 else "scaling-300m")
PARAMS = {"S": "33M", "M": "130M", "L": "360M"}


def main():
    runs = {}
    for d in sorted(EXP.iterdir()):
        if not (d / "test_result.json").exists():
            continue
        head, seed = d.name.rsplit("-s", 1)
        size, tok, batch, lr = head.split("-", 3)
        t = json.loads((d / "test_result.json").read_text())
        r = json.loads((d / "result.json").read_text())
        runs[d.name] = {"size": size, "tokenizer": tok, "setting": f"{batch}-{lr}", "seed": int(seed),
                        "valid_bpc": t["valid_bpc"], "test_bpc": t["test_bpc"],
                        "train_tokens": t["train_tokens_seen"], "train_min": round(r["train_s"] / 60, 1)}
    groups = {}
    for v in runs.values():
        groups.setdefault((v["size"], v["setting"]), {}).setdefault(v["tokenizer"], []).append(v)
    lines = ["# Scaling: morph vs BPE, equal text (300M chars), vocab 58,667, H100 (Modal)", "",
             "| Model | Setting | Seeds | Morph valid / test | BPE valid / test | Morph better (valid / test) |",
             "| --- | --- | ---: | --- | --- | --- |"]
    table = []
    for (size, setting), by in sorted(groups.items(), key=lambda kv: ("SML".index(kv[0][0]), int(kv[0][1].split("-")[0][1:]), kv[0][1])):
        if set(by) != {"morph", "bpe"}:
            continue
        row = {"size": size, "setting": setting, "seeds": min(len(by["morph"]), len(by["bpe"]))}
        cells = []
        for tok in ("morph", "bpe"):
            for split in ("valid", "test"):
                xs = [v[f"{split}_bpc"] for v in by[tok]]
                row[f"{tok}_{split}"] = statistics.mean(xs)
                row[f"{tok}_{split}_sd"] = statistics.stdev(xs) if len(xs) > 1 else None
            sd = lambda s: f" ±{row[f'{tok}_{s}_sd']:.4f}" if row[f"{tok}_{s}_sd"] else ""   # noqa: E731
            cells.append(f"{row[f'{tok}_valid']:.4f}{sd('valid')} / {row[f'{tok}_test']:.4f}{sd('test')}")
        for split in ("valid", "test"):
            row[f"gain_{split}"] = 1 - row[f"morph_{split}"] / row[f"bpe_{split}"]
        table.append(row)
        lines.append(f"| {size} ({PARAMS[size]}) | {setting} | {row['seeds']} | {cells[0]} | {cells[1]} | "
                     f"{row['gain_valid']:.1%} / {row['gain_test']:.1%} |")
    lines += ["", "BPC = bits per character on identical text segments (lower is better). Setting = batch "
              "(segments of <=1024 chars) and peak learning rate. Morph: 95.7M train tokens, BPE: 57.4M.",
              "", "## Runs", "", "| Run | Valid | Test | Train tokens | Train min |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, v in runs.items():
        lines.append(f"| {name} | {v['valid_bpc']:.4f} | {v['test_bpc']:.4f} | {v['train_tokens']:,} | {v['train_min']} |")
    (EXP / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EXP / "summary.json").write_text(json.dumps({"groups": table, "runs": runs}, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
