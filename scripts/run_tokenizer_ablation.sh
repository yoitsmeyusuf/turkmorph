#!/usr/bin/env bash
# Local (4060) tokenizer ablation ladder, S size (d384 L6), 30M chars of BellaTurca, 3 seeds:
#   bpe -> bpe_norm (casing marks, digits, apostrophe) -> morph_sfc (morpheme split, surface allomorphs) -> morph
# Data: python lm/prepare_ablation.py --output-dir data/lm/ablation-30m
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/lm/experiments/tokenizer-ablation-30m
mkdir -p "$OUT"
COMMON="--data-root data/lm/ablation-30m --batch 16 --accum 2 --lr 1e-3 --duty-cycle 0.85"
t0=$(date +%s)
total=12; n=0
for seed in 0 1 2; do
  for tok in bpe bpe_norm morph_sfc morph; do
    n=$((n + 1)); name=$tok-s$seed
    [ -f "$OUT/$name/result.json" ] && continue
    while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -gt 70 ]; do sleep 20; done
    python3 -u lm/train_equal_text.py $COMMON --tokenizer $tok --seed $seed --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
    el=$(( $(date +%s) - t0 ))
    echo "[$(date +%H:%M)] ($n/$total, ${el}s, ETA ~$(( el * (total - n) / n / 60 )) min) $name $(python3 -c "import json;print(round(json.load(open('$OUT/$name/result.json'))['valid_bpc'],4))")"
  done
done
