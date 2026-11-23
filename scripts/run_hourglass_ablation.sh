#!/usr/bin/env bash
# Local (4060) ablation of hourglass variants vs flat GPT, S size, 30M chars, new vocab.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/lm/experiments/hourglass-ablation-30m
COMMON="--data-root data/lm/equal-text-30m-v2 --tokenizer morph --batch 16 --accum 2 --lr 1e-3 --duty-cycle 0.85"
HG="--arch hourglass --n-pre 1 --n-mid 4 --n-post 1"
declare -A V=(
  [A-flat]="--arch gpt"
  [B-hg-sum]="$HG --pool sum --no-in-word-pos"
  [C-hg-sum-inword]="$HG --pool sum"
  [D-hg-lastmean-inword]="$HG --pool last_mean"
  [E-hg-lastmean-inword-wide]="$HG --pool last_mean --d-mid 512"
)
t0=$(date +%s)
for name in A-flat B-hg-sum C-hg-sum-inword D-hg-lastmean-inword E-hg-lastmean-inword-wide; do
  [ -f "$OUT/$name/result.json" ] && continue
  while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -gt 70 ]; do sleep 20; done
  python3 -u lm/train_equal_text.py $COMMON ${V[$name]} --output-dir "$OUT/$name" > "$OUT.$name.log" 2>&1
  echo "[$(( $(date +%s) - t0 ))s] $name $(python3 -c "import json;print(round(json.load(open('$OUT/$name/result.json'))['valid_bpc'],4))")"
done
