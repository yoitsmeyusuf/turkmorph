#!/usr/bin/env bash
# Data-scaling check: does the morph advantage persist with the full 100M-character training text?
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=data/lm/experiments/equal-text-100m
cool() { while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -ge 70 ]; do sleep 20; done; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
[ -f data/lm/equal-text-100m/meta.json ] || python3 -u lm/prepare_equal_text.py --train-chars 100000000 --output-dir data/lm/equal-text-100m
for name in morph bpe; do
  out=$EXP/$name-seed0
  [ -f "$out/result.json" ] && continue
  cool; stamp "starting $name 100M seed 0"
  mkdir -p "$out"
  python3 -u lm/train_equal_text.py --data-root data/lm/equal-text-100m --tokenizer $name --seed 0 \
    --output-dir "$out" > "$out/train.log" 2>&1
  stamp "done: $(grep -o '"valid_bpc": [0-9.]*' "$out/result.json")"
done
cool; python3 -u lm/eval_equal_text.py --experiment-dir $EXP
stamp "all done"
