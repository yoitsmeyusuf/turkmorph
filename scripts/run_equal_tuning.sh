#!/usr/bin/env bash
# Control: is the gap an artifact of one shared learning rate / batch? 30M chars, seed 0.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=data/lm/experiments/equal-text-30m-tuning
cool() { while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -ge 70 ]; do sleep 20; done; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
for spec in "bpe 0.0005 8" "bpe 0.002 8" "bpe 0.001 16" "bpe 0.002 16" "morph 0.0005 8" "morph 0.002 8"; do
  set -- $spec
  out=$EXP/$1-lr$2-b$3
  [ -f "$out/result.json" ] && continue
  cool; stamp "starting $1 lr $2 batch $3"
  mkdir -p "$out"
  python3 -u lm/train_equal_text.py --data-root data/lm/equal-text-30m --tokenizer $1 --seed 0 \
    --lr $2 --batch $3 --output-dir "$out" > "$out/train.log" 2>&1
  stamp "done: $(grep -o '"valid_bpc": [0-9.]*' "$out/result.json")"
done
cool; python3 -u lm/eval_equal_text.py --experiment-dir $EXP
stamp "all done"
