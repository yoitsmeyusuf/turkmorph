#!/usr/bin/env bash
# After the equal-text run: BPE with the same token budget as morph (more text), then held-out test scores.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=data/lm/experiments
cool() { while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -ge 70 ]; do sleep 20; done; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
while pgrep -f "lm/equal_text_experiment.py" >/dev/null; do sleep 20; done
stamp "equal-text run finished"
while [ ! -f data/lm/equal-text-53m/meta.json ]; do
  pgrep -f "lm/prepare_equal_text.py" >/dev/null || { stamp "53M preparation failed"; exit 1; }
  sleep 20
done
for seed in 0 1 2; do
  out=$EXP/equal-compute-bpe-53m/bpe-seed$seed
  [ -f "$out/result.json" ] && continue
  cool; stamp "starting BPE 53M seed $seed"
  mkdir -p "$out"
  python3 -u lm/train_equal_text.py --data-root data/lm/equal-text-53m --tokenizer bpe --seed $seed \
    --output-dir "$out" > "$out/train.log" 2>&1
  stamp "done: $(grep -o '"valid_bpc": [0-9.]*' "$out/result.json")"
done
cool; stamp "held-out test evaluation"
python3 -u lm/eval_equal_text.py --experiment-dir $EXP/equal-text-30m-3seeds-r2
python3 -u lm/eval_equal_text.py --experiment-dir $EXP/equal-compute-bpe-53m
stamp "all done"
