#!/usr/bin/env bash
# Long-context control: ~2800-character segments (up to 1024 tokens), 30M chars, model context 1024.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=data/lm/equal-text-30m-long
EXP=data/lm/experiments/equal-text-30m-long
cool() { while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -ge 70 ]; do sleep 20; done; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
[ -f $DATA/meta.json ] || python3 -u lm/prepare_equal_text.py --train-chars 30000000 --max-tokens 1024 --chunk-chars 2800 --output-dir $DATA
[ -f data/lm/equal-text-test-2m-long/meta.json ] || python3 -u lm/prepare_test_split.py --matched-root $DATA --output-dir data/lm/equal-text-test-2m-long
python3 - <<'PY'
import numpy as np
for n in ("morph", "bpe"):
    l = np.diff(np.load(f"data/lm/equal-text-30m-long/{n}/train_offsets.npy"))
    print(n, "segment tokens: mean", round(float(l.mean()), 1), "max", int(l.max()), flush=True)
PY
cool; stamp "starting 3-seed long-context comparison (batch 4)"
python3 -u lm/equal_text_experiment.py --data-root $DATA --ctx 1024 --batch 4 --output-dir $EXP/main 2>&1 | grep -v "^step"
out=$EXP/tuned/bpe-b8-seed0
if [ ! -f "$out/result.json" ]; then
  cool; stamp "starting BPE batch 8 control"
  mkdir -p "$out"
  python3 -u lm/train_equal_text.py --data-root $DATA --tokenizer bpe --seed 0 --ctx 1024 --batch 8 \
    --output-dir "$out" > "$out/train.log" 2>&1
  stamp "done: $(grep -o '"valid_bpc": [0-9.]*' "$out/result.json")"
fi
cool
python3 -u lm/eval_equal_text.py --experiment-dir $EXP/main --test-root data/lm/equal-text-test-2m-long --batch 4
python3 -u lm/eval_equal_text.py --experiment-dir $EXP/tuned --test-root data/lm/equal-text-test-2m-long --batch 4
stamp "all done"
