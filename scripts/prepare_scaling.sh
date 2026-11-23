#!/usr/bin/env bash
# Data for the scaling experiment: vocab-matched BPE + 300M-char paired segments + held-out test.
set -euo pipefail
cd "$(dirname "$0")/.."
t0=$(date +%s)
python3 -u lm/prepare_bpe_v2.py --train-chars 300e6 --output-dir data/lm/bpe-v2
echo "[$(( $(date +%s) - t0 ))s] bpe-v2 done"
python3 -u lm/prepare_equal_text.py --bpe-dir data/lm/bpe-v2 --train-chars 300000000 \
  --output-dir data/lm/equal-text-300m
echo "[$(( $(date +%s) - t0 ))s] equal-text-300m done"
python3 -u lm/prepare_test_split.py --bpe-dir data/lm/bpe-v2 --matched-root data/lm/equal-text-300m \
  --output-dir data/lm/equal-text-300m-test
echo "[$(( $(date +%s) - t0 ))s] all done"
