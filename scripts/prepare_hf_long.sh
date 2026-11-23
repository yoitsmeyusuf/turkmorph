#!/usr/bin/env bash
# HF corpus (Wikipedia-tr + FineWeb-2) -> long-context paired segments for the hourglass experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
t0=$(date +%s)
C=data/hf/corpus
[ -d data/lm/bpe-hf ] || python3 -u lm/prepare_bpe_v2.py --train-source $C/train.txt --train-chars 1e9 \
  --valid-reference $C/valid.txt --output-dir data/lm/bpe-hf
echo "[$(( $(date +%s) - t0 ))s] bpe-hf done"
[ -d data/lm/equal-text-hf-1b-long ] || python3 -u lm/prepare_equal_text.py --bpe-dir data/lm/bpe-hf \
  --train-chars 1000000000 --max-tokens 2048 --chunk-chars 5000 --output-dir data/lm/equal-text-hf-1b-long
echo "[$(( $(date +%s) - t0 ))s] equal-text done"
# the test text starts right after the validation text; skip its first line so the split
# point is a newline strictly after the validation range (prepare_test_split's disjointness check)
TEST_CHARS=$(python3 -c "t=open('$C/test.txt',encoding='utf-8').read(); print(len(t) - t.index(chr(10)) - 1)")
[ -d data/lm/equal-text-hf-test-long ] || python3 -u lm/prepare_test_split.py --bpe-dir data/lm/bpe-hf \
  --matched-root data/lm/equal-text-hf-1b-long --source $C/valid_test.txt --test-chars "$TEST_CHARS" \
  --output-dir data/lm/equal-text-hf-test-long
python3 lm/add_word_classes.py data/lm/equal-text-hf-1b-long data/lm/equal-text-hf-test-long
echo "[$(( $(date +%s) - t0 ))s] all done"
