#!/usr/bin/env bash
# Run the first GPU pilot; pass extra train.py arguments to override defaults.
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_PYTHON=${TRAIN_PYTHON:-python3}
TRAIN_DATA=${TRAIN_DATA:-morph}
"$TRAIN_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA bu terminalde erişilebilir değil. GPU'nun göründüğü WSL terminalinden çalıştırın.")
print("GPU:", torch.cuda.get_device_name(0), "| PyTorch:", torch.__version__, flush=True)
x = torch.randn(64, 64, device="cuda", requires_grad=True)
x.square().mean().backward()
torch.cuda.synchronize()
print("CUDA forward/backward kontrolü geçti.", flush=True)
PY
TRAIN_RUN_DIR="data/lm/runs/${TRAIN_DATA}-gpt-pilot-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$TRAIN_RUN_DIR"
"$TRAIN_PYTHON" -u lm/train.py --data "$TRAIN_DATA" --model gpt --device cuda \
    --d 384 --layers 6 --ctx 512 --batch 8 --seed 0 --max-tokens 1000000 \
    --duty-cycle 0.7 --tag pilot --output-dir "$TRAIN_RUN_DIR" "$@" 2>&1 | tee "$TRAIN_RUN_DIR/train.log"
