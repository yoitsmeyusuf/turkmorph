#!/usr/bin/env bash
# Build the distributable wheel + sdist (Rust runtime and runtime tables included) into dist/
#     bash scripts/build_wheel.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
trap 'rm -rf turkmorph/data' EXIT
python3 scripts/stage_package_data.py
python3 -m maturin build --release --out dist
python3 -m maturin sdist --out dist
ls -l dist
