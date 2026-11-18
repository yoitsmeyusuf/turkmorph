#!/usr/bin/env bash
# Build the Rust runtime and drop it into the package as turkmorph/_fast*.so
#     bash scripts/build_fast.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
python3 -m turkmorph.export
python3 -m maturin build --release -o build/wheels
whl=$(ls -t build/wheels/turkmorph-*.whl | head -1)
python3 - "$whl" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for n in z.namelist():
    if n.endswith(".so") and n.rsplit("/", 1)[-1].startswith("_fast"):
        out = "turkmorph/" + n.rsplit("/", 1)[-1]
        open(out, "wb").write(z.read(n)); print("installed", out)
PY
