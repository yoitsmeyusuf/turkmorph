"""Copy the runtime tables from data/ into turkmorph/data/ so the wheel ships them.

Run before building a wheel or sdist; scripts/build_wheel.sh removes the copy afterwards.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turkmorph._paths import RUNTIME_FILES


def main():
    src, dst = ROOT / "data", ROOT / "turkmorph" / "data"
    missing = [name for name in RUNTIME_FILES if not (src / name).exists()]
    if missing:
        raise SystemExit(f"Missing runtime tables in data/: {', '.join(missing)}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    for name in RUNTIME_FILES:
        shutil.copy2(src / name, dst / name)
    total = sum((dst / name).stat().st_size for name in RUNTIME_FILES)
    print(f"Staged {len(RUNTIME_FILES)} tables ({total / 1e6:.1f} MB) in {dst}")


if __name__ == "__main__":
    main()
