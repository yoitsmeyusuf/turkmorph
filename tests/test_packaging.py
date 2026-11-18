"""The wheel ships exactly the tables the runtimes read; the data directory resolves correctly."""
from pathlib import Path
import re

from turkmorph import _paths

ROOT = Path(__file__).resolve().parent.parent


def test_checkout_uses_repository_data():
    assert _paths.default_data_dir() == ROOT / "data"


def test_runtime_files_exist():
    missing = [name for name in _paths.RUNTIME_FILES if not (ROOT / "data" / name).exists()]
    assert not missing


def test_runtime_files_cover_rust_loader():
    source = (ROOT / "rust" / "src" / "tokenizer.rs").read_text(encoding="utf-8")
    read_by_rust = set(re.findall(r'"([a-z_]+\.(?:json|tsv|txt))"', source))
    assert read_by_rust <= set(_paths.RUNTIME_FILES)


def test_runtime_files_are_tracked_by_git():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for name in _paths.RUNTIME_FILES:
        assert f"!data/{name}" in ignore, name
