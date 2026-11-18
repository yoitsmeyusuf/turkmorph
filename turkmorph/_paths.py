"""Where the runtime tables live."""
from __future__ import annotations

from pathlib import Path

_PKG = Path(__file__).resolve().parent

# Files the tokenizer reads at runtime (Python full_tokenizer and the Rust runtime).
RUNTIME_FILES = (
    "attested_stems.txt", "costs.json", "langid.json", "lex_bpe.json", "lexicon.tsv",
    "morph_tables.json", "overrides.tsv", "proper_words.txt", "tokenizer_tables.json",
    "unions.txt", "wordfreq.tsv",
)


def default_data_dir() -> Path:
    """The repository's data/ in a source checkout (the build pipeline writes there),
    otherwise the copy shipped inside the installed package."""
    checkout = _PKG.parent / "data"
    if (_PKG.parent / "rust" / "Cargo.toml").exists() and (checkout / "lexicon.tsv").exists():
        return checkout
    return _PKG / "data"
