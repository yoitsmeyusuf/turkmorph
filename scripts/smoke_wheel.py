"""Check an installed turkmorph outside the checkout: tables found, lossless, Python == Rust.

    python -I scripts/smoke_wheel.py     (run from a directory without the source tree)
"""
import turkmorph
from turkmorph.fast import DATA, fast_tokenizer
from turkmorph.presets import full_tokenizer

TEXT = ("Öğretmenlerimizden NATO'ya zeytinyağları\n"
        "Güzel bir gün, değil mi? 1.250,50 TL ödedik. İSTANBUL'DA ılık bir akşam.")

tok = fast_tokenizer()
ids = tok.encode_ids(TEXT)
assert tok.decode_ids(ids) == TEXT, "Rust runtime is not lossless"
assert full_tokenizer().encode(TEXT) == tok.encode(TEXT), "Python and Rust runtimes differ"
print(f"ok: {turkmorph.__file__}, tables {DATA}, vocab {len(tok.vocab)}, {len(ids)} tokens")
