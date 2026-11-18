import hashlib

import numpy as np
import pytest

from lm.prepare_bpe import encode_file, recover_text
from lm.prepare import train_bpe
from turkmorph.presets import demo_tokenizer


def test_recover_reference_preserves_text_across_chunks(tmp_path):
    tok = demo_tokenizer()
    text = "Öğretmenlerimizden 12 34.\n\n  Çocuklar 😀\n" * 10
    source = tmp_path / "source.txt"
    source.write_text(text + "unused remainder", encoding="utf-8")
    # Match prepare.py's per-line encoding, including leading glue after newline.
    ids = [i for line in text.splitlines(keepends=True)
           for i in tok.encode_ids("\n" + line)[1:]]
    binary = tmp_path / "ids.bin"
    np.asarray(ids, dtype=np.uint32).tofile(binary)
    output = tmp_path / "reference.txt"
    result = recover_text(tok, binary, source, output, batch_tokens=50)
    assert output.read_text(encoding="utf-8") == text
    assert result == {"chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    source.write_text("different text", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        recover_text(tok, binary, source, output, batch_tokens=50)


def test_streamed_bpe_encoding_is_lossless(tmp_path):
    pytest.importorskip("tokenizers")
    text = "İstanbul'dan 12 34 😀\n\n öğretmenlerimizden\t\n" * 300
    source = tmp_path / "source.txt"
    source.write_text(text, encoding="utf-8")
    bpe = train_bpe(source, 300)
    output = tmp_path / "ids.bin"
    count = encode_file(bpe, source, output)
    ids = np.fromfile(output, dtype=np.uint32).tolist()
    assert count == len(ids)
    assert bpe.decode(ids) == text
