"""Write morph/word_classes.npy (hourglass word-group bits per token id) into a prepared data dir."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lm.model import word_classes
from turkmorph.fast import fast_tokenizer

for root in map(Path, sys.argv[1:]):
    toks = fast_tokenizer().vocab.toks
    out = root / "morph" / "word_classes.npy"
    np.save(out, word_classes(toks))
    print(out, len(toks))
