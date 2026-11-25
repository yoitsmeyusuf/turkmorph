# turkmorph

Lossless Turkish morphological tokenizer with a Rust runtime and a Python API.
Words are split into **roots and suffixes**. Suffixes are stored as archiphonemic ids, so
`-ler/-lar` are both `S:PL` and `-den/-dan/-ten/-tan` are all `S:ABL`. Decoding regenerates
the exact surface text: `decode(encode(text)) == text` holds for any input.

> **License.** The code is Apache-2.0. The runtime tables shipped inside the wheel
> (`turkmorph/data/`) are **CC BY-SA 4.0**, because they are derived from CC BY-SA resources
> (UD Turkish treebanks, BellaTurca). If you redistribute the tables, or modified versions of
> them, the attribution and share-alike terms apply. No non-commercial resource is used.
> See `NOTICE` in the source distribution for the full list of sources.

## Install

```bash
pip install turkmorph
```

Wheels are built for Linux (x86_64, aarch64), macOS (Intel, Apple Silicon) and Windows (x64),
for Python ≥ 3.10. Other platforms build from source, which needs a Rust toolchain.

## Usage

```python
from turkmorph.fast import fast_tokenizer

tok = fast_tokenizer()
tok.encode("Öğretmenlerimizden NATO'ya gidemeyeceklermiş.")
# ['<cap>', 'R:öğretmen#N', 'S:PL', 'S:P1PL', 'S:ABL', '<capsb>', 'R:nato#PROPER', '<apos>',
#  'S:DAT', 'R:git#V', 'S:IMPOT', 'S:FUT', 'S:Z3PL', 'S:ECOP', 'P:.']

ids = tok.encode_ids(text)              # integer ids for a model (vocabulary: 58,667)
assert tok.decode_ids(ids) == text      # lossless
tok.encode_batch(texts, threads=6)      # multi-threaded, releases the GIL
```

`decode_ids` accepts any id sequence, which makes it safe to use on model samples. A
pure-Python implementation that produces the same tokens is available as
`turkmorph.presets.full_tokenizer()`.

## Vocabulary

| namespace | count | content |
|---|---:|---|
| `R:` | 39,653 | roots, with part of speech (`R:ev#N`, `R:git#V`) |
| `L:` | 16,143 | BPE pieces for words without a known root |
| `E:` | 2,304 | spelling edits |
| `S:` | 240 | suffixes; 39 of them are union ids for ambiguous chains (`S:P3SG\|ACC`) |
| `B:` / `P:` / `D:` | 256 / 43 / 10 | bytes, punctuation, digits |
| specials | 18 | casing (`<cap>`, `<caps>`), apostrophe, whitespace, … |

## Changes in 0.3.0

- **Canonical roots.** One id per lemma where the surface does not decide: `koyun` and
  `koyunlar` now use `R:koyun#N`, and the homograph variant `R:koyun#N.1` appears only where
  it is needed (`koynu`). Tagged root tokens drop by about 80% on running text.
- **`-CAsInA` after person endings.** `yapmışsınızcasına` and `gelmişlercesine` now analyse
  fully.
- **Longer chains.** The maximum suffix chain grew from 10 to 12, so
  `Çekoslovakyalılaştıramadıklarımızdanmışsınızcasına` now analyses fully.
- **Same vocabulary.** The 58,667 tokens are unchanged, so models trained with 0.2.x still
  load. Some words tokenize differently; for exact reproduction of those models, pin
  `turkmorph==0.2.*`.

## Known limitations

- **Ambiguity.** Ambiguous analyses are resolved without context
  (`Gelecek hafta` → `gel+FUT`).
- **Unknown roots.** Words without a known root fall back to BPE pieces. When no suffix
  chain fits, a word is split into BPE pieces as a whole: a known prefix is not kept.
- **Foreign words.** Foreign words can get spurious Turkish analyses (`fine` → `fi+P2SG+DAT`).
- **Acronyms.** Foreign acronyms get Turkish lowercasing (`COVID` → `covıd`). Fixing this
  needs a new case mark, which is planned for a vocabulary-changing release.

## Models and experiments

- [turkmorph-gpt-360m](https://huggingface.co/yoitsmeyusuf/turkmorph-gpt-360m): a Turkish GPT
  trained on turkmorph tokens. It needs the turkmorph 0.2.x vocabulary.
- Language-model comparisons with BPE, evaluation scripts and design notes are in the
  [repository](https://github.com/yoitsmeyusuf/turkmorph).
