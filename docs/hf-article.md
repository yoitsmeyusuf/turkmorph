---
title: "Morphemes, not merges: what a lossless Turkish tokenizer buys a language model, and where it stops"
authors:
  - user: yoitsmeyusuf
---

# Morphemes, not merges: what a lossless Turkish tokenizer buys a language model, and where it stops

**TL;DR**

- I built [turkmorph](https://github.com/yoitsmeyusuf/turkmorph), a Turkish tokenizer that
  emits roots and suffixes instead of BPE pieces, and that decodes back to the exact input
  text.
- In controlled comparisons (same text, same vocabulary size, same model, both tokenizers
  tuned, scored in bits per character), GPT models trained on its tokens beat byte-level BPE:
  - **11%** for a 130M model on 300M characters;
  - **7%** for the same model on 1B characters.
- **The advantage shrinks as data grows.** It is largest where data is scarce.
- **An ablation ladder splits the gain three ways:**
  - **57%** comes from cutting words at morpheme boundaries;
  - **27%** comes from cheap normalisation: casing marks, digits, apostrophes;
  - **16%** comes from storing suffixes as archiphonemes (`-lar/-ler` → one id).
- **A word-level hourglass model built on these tokens lost at scale.**
- **Everything is public:** two 360M models, the code, and all result tables.

| | |
|---|---|
| 🧩 Tokenizer | [github.com/yoitsmeyusuf/turkmorph](https://github.com/yoitsmeyusuf/turkmorph) · `pip install turkmorph` |
| 🤖 Model | [turkmorph-gpt-360m](https://huggingface.co/yoitsmeyusuf/turkmorph-gpt-360m) |
| ⏳ Negative result | [turkmorph-hourglass-360m](https://huggingface.co/yoitsmeyusuf/turkmorph-hourglass-360m) |

An earlier draft of this post claimed "11–15% better" and stopped there. A reviewer pointed
out what that left untested: where the gain comes from, how it behaves with more data, and
what it costs in compute. This version adds those experiments. The answers are less
flattering than the headline, and more useful.

## The problem: BPE doesn't know where Turkish words break

Turkish is agglutinative: a single word can carry a whole clause.
*gidemeyeceklermiş* means "apparently they will not be able to go". BPE learns frequent
character strings, and those often cut across morpheme boundaries. Here is a BPE tokenizer
trained on 1B characters of Turkish, with the same vocabulary size as turkmorph:

| word | BPE (58,667 vocab) | turkmorph |
|---|---|---|
| gidemeyeceklermiş | `gi · dem · eyecek · ler · miş` | `R:git#V S:IMPOT S:FUT S:Z3PL S:ECOP` |
| İstanbul'dakiler | `İstanbul · 'd · ak · iler` | `<cap> R:istanbul#PROPER <apos> S:LOC S:REL S:PL` |
| sokaktan | `s · ok · aktan` | `R:sokak#N S:ABL` |
| kitaplarımızı okuyamadık. | `kitap · larımızı · ␣okuy · amadık · .` | `R:kitap#N S:PL S:P1PL S:ACC R:oku#V S:IMPOT S:PAST S:K1PL P:.` |

Two things stand out:

1. **BPE breaks roots apart.** *git-* ("go", surfacing as *gid-*) becomes `gi · dem`, and
   *sokak* ("street") becomes `s · ok`.
2. **BPE spreads one suffix over many tokens.** Vowel harmony and consonant assimilation give
   the ablative four surface forms: *-den, -dan, -ten, -tan*. In turkmorph they are always one
   token:

```
evden    → R:ev#N     S:ABL        okuldan  → R:okul#N  S:ABL
kitaptan → R:kitap#N  S:ABL        sokaktan → R:sokak#N S:ABL
```

## How turkmorph works

- **Archiphonemic suffixes.** turkmorph stores suffixes as archiphonemic ids, for example
  `S:PL` = *-lAr* and `S:ABL` = *-DAn*. It stores roots with lexical flags: does the final
  consonant soften, does a vowel drop, does the word break vowel harmony.
- **Decoding.** On decode, the surface form is regenerated deterministically from the root's
  flags and the suffix chain.
- **Lossless by construction.** An analysis is accepted only if regenerating it reproduces
  the word, so `decode(encode(text)) == text` is an invariant. It is tested on every UD test
  sentence, on held-out BellaTurca documents, and with random-string fuzzing.
- **Fallbacks.** Words outside the lexicon fall back to a small BPE (`L:`), and from there to
  bytes (`B:`). Casing, apostrophes and spacing are explicit marks.

The vocabulary has 58,667 tokens:

| namespace | count | what it holds |
|---|---:|---|
| `R:` | 39,653 | roots |
| `L:` | 16,143 | BPE fallback pieces |
| `E:` | 2,304 | spelling edits |
| `B:` | 256 | bytes |
| `S:` | 240 | suffixes |
| `P:` | 43 | punctuation |
| `D:` | 10 | digits |
| specials | 18 | casing marks, apostrophe, newline, … |

**Two honest caveats about "one morpheme, one token".**

- **Union ids.** 39 of the suffix ids are unions for label-only ambiguity. In *modellerini*,
  the plural may belong to the noun or to the possessive, so the word ends in a single
  `S:P3PL+ACC|PL+P2SG+ACC|PL+P3SG+ACC`. Plural information therefore sometimes sits in `S:PL`
  and sometimes inside a union.
- **Context-free choices.** Ambiguous analyses are resolved without context: *Gelecek hafta*
  ("next week") comes out as *gel+FUT*.

**Root quality.** On the Universal Dependencies Turkish test sets (49K words), the chosen
root matches the gold lemma 88.8% of the time.

**The price.** turkmorph produces more tokens. On the 1B-character corpus below, the same
text is 344M morph tokens versus 208M BPE tokens, 1.66× as many.

## A fair comparison

Tokenizer comparisons are easy to get wrong. The protocol:

1. **Same text.** The corpus is cut into segments that every tokenizer encodes within the
   context. All models see exactly these segments, in the same order.
2. **Same vocabulary size.** 58,667 for both. The BPE is byte-level and trained on the same
   corpus.
3. **Bits per character (BPC).** All tokens are scored from a shared BOS, and the total is
   divided by the number of characters. The metric does not care how the text was cut.
4. **Same model.** A plain GPT: pre-LN, GELU, learned positions, tied embeddings.
5. **Both sides tuned.** Batches count segments, so BPE sees fewer tokens per step at the
   same batch size. In a sweep on BellaTurca, both tokenizers were best at about 21–25K
   tokens per step, at their own learning rates.
   - Result 1 uses each tokenizer's best setting.
   - Result 2 carries those settings over to a new corpus without re-tuning.
   - The ablation uses one shared setting, the best one for BPE at that size.

## Result 1: a clear gain at small scale

On 300M characters of BellaTurca, BPC valid / test:

| model | turkmorph | BPE | turkmorph better by |
|---|---|---|---|
| 130M, each at its best batch/lr | 1.146 / 1.505 | 1.283 / 1.714 | **10.7% / 12.2%** |
| 130M, same setting, 3 seeds | 1.168 / 1.525 | 1.344 / 1.774 | 13.1% / 14.0% |

The seed spread is 0.002–0.007 BPC.

**An unexplained gap.** Valid and test differ by about 0.35 BPC here, and both come from the
same held-out file. I have not explained this. It may be a non-random split, such as a
change of source document type across the file, or near-duplicates that exact-hash
deduplication missed. It does not change the comparison, because both tokenizers see the
same text, but it is a reason not to over-read absolute values.

## Result 2: the gap shrinks with data

A 130M model at 300M characters is heavily undertrained, and that is exactly where a strong
inductive bias helps most. So I trained the same 130M model on a different corpus:
Wikipedia-tr plus FineWeb-2, with a 2,048-token context and its own test set. The data was
nested: the 300M run sees a prefix of the 1B run's shuffled order.

| training text | turkmorph valid / test | BPE valid / test | turkmorph better by |
|---|---|---|---|
| 300M chars | 1.412 / 1.378 | 1.592 / 1.554 | **11.3% / 11.3%** |
| 1B chars | 1.213 / 1.185 | 1.307 / 1.281 | **7.2% / 7.4%** |

**What this shows.** The same 11% gain appears on a corpus the tokenizer tables were not
built from, and this corpus has no strange valid/test gap. With 3.3× more text, the advantage
falls to 7%. I could not afford 3B characters, so I do not know where the curve flattens or
whether it reaches zero.

**Compute.** Morph's 1.66× more tokens make training about 1.6× slower: 27.5 versus 17.5 min
for the 1B runs on an H100. If I interpolate morph's two runs to BPE's training time, it is
roughly **1% better** at equal wall-clock. That is an estimate across a data curve, not a
measured iso-FLOP run.

An older small-scale check (33M parameters) gave BPE 1.77× more text, so both models
processed the same number of tokens. There, morph still won by 15.8%. Taken together: at
small scale the gain holds at equal compute, but at 130M with 1B characters, most of the gap
is closed by giving BPE the compute morph spends on its extra tokens.

## Result 3: where the gain comes from

turkmorph does several things at once. Byte-level BPE does none of them, so it is not
obvious which ones matter. I built two intermediate tokenizers and trained the same 33M
model on identical segments, with 3 seeds each:

| step | tokenizer | valid BPC (mean of 3) | gain | share of total |
|---|---|---:|---:|---:|
| 0 | byte-level BPE | 1.952 | | |
| 1 | + casing marks, one token per digit, isolated apostrophe, then BPE (same vocab size) | 1.898 | 0.053 | **27%** |
| 2 | + morpheme split, but suffixes keep their surface form (`S:PL@lar` ≠ `S:PL@ler`) | 1.788 | 0.111 | **57%** |
| 3 | + archiphonemic suffix ids: turkmorph | 1.757 | 0.030 | **16%** |

- **Consistency.** Every step improves on every seed. The seed spread for turkmorph is
  0.001 BPC.
- **Morpheme boundaries do most of the work.** Cutting at morpheme boundaries is the
  largest step, well beyond what normalisation alone gives.
- **Archiphonemes help, but less.** Merging allomorphs into one id is a real but smaller
  effect, about a sixth of the total.
- **Scale.** This is a small model on 30M characters. Given Result 2, the shares at larger
  scale may differ.

## Side quest: an hourglass that lost

turkmorph knows exactly where each word starts and ends. So I tried to win back the 1.66×
sequence cost with an **hourglass** transformer: 4 token blocks, then 16 blocks that run
once per *word*, then 4 token blocks. The word stack runs on about 0.51 positions per token.

Three changes were specific to turkmorph:

- the tokenizer's own word groups;
- an in-word morpheme position embedding;
- last-token plus mean pooling.

With these changes, the hourglass beat the flat model by 0.4% at 33M parameters. At 360M
parameters, with 1B characters and a 2,048-token context, it lost:

| model | train time (H100) | valid BPC | test BPC |
|---|---:|---:|---:|
| flat GPT, 24 layers | 54.4 min | **1.185** | **1.158** |
| hourglass 4 + 16 + 4 | 43.9 min | 1.219 (+2.9%) | 1.191 (+2.9%) |

The hourglass was 19% faster and 2.9% worse. This is one seed, and the hourglass did not get
its saved time back as extra steps. I publish it because "word-level compression is free" is
a tempting conclusion that this experiment does not support.

## What I take from this

1. **Morpheme boundaries are worth more than the clever encoding.** For Turkish, cutting
   words where the morphology cuts them gives most of the gain. Storing suffixes as
   archiphonemes adds a smaller part on top.
2. **The benefit is a data-efficiency effect.** It is large when text is scarce and shrinks
   as text grows. That still matters for Turkish: many domains and many small labs live in
   the low-data regime. But the effect has not been shown for pretraining at scale.
3. **Cost goes up.** Longer sequences mean about 1.6× more training compute and a context
   window that holds about 0.6× as much text.

## Try it

```bash
pip install turkmorph transformers torch
```

```python
import torch
from transformers import GPT2LMHeadModel
from turkmorph.fast import fast_tokenizer

model = GPT2LMHeadModel.from_pretrained("yoitsmeyusuf/turkmorph-gpt-360m").eval()
tok, BOS = fast_tokenizer(), 58667
ids = torch.tensor([[BOS] + tok.encode_ids("Bu tarifte kullanacağınız malzemeler şunlardır:")])
out = model.generate(ids, max_new_tokens=100, do_sample=True, top_p=0.95, temperature=0.8,
                     bad_words_ids=[[BOS], [0], [1], [2]], pad_token_id=BOS)
print(tok.decode_ids(out[0, 1:].tolist()))
```

The models were trained with turkmorph 0.2. Version 0.3 keeps the same vocabulary, but
picks canonical root ids, so a few words tokenize differently. Pin `turkmorph==0.2.*` to
reproduce the models exactly.

## Limitations

- **Only BPC was measured.** I did not run downstream tasks such as classification, QA or
  NER.
- **Statistics.** Larger runs use one seed. Three seeds exist only for the 130M BellaTurca
  setting and the ablation.
- **No iso-FLOP curve.** The equal-compute statements above are a token-matched small run and
  an interpolation.
- **Tokenizer gaps.**
  - Foreign words can get spurious analyses: `fine` → *fi+P2SG+DAT*.
  - Acronyms get Turkish lowercasing: `COVID` → *covıd*.
  - A word with a missing suffix falls back to BPE as a whole.

  These are listed in the package README.

## Credits

- **Lexicon:** built from [Zemberek](https://github.com/ahmetaa/zemberek-nlp).
- **Training data:**
  - [Turkish Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (CC BY-SA 4.0)
  - [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `tur_Latn` (ODC-By 1.0)
  - BellaTurca
- **Evaluation:** the Universal Dependencies Turkish treebanks.
- **Compute:** large runs used Modal's free credits; ablations ran on an RTX 4060.
- **Review:** thanks to the reviewer whose critique produced Results 2 and 3.

Every result table is in the repository under `results/`, and every experiment script is
under `lm/` and `scripts/`.
