---
title: "Morphemes, not merges: a lossless Turkish tokenizer and what it does to language models"
thumbnail: ""
authors:
  - user: yoitsmeyusuf
---

# Morphemes, not merges: a lossless Turkish tokenizer and what it does to language models

**TL;DR**

- I built [turkmorph](https://github.com/yoitsmeyusuf/turkmorph), a Turkish tokenizer that
  emits roots and suffixes instead of BPE pieces, and that decodes back to the exact input
  text.
- In a controlled comparison, GPT models trained on its tokens are 11–15% better than models
  trained on BPE. The comparison holds text, vocabulary size and architecture fixed, tunes
  both tokenizers, and scores by bits per character.
- A word-level "hourglass" architecture built on top of it lost at scale.
- Two 360M models and all code are public.

| | |
|---|---|
| 🧩 Tokenizer | [github.com/yoitsmeyusuf/turkmorph](https://github.com/yoitsmeyusuf/turkmorph) (`pip install turkmorph`) |
| 🤖 Model | [turkmorph-gpt-360m](https://huggingface.co/yoitsmeyusuf/turkmorph-gpt-360m) |
| ⏳ Negative result | [turkmorph-hourglass-360m](https://huggingface.co/yoitsmeyusuf/turkmorph-hourglass-360m) |

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

1. **BPE splits roots apart.** BPE breaks the root *git-* ("go"), which surfaces as *gid-*,
   into `gi · dem`, and breaks *sokak* ("street") into `s · ok`. The model has to learn
   every time that these fragments belong together.
2. **BPE spreads one suffix over many tokens.** Vowel harmony and consonant assimilation give
   the ablative ("from") four surface forms: *-den, -dan, -ten, -tan*. To BPE these are
   different tokens, and they often merge with neighbouring material. In turkmorph they are
   always one token, `S:ABL`:

```
evden    → R:ev#N     S:ABL        okuldan  → R:okul#N  S:ABL
kitaptan → R:kitap#N  S:ABL        sokaktan → R:sokak#N S:ABL
```

## How turkmorph works

- **Archiphonemic suffixes.** turkmorph stores suffixes as archiphonemic ids, for example
  `S:PL` = *-lAr* and `S:ABL` = *-DAn*. It stores roots with their lexical flags: does the
  final consonant soften, does a vowel drop, does the word break vowel harmony.
- **Decoding.** On decode, the surface form is regenerated deterministically from the root's
  flags and the suffix chain.
- **Lossless by construction.** An analysis is accepted only if regenerating it gives back
  the original word. That makes `decode(encode(text)) == text` an invariant. It is checked
  on every UD test sentence, on held-out BellaTurca documents, and with random-string
  fuzzing.
- **Fallbacks.** Words outside the lexicon fall back to a small BPE over leftover word pieces
  (`L:`), and from there to bytes (`B:`). Casing, apostrophes and spacing are explicit marks.

The 58,667-token vocabulary:

| namespace | count | what it holds |
|---|---:|---|
| `R:` | 39,653 | roots |
| `L:` | 16,143 | BPE fallback pieces |
| `E:` | 2,304 | edits |
| `B:` | 256 | bytes |
| `S:` | 240 | suffixes |
| `P:` | 43 | punctuation |
| `D:` | 10 | digits |
| specials | 18 | casing marks, apostrophe, newline, … |

**Root quality.** On the Universal Dependencies Turkish test sets (49K words), the chosen
root matches the gold lemma 88.8% of the time. Counting derivational suffixes it is 94.0%;
the oracle ceilings are 90.6% and 96.0%. Ambiguity costs were learned with EM on 118M
tokens of BellaTurca. The runtime is Rust with a Python API.

**The price.** turkmorph produces more tokens: 2.2 per word, against 1.3 for BPE on the
same vocabulary budget. On the 1B-character corpus below that is 344M tokens versus 208M,
or 1.66× as many. Every LM result below has to pay for that.

## A fair comparison

Tokenizer comparisons are easy to get wrong. Measure loss per token and the tokenizer that
makes more, easier tokens wins for free. Train on "the same number of tokens" and one model
has seen much more text. The protocol here:

1. **Same text.** The corpus is cut into segments that both tokenizers encode within the
   length limit. Both models see exactly these segments in exactly the same order.
2. **Same vocabulary size.** 58,667 for both. The BPE is a byte-level BPE trained on the same
   training text.
3. **Bits per character.** All tokens are scored from a shared BOS, and the total is divided
   by the number of characters. The metric does not care how the text was cut.
4. **Same model.** A plain GPT: pre-LN, GELU, learned positions, tied embeddings.
5. **Tuning both sides.** I swept batch size and learning rate for each tokenizer. Because
   batches count segments, BPE sees fewer tokens per step at the same batch size, so it needs
   a larger batch. BPE was best at batch 128 and morph at batch 64, both about 21–25K tokens
   per step. The headline number is best against best.
6. **Held-out test text** that was never used for any choice.

Training ran on single H100s on [Modal](https://modal.com).

## Results

**Scaling: 300M characters of BellaTurca.** BPC, lower is better, valid / test:

| model | setting | turkmorph | BPE | turkmorph better by |
|---|---|---|---|---|
| 130M (d768, 12 layers), 3 seeds | same batch/lr | 1.168 / 1.525 | 1.344 / 1.774 | 13.1% / 14.0% |
| 130M, **each at its best batch/lr** | morph b64 · BPE b128 | **1.146 / 1.505** | **1.283 / 1.714** | **10.7% / 12.2%** |
| 360M (d1024, 24 layers) | same batch/lr | 1.153 / 1.516 | 1.359 / 1.779 | 15.2% / 14.8% |

- **Seeds.** The spread across seeds is 0.002–0.007 BPC, so the gap is far outside noise.
- **Tuning.** The gap shrinks when BPE gets its best settings. The honest figure is the
  best-against-best one: **about 11% on valid and 12% on test**. Higher numbers from untuned
  comparisons overstate it.
- **Model size.** At the same settings the gap does not close as the model grows: at batch 16,
  it is 15.2% at 130M and 15.7% at 360M on valid.

**What about compute?** turkmorph makes 1.7–1.8× more tokens, so a morph model does more
work per character. In a smaller study (33M parameters) I gave BPE 1.77× more text, so both
models processed the same number of tokens. turkmorph still won by **15.8%** (2.015 vs 2.395
BPC). The advantage is not just extra compute.

### Why it helps

Look again at the BPE column of the first table. With morphemes as tokens, the model does
not have to:

- learn that `gi · dem`, `gid` and `git` are one verb;
- learn that `dan`, `tan`, `den` and `ten` are one suffix;
- learn which fragments reassemble into which words.

The tokenizer has already done that. A suffix such as `S:ABL` collects every occurrence of
the ablative in one token, so its embedding is learned from all of them instead of being
split across many BPE pieces. Roots are
whole units, so the model can spend its capacity on syntax and meaning.

## The hourglass experiment, and why it lost

If morph tokens cost 1.66× more positions, can we win that back? turkmorph knows exactly
where each word starts and ends, so I tried an **hourglass** transformer:

```
morph tokens ─► 4 token blocks ─► pool each word ─► 16 word blocks ─► shift by one word ─► 4 token blocks ─► next token
```

The middle 16 layers run once per word instead of once per token, which is about 0.51
positions per token. Three changes were specific to turkmorph:

- **Word boundaries.** Word groups come from the tokenizer's own grouping rule: root plus its
  suffixes, apostrophe and casing marks.
- **Order inside a word.** An in-word position embedding (root = 0, first suffix = 1, …),
  because sum pooling cannot tell `S:PL S:P1PL` from `S:P1PL S:PL`.
- **Pooling.** The word state is the last token state concatenated with the mean token state.

**At small scale it looked promising.** With 33M parameters and 30M characters:

| variant | valid BPC |
|---|---:|
| flat GPT | 1.7566 |
| hourglass, sum pooling | 1.7590 |
| + in-word position | 1.7506 |
| + last+mean pooling | **1.7501** (0.4% better than flat) |

**At scale it lost.** Both models have 360M parameters, a 2,048-token context and the same
1B characters: Turkish Wikipedia plus FineWeb-2.

| model | train time (H100) | valid BPC | test BPC |
|---|---:|---:|---:|
| flat GPT, 24 layers | 54.4 min | **1.185** | **1.158** |
| hourglass 4 + 16 + 4 | 43.9 min | 1.219 (+2.9%) | 1.191 (+2.9%) |

The hourglass is 19% faster, and 2.9% worse.

**Samples.** I generated from both models with the same prompts and seeds. Both handle
morphology equally well: *İstanbul'daki*, *kaydedilmiştir*, *ev sahipliği yapmaktadır*. The
hourglass drifts off topic sooner. My reading is that a single vector per word throws away
information a larger model could have used. The small model could not use it anyway, so the
small model hid the loss.

**Caveats.** This is one seed, and the hourglass was not given its saved 19% back as extra
steps, so an equal-compute run could narrow the gap. I am publishing it as a negative result,
because "word-level compression is free" is a tempting conclusion that this experiment does
not support.

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

A real sample:

> **Bu tarifte kullanacağınız malzemeler şunlardır:**
> - Yarım bardak su
> - Bir tatlı kaşığı biber salçası
> - Hazırlanışı için
> - Tuz, karabiber
> …
> Tencereye koyduktan sonra, daha fazla su ilave ederek kaynamaya bırakın.

The 360M model is a research base model. It writes fluent, correctly inflected Turkish and
invents facts freely.

## Limitations

- **Only language-modelling quality was measured.** I did not test downstream tasks such as
  classification, QA or NER.
- **Small models.** 360M parameters at most, and at most 1B training characters.
- **More tokens.** About 1.7× as many tokens per character means longer sequences at
  inference. The equal-compute result says this is worth paying at small scale; at large
  scale it is untested.
- **Lexicon coverage.** turkmorph analyses 96–99% of words with a known root. The rest go
  through the BPE or byte fallback, which is still lossless but less meaningful.

## Credits

- **Lexicon:** built from [Zemberek](https://github.com/ahmetaa/zemberek-nlp).
- **Training data:** [Turkish Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
  (CC BY-SA 4.0), [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
  `tur_Latn` (ODC-By 1.0), and BellaTurca for the tokenizer statistics and the scaling study.
- **Evaluation:** the Universal Dependencies Turkish treebanks.
- **Compute:** the scaling and hourglass runs used Modal's free credits.

The full experiment logs, scripts and result tables are in the repository under `lm/` and
`results/`. Feedback and issues are welcome!
