# Tokenizer ablation ladder (33M parameters, 30M characters)

S model (d384, 6 layers, ctx 512), BellaTurca, identical segments and order for all four tokenizers,
batch 16 / lr 1e-3 for all (BPE's best small-scale setting), 3 seeds, RTX 4060. Valid BPC.

| step | tokenizer | vocab | train tokens | seeds 0 / 1 / 2 | mean | gain vs previous | share of total |
|---|---|---:|---:|---|---:|---:|---:|
| 0 byte-level BPE | `bpe` | 58,667 | 5.49M | 1.9425 / 1.9523 / 1.9603 | 1.9517 |  |  |
| 1 + casing marks, one token per digit, isolated apostrophe (then BPE) | `bpe_norm` | 58,667 | 6.56M | 1.8964 / 1.8978 / 1.9008 | 1.8983 | 0.0534 | 27% |
| 2 + morpheme split (turkmorph analyses, suffix ids split by surface allomorph: S:PL@lar / S:PL@ler) | `morph_sfc` | 59,372 | 9.46M | 1.7847 / 1.7879 / 1.7899 | 1.7875 | 0.1109 | 57% |
| 3 + archiphonemic suffix ids (turkmorph) | `morph` | 58,667 | 9.46M | 1.7573 / 1.7579 / 1.7567 | 1.7573 | 0.0302 | 16% |

Total BPE → morph: 0.1944 BPC (10.0%). Seed spread per tokenizer: bpe 0.0178, bpe_norm 0.0044, morph_sfc 0.0052, morph 0.0012. Every step improves on every seed.

Reading: most of the gain comes from splitting at morpheme boundaries; the cheap normalisations explain about a quarter;
archiphonemic suffix ids add a smaller but consistent part. morph_sfc has 705 extra suffix-variant ids (1.2% larger vocab).
Limits: small model, little data (the HF data-axis run shows the morph lead shrinking with data); no iso-FLOP control
(bpe/bpe_norm see fewer tokens per character).

Code: `lm/ablation_tokenizers.py`, `lm/prepare_ablation.py`, `scripts/run_tokenizer_ablation.sh`.
