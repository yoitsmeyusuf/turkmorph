# Hourglass ablation (local RTX 4060, 2026-11-23)

33M-class models (d384), morph tokens, 30M chars of BellaTurca (`data/lm/equal-text-30m-v2`, vocab 58,667),
batch 16 segments, lr 1e-3, one seed. Hourglass = 1 token block + 4 word blocks + 1 token block.
Script: `scripts/run_hourglass_ablation.sh`.

| Variant | Valid BPC | vs flat |
|---|---:|---:|
| A flat GPT, 6 blocks | 1.7566 | — |
| B hourglass, sum pooling (MorphComp S3 design) | 1.7590 | +0.14% |
| C + in-word (morpheme slot) position | 1.7506 | −0.34% |
| D + last-token & mean pooling | 1.7501 | −0.37% |
| E + wider word stack (d_mid 512) | 1.7499 | −0.38% |

Word groups average 0.507 per morph token, so the word stack runs on ~half the positions.
Single seed: differences of this size need confirmation at scale (see `hf-1b-long`).
