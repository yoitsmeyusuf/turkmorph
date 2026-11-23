# Flat GPT vs Hourglass on turkmorph tokens — 360M, long context (2026-11-23)

Corpus: 1B chars = 300M Wikipedia-tr (CC BY-SA 4.0) + 700M FineWeb-2 tur_Latn (ODC-By 1.0),
documents shuffled (`scripts/build_hf_corpus.py`). Valid: 2M chars from held-out documents.
Test: FineWeb-2 tur_Latn official test shard (~2M chars). Segments <=5000 chars / <=2048 tokens,
ctx 2048, morph tokens (vocab 58,667): 344M train tokens. One pass, batch 16 segments (~26K tokens),
peak lr 8e-4, one seed, H100 (Modal). Hourglass: 4 token blocks + 16 word blocks + 4 token blocks,
d 1024 everywhere, in-word position + last-token/mean pooling (best variant of
`results/hourglass-ablation-30m.md`).

| Model | Non-emb params | Block FLOPs / token | Train time | Valid BPC | Test BPC |
|---|---:|---:|---:|---:|---:|
| Flat GPT, 24 blocks | 304M | 1.00 | 54.4 min | **1.1850** | **1.1577** |
| Hourglass 4+16+4 | 309M | 0.67 | 43.9 min | 1.2192 (+2.9%) | 1.1910 (+2.9%) |

Word groups average 0.507 per morph token, so the word stack runs on about half the positions.
At equal parameters the hourglass is 19% faster in wall-clock but 2.9% worse in BPC. The small-scale
gain (−0.4% at 33M/30M chars) did not carry over: compressing each word into one position costs
quality that a 360M model with long context does not recover. Not tested: hourglass at equal
wall-clock/FLOPs (more steps or a deeper word stack), several seeds.
