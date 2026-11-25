# Data axis on the HF corpus: morph vs BPE at 130M parameters

M model (d768, 12 layers), ctx 2048, Wikipedia-tr + FineWeb-2. The 300M run trains on a 300M-character prefix of the same
seed order as the 1B run (`--train-chars`), so the data is nested. Test = FineWeb-2 test shard (not BellaTurca).
Settings from the BellaTurca sweep at matched tokens/step (~23–26K): morph b16 lr1.5e-3, BPE b24 lr2e-3. One seed, H100.

| train chars | tokenizer | valid BPC | test BPC | train tokens | train min |
|---|---|---:|---:|---:|---:|
| 300M | morph | 1.4124 | 1.3780 | 103.3M | 8.6 |
| 300M | bpe | 1.5922 | 1.5540 | 62.3M | 5.4 |
| 1B | morph | 1.2129 | 1.1854 | 344.2M | 27.5 |
| 1B | bpe | 1.3068 | 1.2806 | 207.6M | 17.5 |

| train chars | morph better by (valid / test) |
|---|---|
| 300M | 11.3% / 11.3% |
| 1B | 7.2% / 7.4% |

**The gap shrinks with data**: 11.3% at 300M characters, 7.2% at 1B (3.3× more text). The morph advantage is at least
partly a low-data effect; this experiment cannot say where it ends (3B characters was outside the budget).

**Rough equal-time view**: BPE at 1B took 17.5 min. Interpolating morph's two points in log-time gives about
1.291 valid BPC at that time vs BPE's 1.3068 (1.2% better). This is an interpolation across
a data curve, not a measured iso-FLOP run.

Code: `lm/modal_run.py` (datasets `hf-300m-long`, `hf-1b-long`), `lm/modal_launch.py`.
