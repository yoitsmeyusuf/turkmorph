"""Hourglass: word boundaries match the tokenizer, and no position sees its future."""
import numpy as np
import pytest
import torch

from lm.model import Hourglass, word_classes, word_starts
from turkmorph.fast import fast_tokenizer
from turkmorph.tokenizer import group_flags

TEXTS = ["Evlerimizden geldiniz, 1923'te kuruldu.", "Kubernetes ile ÇALIŞIYORUZ; İstanbul'da 3,5 km.",
         "gelmeyecekmişsiniz\nyeni satır ve 'tırnak' (parantez) 2026-11-23."]


@pytest.fixture(scope="module")
def tok():
    return fast_tokenizer()


def test_word_starts_match_group_flags(tok):
    wc = torch.as_tensor(word_classes(tok.vocab.toks + ["<bos>"]))
    for text in TEXTS:
        ids = tok.encode_ids(text)
        x = torch.tensor([[len(tok.vocab)] + ids[:-1]])          # BOS + inputs, as in training
        ws = word_starts(x, wc)[0, 1:].tolist()
        assert ws == [bool(f) for f in group_flags([tok.vocab.toks[i] for i in ids])][:len(ids) - 1]


@pytest.mark.parametrize("pool,in_word", [("sum", False), ("sum", True), ("last_mean", True)])
def test_no_future_leak(tok, pool, in_word):
    torch.manual_seed(0)
    vocab = len(tok.vocab) + 1
    model = Hourglass(vocab, d=64, n_pre=1, n_mid=2, n_post=1, heads=2, ctx=128, d_mid=96, pool=pool,
                      in_word_pos=in_word, wclass=word_classes(tok.vocab.toks + ["<bos>"])).eval()
    torch.nn.init.normal_(model.up.weight, std=0.02)             # make the word path matter
    ids = tok.encode_ids(TEXTS[0] + " " + TEXTS[1])
    x = torch.tensor([[len(tok.vocab)] + ids[:-1]])
    with torch.no_grad():
        base = model.hidden(x)
        for j in range(2, x.shape[1]):
            y = x.clone()
            y[0, j] = ids[0] if x[0, j] != ids[0] else ids[1]       # change token j (and maybe boundaries)
            other = model.hidden(y)
            assert torch.allclose(base[0, :j], other[0, :j], atol=1e-5), f"position < {j} saw token {j}"
        # the word path is actually used: later positions do change
        assert not torch.allclose(base[0, -1], model.hidden(torch.cat([x[:, :1], x[:, 2:], x[:, -1:]], 1))[0, -1])


def test_loss_matches_manual(tok):
    torch.manual_seed(1)
    vocab = len(tok.vocab) + 1
    model = Hourglass(vocab, d=64, n_pre=1, n_mid=1, n_post=1, heads=2, ctx=64,
                      wclass=word_classes(tok.vocab.toks + ["<bos>"]))
    ids = tok.encode_ids(TEXTS[2])
    x = torch.tensor([[len(tok.vocab)] + ids[:-1]])
    y = torch.tensor([ids])
    logits = model.hidden(x) @ model.emb.weight.T
    manual = torch.nn.functional.cross_entropy(logits[0], y[0], reduction="sum")
    assert torch.allclose(model(x, y), manual, atol=1e-4)


def test_padding_does_not_change_real_positions(tok):
    torch.manual_seed(2)
    vocab = len(tok.vocab) + 1
    model = Hourglass(vocab, d=64, n_pre=1, n_mid=2, n_post=1, heads=2, ctx=128,
                      wclass=word_classes(tok.vocab.toks + ["<bos>"])).eval()
    torch.nn.init.normal_(model.up.weight, std=0.02)
    ids = tok.encode_ids(TEXTS[0])
    x = torch.tensor([[len(tok.vocab)] + ids[:-1]])
    y = torch.tensor([ids])
    xp = torch.cat([x, torch.zeros(1, 9, dtype=torch.long)], 1)
    yp = torch.cat([y, torch.full((1, 9), -100)], 1)
    with torch.no_grad():
        assert torch.allclose(model.hidden(x)[0], model.hidden(xp, yp < 0)[0, :x.shape[1]], atol=1e-5)
        assert torch.allclose(model(x, y), model(xp, yp), atol=1e-3)
