"""Small LMs for the tokenizer comparison.

GPT       flat token-level decoder (used with every tokenizer).
Composer  word-level model for TurkMorph (design doc, solution 2): a word encoder
          pools the morpheme embeddings of each word group, a causal transformer
          runs over words, and a GRU decoder emits the next word's morphemes plus
          an end-of-word symbol. Its loss covers every token, so BPC is comparable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))
        self.heads, self.dropout = heads, dropout

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, dropout_p=self.dropout if self.training else 0)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=384, layers=6, heads=6, ctx=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.apply(_init)

    def forward(self, idx, targets):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.ln(x) @ self.emb.weight.T                     # tied output
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.view(-1),
                               reduction="sum")


class Composer(nn.Module):
    """words: (B, W, K) token ids of each word group, padded with -1 (K = max tokens/word)."""

    def __init__(self, vocab, d=384, layers=6, heads=6, ctx_words=256, k_max=12, dec_layers=2):
        super().__init__()
        self.vocab, self.eow = vocab, vocab                          # extra output: end of word
        self.emb = nn.Embedding(vocab + 2, d)                        # +eow, +bos-of-word
        self.inpos = nn.Embedding(k_max, d)
        self.enc = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Embedding(ctx_words + 1, d)
        self.start = nn.Parameter(torch.zeros(d))                   # context before the first word
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.dec = nn.GRU(d, d, num_layers=dec_layers, batch_first=True)
        self.h0 = nn.Linear(d, dec_layers * d)
        self.out_ln = nn.LayerNorm(d)
        self.k_max, self.dec_layers = k_max, dec_layers
        self.apply(_init)

    def encode_words(self, words):
        mask = (words >= 0).unsqueeze(-1)
        e = self.emb(words.clamp(min=0)) + self.inpos.weight[: words.shape[-1]]
        pooled = (e * mask).sum(2) / mask.sum(2).clamp(min=1)
        return self.enc(pooled)

    def forward(self, words, targets=None):
        b, w, k = words.shape
        x = self.encode_words(words)                                 # (B, W, d)
        x = torch.cat([self.start.expand(b, 1, -1), x[:, :-1]], 1)   # predict word t from words < t
        x = x + self.pos(torch.arange(w, device=words.device))
        for blk in self.blocks:
            x = blk(x)
        ctx = self.ln(x).reshape(b * w, -1)                          # (BW, d)
        tgt = words.reshape(b * w, k)
        lens = (tgt >= 0).sum(1)
        # decoder input: [bos, t0, t1, ...], output: [t0, t1, ..., eow]
        bos = torch.full((b * w, 1), self.vocab + 1, device=words.device)
        dec_in = torch.cat([bos, tgt.clamp(min=0)], 1)               # (BW, k+1)
        out_t = torch.cat([tgt, torch.full((b * w, 1), -1, device=words.device)], 1)
        out_t[torch.arange(b * w, device=words.device), lens] = self.eow   # eow after the last token
        h0 = self.h0(ctx).view(b * w, self.dec_layers, -1).transpose(0, 1).contiguous()
        y, _ = self.dec(self.emb(dec_in) + ctx.unsqueeze(1), h0)
        keep = out_t.reshape(-1) >= 0                                # only real tokens + eow
        logits = self.out_ln(y.reshape(-1, y.shape[-1])[keep]) @ self.emb.weight[: self.vocab + 1].T
        return F.cross_entropy(logits.float(), out_t.reshape(-1)[keep], reduction="sum")


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


def n_params(model, exclude_embeddings=True):
    n = sum(p.numel() for p in model.parameters())
    if exclude_embeddings:
        n -= model.emb.weight.numel()
    return n


def bpc(total_nats, n_chars):
    return total_nats / n_chars / math.log(2)
