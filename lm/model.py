"""Small LMs for the tokenizer comparison.

GPT       flat token-level decoder (used with every tokenizer).
Hourglass token blocks -> word-level blocks -> token blocks (ported from MorphComp):
          the middle runs once per word, so it costs ~1/2.2 of a flat block per token.
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


# word-class bits per token id (see turkmorph.tokenizer.group_flags)
WC_CONT, WC_PREFIX, WC_DIGIT = 1, 2, 4


def word_starts(idx, wclass):
    """1 where a token opens a word group. Same rule as group_flags; the first two
    positions (BOS, first token) always open a group."""
    c = wclass[idx]
    prev = torch.nn.functional.pad(c[:, :-1], (1, 0))
    cont = ((c & WC_CONT) > 0) | ((prev & WC_PREFIX) > 0) | (((c & WC_DIGIT) > 0) & ((prev & WC_DIGIT) > 0))
    ws = ~cont
    ws[:, :2] = True
    return ws


class Hourglass(nn.Module):
    """Flat token prefix/suffix stacks around a word-level stack.

    Word k's pooled state (sum of its token states after the prefix stack) can only be
    used by tokens of word k+1 onwards: tokens of word k receive G[k-1]. Giving G[k]
    to any token of word k would leak (word k may contain later tokens)."""

    POOLS = ("sum", "last_mean")
    MAX_IN_WORD = 16

    def __init__(self, vocab, d=1024, n_pre=4, n_mid=16, n_post=4, heads=16, ctx=2048, d_mid=None,
                 wclass=None, pool="last_mean", in_word_pos=True):
        super().__init__()
        if pool not in self.POOLS:
            raise ValueError(f"pool must be one of {self.POOLS}")
        d_mid = d_mid or d
        self.pool, self.in_word_pos = pool, in_word_pos
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        # morpheme slot inside the word (root=0, 1st suffix=1, ...): sum pooling alone is order-blind
        self.in_word = nn.Embedding(self.MAX_IN_WORD, d) if in_word_pos else None
        self.pos_w = nn.Embedding(ctx, d_mid)
        self.pre = nn.ModuleList(Block(d, heads) for _ in range(n_pre))
        self.down = nn.Linear(2 * d if pool == "last_mean" else d, d_mid, bias=False)
        self.mid = nn.ModuleList(Block(d_mid, d_mid // 64) for _ in range(n_mid))
        self.ln_mid = nn.LayerNorm(d_mid)
        self.up = nn.Linear(d_mid, d, bias=False)
        self.post = nn.ModuleList(Block(d, heads) for _ in range(n_post))
        self.ln = nn.LayerNorm(d)
        self.register_buffer("wclass", torch.zeros(vocab, dtype=torch.uint8) if wclass is None
                             else torch.as_tensor(wclass, dtype=torch.uint8))
        self.apply(_init)
        nn.init.zeros_(self.up.weight)                      # starts as a flat 8-layer model

    def hidden(self, idx, pad=None):
        """pad: bool (B, T), True on right padding. Padding joins the last word instead of
        forming one word per position (it follows every real token, so nothing real sees it)."""
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        ws = word_starts(idx, self.wclass)
        if pad is not None:
            ws = ws & ~pad
        widx = torch.cumsum(ws.long(), 1) - 1                                     # 0-based word number
        x = self.emb(idx) + self.pos(pos)
        if self.in_word is not None:
            start = torch.cummax(torch.where(ws, pos, torch.zeros_like(pos)), 1).values
            x = x + self.in_word((pos - start).clamp(max=self.MAX_IN_WORD - 1))
        for blk in self.pre:
            x = blk(x)
        n_w = int(widx.max()) + 1
        index = widx.unsqueeze(-1).expand_as(x)
        g = x.new_zeros(b, n_w, x.shape[-1]).scatter_add(1, index, x)
        if self.pool == "last_mean":
            count = x.new_zeros(b, n_w).scatter_add(1, widx, torch.ones_like(widx, dtype=x.dtype))
            last = torch.zeros(b, n_w, dtype=torch.long, device=idx.device).scatter_reduce(
                1, widx, pos.expand(b, -1), reduce="amax", include_self=False)   # last token of each word
            g = torch.cat([torch.gather(x, 1, last.unsqueeze(-1).expand(-1, -1, x.shape[-1])),
                           g / count.clamp(min=1).unsqueeze(-1)], -1)
        g = self.down(g) + self.pos_w(torch.arange(n_w, device=idx.device))
        for blk in self.mid:
            g = blk(g)
        g = self.ln_mid(g)
        g = torch.cat([g.new_zeros(b, 1, g.shape[-1]), g[:, :-1]], 1)            # shift by one word
        x = x + self.up(torch.gather(g, 1, widx.unsqueeze(-1).expand(-1, -1, g.shape[-1])))
        for blk in self.post:
            x = blk(x)
        return self.ln(x)

    def forward(self, idx, targets):
        logits = self.hidden(idx, targets < 0) @ self.emb.weight.T
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.view(-1),
                               reduction="sum")


def word_classes(toks):
    """uint8 class bits for a vocabulary (list of token strings); BOS/extra ids get 0."""
    import numpy as np
    from turkmorph.tokenizer import MERGE_PREFIX, WORD_PREFIX
    out = np.zeros(len(toks), dtype=np.uint8)
    for i, t in enumerate(toks):
        if (t.startswith(("S:", "B:", "E:", MERGE_PREFIX)) or (t.startswith("L:") and not t.startswith("L:▁"))
                or t in ("<apos>", "<apos2>", "<lit>")):
            out[i] |= WC_CONT
        if t in WORD_PREFIX:
            out[i] |= WC_PREFIX
        if t.startswith("D:"):
            out[i] |= WC_DIGIT
    return out


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
