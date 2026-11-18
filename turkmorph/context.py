"""(B) Context disambiguation: pick each word's analysis by reading the sentence.

The deterministic tokenizer picks an analysis per word type (EM costs + audited
overrides) and leaves label-only ambiguity open (P3SG|ACC). ContextTokenizer
instead lets a small BiLSTM read the sentence and choose among the analyzer's own
candidates (alan: alan#N vs al+PART_AN; kitabı: ACC vs P3SG). The output is the
same lossless token format; only the choice differs, so decode() is unchanged.
Requires torch (only for this mode).
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

from .morphology import analyze_known, underspecify
from .tokenizer import CHUNK, tr_lower

K_MAX = 16            # candidates per word
T_MAX = 8             # default tokens per word (word representation)
S_MAX = 8             # suffixes per candidate
C_MAX = 24            # characters per word (char features: the written form itself)
CHARS = "abcçdefgğhıijklmnoöprsştuüvyzqwxâîû'’"
CHAR_ID = {c: i + 1 for i, c in enumerate(CHARS)}            # 0 = padding / other
WORD = re.compile(r"^[^\W\d_]+$")
POS = ["N", "V", "PRON", "PROPER", "Q", "X"]


class ContextModel(nn.Module):
    def __init__(self, vocab: int, d: int = 256, hidden: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, d, padding_idx=vocab)   # last id = padding
        self.case = nn.Embedding(3, d)
        self.char = nn.Embedding(len(CHARS) + 1, 64, padding_idx=0)
        self.char_cnn = nn.Conv1d(64, d, kernel_size=3, padding=1)
        self.last = nn.Embedding(vocab + 1, d, padding_idx=vocab)  # candidate's last suffix: its role
        self.lstm = nn.LSTM(d, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.pos = nn.Embedding(len(POS), d)
        self.spos = nn.Embedding(S_MAX, d)
        self.cand = nn.Sequential(nn.Linear(d, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, 2 * hidden))
        self.proj = nn.Linear(2 * hidden, 2 * hidden)
        self.cost_w = nn.Parameter(torch.tensor(-1.0))           # prior: EM cost of the analysis
        self.pad = vocab

    def forward(self, wtok, wcase, wchar, root, pos, sfx, cost, cmask):
        """wtok (B,W,T) wcase (B,W) wchar (B,W,C) root/pos (B,W,K) sfx (B,W,K,S) cost (B,W,K)
        cmask (B,W,K) -> scores (B,W,K), -inf where there is no candidate."""
        b, w, c_ = wchar.shape
        ch = self.char(wchar.view(b * w, c_)).transpose(1, 2)         # (BW,64,C)
        ch = torch.relu(self.char_cnn(ch)).max(-1).values.view(b, w, -1)
        x = self.emb(wtok).sum(2) + self.case(wcase) + ch
        h, _ = self.lstm(x)
        q = self.proj(h)                                          # (B,W,2H)
        c = self.emb(root) + self.pos(pos) + (self.emb(sfx) + self.spos.weight[:sfx.shape[-1]]).masked_fill(
            (sfx == self.pad).unsqueeze(-1), 0).sum(3)
        n_sfx = (sfx != self.pad).sum(-1, keepdim=True).clamp(min=1) - 1
        last = sfx.gather(-1, n_sfx).squeeze(-1)                  # last suffix (pad if none)
        c = self.cand(c + self.last(last))                        # (B,W,K,2H)
        s = (q.unsqueeze(2) * c).sum(-1) + self.cost_w * cost
        return s.masked_fill(~cmask, float("-inf"))


def word_case(form: str) -> int:
    return 0 if form.islower() else (2 if form.isupper() and len(form) > 1 else 1)


class Featurizer:
    """Turns words into model inputs using the tokenizer's vocabulary ids."""

    def __init__(self, tok):
        self.tok = tok
        self.v = tok.vocab.index
        self.pad = len(tok.vocab)
        self._cache: dict[str, tuple] = {}

    def candidates(self, form: str):
        """(analyses, word token ids) for one word (cached by form)."""
        hit = self._cache.get(form)
        if hit is None:
            low = tr_lower(form)
            cands = analyze_known(low, self.tok.lex, self.tok.costs) if WORD.match(form) else []
            if form == low:                                   # a lowercase word is not a name:
                common = [a for a in cands if a.root.pos != "PROPER"]   # never offer one
                cands = common or cands
            ov = self.tok.overrides.get(low)
            if ov is not None:
                cands.sort(key=lambda a: (a.root.key, a.sfx) != ov)
            cands = cands[:K_MAX]
            ids = [self.v.get(t, self.pad) for t in self.tok.encode(form)][:T_MAX]
            hit = self._cache[form] = (cands, ids)
        return hit

    def tensors(self, sentences):
        """sentences: list of lists of word forms -> padded tensors + candidate lists."""
        B, W = len(sentences), max(len(s) for s in sentences)
        wtok = torch.full((B, W, T_MAX), self.pad)
        wcase = torch.zeros((B, W), dtype=torch.long)
        wchar = torch.zeros((B, W, C_MAX), dtype=torch.long)
        root = torch.full((B, W, K_MAX), self.pad)
        pos = torch.zeros((B, W, K_MAX), dtype=torch.long)
        sfx = torch.full((B, W, K_MAX, S_MAX), self.pad)
        cost = torch.zeros((B, W, K_MAX))
        cmask = torch.zeros((B, W, K_MAX), dtype=torch.bool)
        all_cands = []
        for b, sent in enumerate(sentences):
            row = []
            for i, form in enumerate(sent):
                cands, ids = self.candidates(form)
                row.append(cands)
                wtok[b, i, :len(ids)] = torch.tensor(ids)
                wcase[b, i] = word_case(form)
                low = tr_lower(form)[-C_MAX:]                     # the end carries the suffixes
                wchar[b, i, :len(low)] = torch.tensor([CHAR_ID.get(ch, 0) for ch in low])
                if not cands:
                    continue
                c0 = cands[0].cost
                for k, a in enumerate(cands):
                    root[b, i, k] = self.v.get(f"R:{a.root.key}", self.pad)
                    pos[b, i, k] = POS.index(a.root.pos) if a.root.pos in POS else 0
                    sids = [self.v.get(f"S:{s}", self.pad) for s in a.sfx][:S_MAX]
                    if sids:
                        sfx[b, i, k, :len(sids)] = torch.tensor(sids)
                    cost[b, i, k] = min(a.cost - c0, 20.0)
                    cmask[b, i, k] = True
            all_cands.append(row)
        return (wtok, wcase, wchar, root, pos, sfx, cost, cmask), all_cands


class ContextTokenizer:
    """Same encode/decode as the wrapped TurkMorphTokenizer, with context-chosen analyses."""

    LABEL_CONFIDENCE = 0.9      # below this, label-only alternatives stay open (P3SG|ACC)
    PLAUSIBLE = 0.1             # an alternative the model gives at least this is "open"

    def __init__(self, tok, model_path, device: str | None = None):
        self.tok = tok
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(model_path, map_location=self.device, weights_only=False)
        self.kind = ck.get("kind", "lstm")
        cls = EncoderContextModel if self.kind == "encoder" else ContextModel
        self.model = cls(ck["vocab"], **ck["config"]).to(self.device).eval()
        self.model.load_state_dict(ck["state"])
        self.feat = Featurizer(tok)

    @torch.no_grad()
    def choose(self, words: list[str], window: int = 200, with_probs: bool = False):
        """word forms -> per word: chosen Analysis or None (no / single candidate);
        with_probs: (Analysis, [(candidate, probability), ...]) instead."""
        out = []
        for i in range(0, len(words), window):
            chunk = words[i:i + window]
            (x, cands) = self.feat.tensors([chunk])
            probs = self.model(*[t.to(self.device) for t in x])[0].softmax(-1).cpu()
            for j, cs in enumerate(cands[0]):
                if len(cs) < 2:
                    out.append(None)
                    continue
                k = int(probs[j].argmax())
                out.append((cs[k], [(c, float(probs[j, n])) for n, c in enumerate(cs)])
                           if with_probs else cs[k])
        return out

    def chain_for(self, best, scored):
        """The model picks the root and segmentation; a label it is not sure of stays
        underspecified, exactly as in the deterministic tokenizer (A)."""
        p_best = dict((id(c), p) for c, p in scored)[id(best)]
        if p_best >= self.LABEL_CONFIDENCE or not self.tok._union_set:
            return best.sfx
        alts = [c for c, p in scored if c is not best and c.root.key == best.root.key and p >= self.PLAUSIBLE]
        u = underspecify(best, alts, margin=float("inf"))
        if any("|" in x for x in u) and all("|" not in x or x in self.tok._union_set for x in u):
            return tuple(u)
        return best.sfx

    def encode(self, text: str) -> list[str]:
        if self.kind == "encoder":                    # the encoder reads punctuation and numbers too
            chunks = [m for m in CHUNK.finditer(text) if not m.group(3)]
            forms = [m.group(0) for m in chunks]
            is_word = [bool(m.group(1)) for m in chunks]
        else:
            forms = [m.group(1) for m in CHUNK.finditer(text) if m.group(1)]
            is_word = [True] * len(forms)
        picks = self.choose(forms, with_probs=True) if forms else []
        word_picks = [p for p, w in zip(picks, is_word) if w]
        forced = {i: (p[0].root.key, self.chain_for(*p)) for i, p in enumerate(word_picks) if p is not None}
        return self.tok.encode(text, forced=forced)

    def encode_ids(self, text):
        return [self.tok.vocab.id(t) for t in self.encode(text)]

    def decode(self, toks):
        return self.tok.decode(toks)

    def decode_ids(self, ids):
        return self.tok.decode_ids(ids)


# ------------------------------------------------- pretrained encoder (B, large) --
class TokenEncoder(nn.Module):
    """Bidirectional transformer over TurkMorph tokens, pretrained with whole-word
    masking on raw text (scripts/pretrain_context.py). ids: vocab, vocab = [MASK], vocab+1 = pad."""

    def __init__(self, vocab: int, d: int = 384, layers: int = 6, heads: int = 6, ctx: int = 512):
        super().__init__()
        self.vocab, self.mask_id, self.pad_id = vocab, vocab, vocab + 1
        self.emb = nn.Embedding(vocab + 2, d, padding_idx=vocab + 1)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        for e in (self.emb, self.pos):                       # small init: the output layer is tied
            nn.init.normal_(e.weight, std=0.02)              # to emb, N(0,1) logits diverge in bf16
        with torch.no_grad():
            self.emb.weight[self.pad_id].zero_()

    def forward(self, ids, positions=None):
        pad = ids == self.pad_id
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        h = self.emb(ids) + self.pos(positions.clamp(max=self.pos.num_embeddings - 1))
        return self.ln(self.layers(h, src_key_padding_mask=pad))

    def mlm_logits(self, h):
        return h @ self.emb.weight[: self.vocab].T                   # tied output


class EncoderContextModel(ContextModel):
    """ContextModel whose word vectors come from a pretrained TokenEncoder (mean over
    each word's tokens) instead of the BiLSTM; candidate scoring is unchanged."""

    def __init__(self, vocab: int, d: int = 256, hidden: int = 256, enc: dict | None = None):
        super().__init__(vocab, d, hidden)
        self.encoder = TokenEncoder(vocab, **(enc or {}))
        de = self.encoder.emb.embedding_dim
        self.to_word = nn.Linear(de + d, 2 * hidden)
        del self.lstm

    def forward(self, wtok, wcase, wchar, root, pos, sfx, cost, cmask):
        b, w, t = wtok.shape
        ids = wtok.view(b, w * t).clone()
        real = ids != self.pad
        ids[~real] = self.encoder.pad_id
        positions = (real.long().cumsum(1) - 1).clamp(min=0)
        hs = self.encoder(ids, positions).view(b, w, t, -1)
        m = real.view(b, w, t, 1).float()
        hw = (hs * m).sum(2) / m.sum(2).clamp(min=1)              # word vector: mean of its tokens
        c_ = wchar.shape[-1]
        ch = self.char(wchar.view(b * w, c_)).transpose(1, 2)
        ch = torch.relu(self.char_cnn(ch)).max(-1).values.view(b, w, -1)
        # bounded like the BiLSTM's output: LayerNorm-scale encoder states would saturate the scores
        q = self.proj(torch.tanh(self.to_word(torch.cat([hw, ch + self.case(wcase)], -1))))
        c = self.emb(root) + self.pos(pos) + (self.emb(sfx) + self.spos.weight[:sfx.shape[-1]]).masked_fill(
            (sfx == self.pad).unsqueeze(-1), 0).sum(3)
        n_sfx = (sfx != self.pad).sum(-1, keepdim=True).clamp(min=1) - 1
        last = sfx.gather(-1, n_sfx).squeeze(-1)
        c = self.cand(c + self.last(last))
        s = (q.unsqueeze(2) * c).sum(-1) / q.shape[-1] ** 0.5 + self.cost_w * cost   # scaled dot product
        return s.masked_fill(~cmask, float("-inf"))
