#
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
kraken.lib.ppocr.nrtr
~~~~~~~~~~~~~~~~~~~~~~~~

The NRTR auxiliary recognition head used for guided training of CTC (GTC). It is
a teacher-forced attention decoder over the backbone features (no encoder), used
during training only and discarded at inference.

Token convention (blank/PAD = 0): ``0`` = PAD, ``1..max_label`` = code points,
``max_label+1`` = BOS, ``max_label+2`` = EOS; vocabulary size = ``max_label + 3``.
"""
import math

import torch
import torch.nn as nn

__all__ = ['NRTRHead']


class Embeddings(nn.Module):
    """Token embedding scaled by sqrt(d_model) (PAD index 0)."""

    def __init__(self, d_model, vocab, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, dropout, dim, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):  # x: (B, T, dim)
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class MultiheadAttention(nn.Module):
    """MHA with separate self-attention (fused qkv) and cross-attention paths."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        self.scale = self.head_dim ** -0.5
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, query, key=None, attn_mask=None):
        B, qN, _ = query.shape
        if self.self_attn:
            qkv = self.qkv(query).reshape(B, qN, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            kN = key.shape[1]
            q = self.q(query).reshape(B, qN, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(key).reshape(B, kN, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, qN, self.embed_dim)
        return self.out_proj(x)


class _Mlp(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DecoderBlock(nn.Module):
    """Post-norm decoder block: causal self-attention, cross-attention, FFN."""

    def __init__(self, d_model, nhead, dim_feedforward, attn_dropout=0.0, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=attn_dropout, self_attn=True)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=attn_dropout, self_attn=False)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = _Mlp(d_model, dim_feedforward, dropout)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, self_mask=None):
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(tgt, attn_mask=self_mask)))
        tgt = self.norm2(tgt + self.dropout2(self.cross_attn(tgt, key=memory)))
        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class NRTRHead(nn.Module):
    """
    NRTR transformer-decoder guidance head.

    Args:
        in_channels: backbone output channels.
        vocab_size: ``codec.max_label + 3`` (PAD/0, codepoints, BOS, EOS).
        d_model: decoder width.
        num_decoder_layers: number of decoder blocks.
        nhead: attention heads (defaults to ``d_model // 32``).
        dim_feedforward: FFN width (defaults to ``d_model * 4``).
    """

    def __init__(self, in_channels, vocab_size, d_model=384, num_decoder_layers=4,
                 nhead=None, dim_feedforward=None, dropout=0.1, max_len=512):
        super().__init__()
        nhead = nhead or max(1, d_model // 32)
        dim_feedforward = dim_feedforward or d_model * 4
        # project the backbone memory: (B, C, 1, W) -> (B, W, d_model)
        self.input_proj = nn.Linear(in_channels, d_model, bias=False)
        self.embedding = Embeddings(d_model, vocab_size, padding_idx=0)
        self.positional_encoding = PositionalEncoding(dropout, d_model, max_len=max_len)
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward, dropout=dropout)
            for _ in range(num_decoder_layers)])
        self.tgt_word_prj = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _causal_mask(sz, device):
        # additive mask: 0 on/below the diagonal, -inf above
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, feat, tgt_in):
        """
        Args:
            feat: backbone feature ``(B, C, 1, W)``.
            tgt_in: teacher-forcing input tokens ``(B, T)`` ([BOS, c1, ..., EOS][:-1]).
        Returns:
            logits ``(B, T, vocab_size)``.
        """
        memory = self.input_proj(feat.flatten(2).transpose(1, 2))  # (B, W, d_model)
        tgt = self.positional_encoding(self.embedding(tgt_in))     # (B, T, d_model)
        mask = self._causal_mask(tgt.shape[1], tgt.device)
        for layer in self.decoder:
            tgt = layer(tgt, memory, self_mask=mask)
        return self.tgt_word_prj(tgt)
