from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionHead(nn.Module):
    def __init__(
        self,
        emb_size: int,
        head_size: int,
        context_len: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.querries = nn.Linear(emb_size, head_size, bias=False)
        self.keys = nn.Linear(emb_size, head_size, bias=False)
        self.values = nn.Linear(emb_size, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len)))

    def forward(self, x):
        T = x.shape[1]
        q = self.querries(x)
        k = self.keys(x)
        v = self.values(x)
        w = q @ k.transpose(-2, -1) * self.head_size ** (-1/2)
        attention = w.masked_fill(self.mask[:T, :T] == 0, float("-inf")) # type: ignore
        self.out = F.softmax(attention, dim=-1) @ v
        return self.out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        context_len: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        head_size = emb_size // num_heads
        self.heads = nn.ModuleList(
            AttentionHead(emb_size, head_size, context_len) for _ in range(num_heads)
        )
        self.projection_layer = nn.Linear(num_heads * head_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.projection_layer(x)
        self.out = self.dropout(x)
        return self.out


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, mid_layer_factor: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_size, mid_layer_factor * emb_size),
            nn.GELU(),
            nn.Linear(mid_layer_factor * emb_size, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        context_len: int,
    ) -> None:
        super().__init__()
        self.sa_heads = MultiHeadAttention(num_heads, emb_size, context_len)
        self.feedforward = FeedForward(emb_size)
        self.layer_norm_preheads = nn.LayerNorm(emb_size)
        self.layer_norm_postheads = nn.LayerNorm(emb_size)

    def forward(self, x):
        # skip connections
        x = x + self.sa_heads(self.layer_norm_preheads(x))
        x = x + self.feedforward(self.layer_norm_postheads(x))
        return x


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        context_len: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.tok_embeddings = nn.Embedding(vocab_size, emb_size)
        self.pos_embeddings = nn.Embedding(context_len, emb_size)
        self.flow = nn.Sequential(*[AttentionBlock(emb_size, num_heads, context_len) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(emb_size)
        self.logits_layer = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        tok_emb = self.tok_embeddings(x)
        pos_emb = self.pos_embeddings(torch.arange(self.context_len, device=device))
        x = tok_emb + pos_emb
        x = self.flow(x)
        x = self.layer_norm(x)
        return self.logits_layer(x)
