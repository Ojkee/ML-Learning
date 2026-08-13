from __future__ import annotations

from torch import nn
import torch
import torch.nn.functional as F

from config import ModelConfig


class Embeddings(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.d_model = torch.tensor(cfg.d_model)
        self.token_embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.register_buffer("positional_encoding", self._init_positional_encoding(cfg))
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def _init_positional_encoding(self, cfg: ModelConfig) -> torch.Tensor:
        powers = torch.arange(0, cfg.d_model, 2) / cfg.d_model
        denum = torch.scalar_tensor(10_000).pow(powers)
        pos = torch.arange(cfg.context_len, dtype=torch.float)
        pos = torch.cat([pos.view(-1, 1) for _ in range(cfg.d_model // 2)], dim=1)
        pos /= denum
        encoding = torch.zeros(cfg.context_len, cfg.d_model)
        encoding[:, 0::2] = pos.sin()
        encoding[:, 1::2] = pos.cos()
        encoding.requires_grad = False
        return encoding

    def forward(self, x):
        T = x.shape[1]
        tok_emb = self.token_embedding(x) * self.d_model.sqrt()
        return self.dropout(tok_emb + self.positional_encoding[:T])  # type: ignore


class AttentionHead(nn.Module):
    def __init__(self, cfg: ModelConfig, masked: bool) -> None:
        super().__init__()
        self.d_k = cfg.d_k
        self.keys = nn.Linear(cfg.d_model, cfg.d_k, bias=False)
        self.querries = nn.Linear(cfg.d_model, cfg.d_k, bias=False)
        self.values = nn.Linear(cfg.d_model, cfg.d_v, bias=False)
        self.masked = masked
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.context_len, cfg.context_len))
        )

    def forward(self, x, kv=None, pad_mask=None):
        T = x.shape[1]
        kv = x if kv is None else kv
        q = self.querries(x)
        k = self.keys(kv)
        v = self.values(kv)
        w = q @ k.transpose(-2, -1) * self.d_k ** (-1 / 2)
        if self.masked:
            w = w.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # type: ignore

        if pad_mask is not None:
            w = w.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        return F.softmax(w, dim=-1) @ v


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.ReLU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
            nn.Dropout(cfg.dropout_rate),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, masked: bool) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(cfg, masked) for _ in range(cfg.h)])
        self.projection = nn.Linear(cfg.h * cfg.d_v, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, x, kv=None, pad_mask=None):
        y_heads = torch.cat([head(x, kv, pad_mask) for head in self.heads], dim=-1)
        y_projection = self.projection(y_heads)
        return self.dropout(y_projection)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(cfg, masked=False)
        self.layer_norm_of_attention = nn.LayerNorm(cfg.d_model)

        self.ff = FeedForward(cfg)
        self.layer_norm_of_ff = nn.LayerNorm(cfg.d_model)

    def forward(self, x, pad_mask=None):
        y_attention = self.attention(x, pad_mask=pad_mask)
        x = self.layer_norm_of_attention(x + y_attention)
        y_ff = self.ff(x)
        return self.layer_norm_of_ff(x + y_ff)


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.masked_attention = MultiHeadAttention(cfg, masked=True)
        self.layer_norm_of_masked = nn.LayerNorm(cfg.d_model)

        self.cross_attention = MultiHeadAttention(cfg, masked=False)
        self.layer_norm_of_cross = nn.LayerNorm(cfg.d_model)

        self.ff = FeedForward(cfg)
        self.layer_norm_of_ff = nn.LayerNorm(cfg.d_model)

    def forward(self, x, encoder_output, src_pad_mask=None, tgt_pad_mask=None):
        y_masked = self.masked_attention(x, pad_mask=tgt_pad_mask)
        x = self.layer_norm_of_masked(x + y_masked)

        y_cross = self.cross_attention(x, encoder_output, pad_mask=src_pad_mask)
        x = self.layer_norm_of_cross(x + y_cross)

        y_ff = self.ff(x)
        return self.layer_norm_of_ff(x + y_ff)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        shared_embedding = Embeddings(cfg)
        with torch.no_grad():
            shared_embedding.token_embedding.weight *= 0.1
        self.encoder_emb = shared_embedding
        self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.N)])

        self.decoder_emb = shared_embedding
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.N)])

        self.logits_layer = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)
        self.logits_layer.weight = shared_embedding.token_embedding.weight

    def forward(
        self,
        encoder_input,
        decoder_input,
        src_pad_mask=None,
        tgt_pad_mask=None,
    ):
        x_encoder = self.encoder_emb(encoder_input)
        for layer in self.encoder:
            x_encoder = layer(x_encoder, pad_mask=src_pad_mask)
        y_encoder = x_encoder

        x_decoder = self.decoder_emb(decoder_input)
        for decode_layer in self.decoder:
            x_decoder = decode_layer(x_decoder, y_encoder, src_pad_mask, tgt_pad_mask)

        return self.logits_layer(x_decoder)
