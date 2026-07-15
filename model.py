"""CPU-friendly GPT variant in plain PyTorch.

Implements RMSNorm, causal multi-head self-attention (no flash attention),
SwiGLU MLP, and GPT-2 style initialization.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 512
    block_size = 256
    n_layer = 4
    n_head = 6
    n_embd = 192
    dropout = 0.0
    tie_weights = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x_norm * self.weight


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj._is_residual_proj = True
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        bsz, seq_len, channels = x.shape

        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:seq_len, :seq_len]
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = (8 * cfg.n_embd) // 3
        self.w_gate = nn.Linear(cfg.n_embd, hidden)
        self.w_val = nn.Linear(cfg.n_embd, hidden)
        self.proj = nn.Linear(hidden, cfg.n_embd)
        self.proj._is_residual_proj = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        val = self.w_val(x)
        return self.drop(self.proj(gate * val))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.norm_2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "_is_residual_proj", False):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, seq_len = idx.shape
        if seq_len > self.cfg.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.cfg.block_size}")

        pos = torch.arange(seq_len, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
