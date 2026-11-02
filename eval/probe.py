# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adapted from V-JEPA 2 (https://github.com/facebookresearch/vjepa2), which is licensed under the MIT license.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# V-JEPA 2's frozen-eval head, ported from vjepa2/src/models/attentive_pooler.py (AttentiveClassifier, depth=4,
# complete_block=True) and src/models/utils/modules.py (Block, Attention, CrossAttentionBlock, CrossAttention, MLP):
# three self-attention blocks over the encoder's tokens, then one learned query cross-attending to them, then a linear
# classifier. Module names and construction order follow upstream, so a seeded init matches. Upstream's CrossAttention
# builds no output projection, so there is none here.

def trunc_normal_(tensor, std):
    return nn.init.trunc_normal_(tensor, std=std, a=-2.0, b=2.0)

class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = rearrange(self.qkv(x), "b n (three heads dim) -> three b heads n dim", three=3, heads=self.num_heads)
        return self.proj(rearrange(F.scaled_dot_product_attention(q, k, v), "b heads n dim -> b n (heads dim)"))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)

    def forward(self, q, x):
        k, v = rearrange(self.kv(x), "b n (two heads dim) -> two b heads n dim", two=2, heads=self.num_heads)
        attn = F.scaled_dot_product_attention(rearrange(self.q(q), "b n (heads dim) -> b heads n dim", heads=self.num_heads), k, v)
        return rearrange(attn, "b heads n dim -> b n (heads dim)")

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.xattn = CrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, q, x):
        # norm1 is applied to the tokens only (the query is projected raw), norm2 to the query before the MLP
        q = q + self.xattn(q, self.norm1(x))
        return q + self.mlp(self.norm2(q))

class AttentivePooler(nn.Module):
    """V-JEPA 2 depth-4 complete pooler, with the upstream construction order."""

    def __init__(self, embed_dim=768, num_heads=16, init_std=0.02):
        super().__init__()
        if num_heads < 1 or embed_dim < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Build the cross block first, as upstream does, to preserve seeded init.
        self.cross_attention_block = CrossAttentionBlock(embed_dim, num_heads)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(3)])
        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=init_std)
        self.apply(self._init_weights)
        with torch.no_grad():
            for layer_id, block in enumerate(self.blocks, start=1):
                block.attn.proj.weight.div_(math.sqrt(2.0 * layer_id))
                block.mlp.fc2.weight.div_(math.sqrt(2.0 * layer_id))
            # Upstream reuses the last self-attention layer index: sqrt(6), not sqrt(8).
            self.cross_attention_block.mlp.fc2.weight.div_(math.sqrt(6.0))

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            trunc_normal_(layer.weight, std=self.init_std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.zeros_(layer.bias)
            nn.init.ones_(layer.weight)

    def forward(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        query = self.query_tokens.repeat(len(tokens), 1, 1)
        return self.cross_attention_block(query, tokens)

class AttentiveClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_heads=16, num_classes=1000, init_std=0.02):
        super().__init__()
        self.pooler = AttentivePooler(embed_dim, num_heads, init_std)
        # Created after the initialized pooler; keep PyTorch's default Linear init.
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens):
        return self.linear(self.pooler(tokens).squeeze(1))

def param_groups(module):
    # V-JEPA 2's init_opt: biases and 1-D parameters are excluded from weight decay
    decay = [p for n, p in module.named_parameters() if "bias" not in n and p.ndim != 1]
    no_decay = [p for n, p in module.named_parameters() if "bias" in n or p.ndim == 1]
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

class CosineLRWD(torch.optim.lr_scheduler.LambdaLR):
    """V-JEPA 2's WarmupCosineSchedule (warmup 0, final_lr 0) and CosineWDSchedule as one per-step scheduler: the lr
    follows a cosine from its initial value to 0, the weight decay of every decayed group one to final_wd. Upstream steps
    before every iteration, so iteration i uses step i + 1."""

    def __init__(self, optimizer, total_steps, final_wd):
        self.total_steps, self.final_wd = total_steps, final_wd
        self.ref_wd = [g["weight_decay"] for g in optimizer.param_groups]
        super().__init__(optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step + 1, total_steps) / total_steps)))

    def step(self, epoch=None):
        super().step(epoch)
        cos = 0.5 * (1 + math.cos(math.pi * min(self.last_epoch + 1, self.total_steps) / self.total_steps))
        for group, wd in zip(self.optimizer.param_groups, self.ref_wd):
            if wd > 0:
                group["weight_decay"] = max(self.final_wd, self.final_wd + (wd - self.final_wd) * cos)
