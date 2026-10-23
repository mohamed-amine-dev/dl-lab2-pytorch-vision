"""Model definitions: custom CNN and a from-scratch Vision Transformer."""
from __future__ import annotations

import torch
import torch.nn as nn


# ------------------------------------------------------------------ 1) CNN
class CNN(nn.Module):
    """Conv(1->32,3x3,p1)-ReLU-Pool2 -> Conv(32->64,3x3,p1)-ReLU-Pool2 -> FC128-Dropout -> FC10.

    Shapes for a 28x28 input:
        conv1 -> (B,32,28,28)   pool1 -> (B,32,14,14)
        conv2 -> (B,64,14,14)   pool2 -> (B,64,7,7)
        flatten -> (B,3136)     fc1 -> (B,128)     fc2 -> (B,10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ------------------------------------------------------------------ 2) ViT from scratch
class PatchEmbedding(nn.Module):
    """Split the image into non-overlapping patches and project each to `dim`."""

    def __init__(self, in_channels: int = 1, patch_size: int = 4, dim: int = 64, img_size: int = 28):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # (B, dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)      # (B, n_patches, dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product attention written by hand (no nn.MultiheadAttention)."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale          # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)      # (B, N, C)
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm block: LN -> MHSA -> residual -> LN -> MLP -> residual."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer from scratch: patchify -> [CLS] + positional embeddings -> blocks -> head."""

    def __init__(self, img_size: int = 28, patch_size: int = 4, in_channels: int = 1,
                 num_classes: int = 10, dim: int = 64, depth: int = 6,
                 n_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim, img_size)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)                               # (B, N, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed        # (B, N+1, dim)
        x = self.blocks(self.pos_drop(x))
        return self.head(self.norm(x)[:, 0])                   # classify the [CLS] token
