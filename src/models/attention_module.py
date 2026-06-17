import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.relative_position_bias = nn.Parameter(
            torch.zeros(num_heads, 1, 1)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )

        self.gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = self.shared_mlp(x)
        max_out = self.shared_mlp(x)
        attention = torch.sigmoid(avg_out + max_out)
        gated = self.gate(torch.cat([x * attention, x], dim=-1))
        return x * gated


class CrossModalAttention(nn.Module):
    def __init__(self, dim_spatial, dim_frequency, hidden_dim=256, num_heads=4):
        super().__init__()
        self.proj_spatial = nn.Linear(dim_spatial, hidden_dim)
        self.proj_frequency = nn.Linear(dim_frequency, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, spatial_features, frequency_features):
        q = self.proj_spatial(spatial_features).unsqueeze(1)
        kv = self.proj_frequency(frequency_features).unsqueeze(1)

        attended, attn_weights = self.cross_attn(q, kv, kv)
        attended = self.norm1(attended + q)
        out = self.norm2(attended + self.ffn(attended))

        return out.squeeze(1), attn_weights


class FeaturePyramidFusion(nn.Module):
    def __init__(self, feature_dims, output_dim=512):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
            )
            for dim in feature_dims
        ])
        self.level_weights = nn.Parameter(torch.ones(len(feature_dims)))
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features):
        weights = F.softmax(self.level_weights, dim=0)
        projected = [proj(feat) for proj, feat in zip(self.projectors, features)]
        fused = sum(w * p for w, p in zip(weights, projected))
        return self.output_proj(fused)
