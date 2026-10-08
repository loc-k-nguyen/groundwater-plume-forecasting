"""
Earthformer-inspired lightweight spatiotemporal transformer for long-horizon groundwater plume forecasting.

This is not a byte-for-byte port of the official Earthformer implementation.
Instead, it is a tractable backup baseline that keeps the key high-level idea:
joint space-time token processing with direct future-time querying.

Design:
  - patchify each concentration frame into a spatial token grid
  - add a matching conductivity token grid
  - apply factorized spatial then temporal self-attention blocks
  - decode future times with learned time queries plus cross-attention to memory
  - reconstruct future concentration fields patch-by-patch
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizedSTBlock(nn.Module):
    """
    Factorized space-time attention:
      1. spatial attention independently at each timestep
      2. temporal attention independently at each spatial token location
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C)
        bsz, t_steps, n_tokens, dim = x.shape

        # spatial attention over N at each timestep
        xs = x.reshape(bsz * t_steps, n_tokens, dim)
        xs_norm = self.spatial_norm(xs)
        xs_attn, _ = self.spatial_attn(xs_norm, xs_norm, xs_norm, need_weights=False)
        xs = xs + xs_attn
        x = xs.reshape(bsz, t_steps, n_tokens, dim)

        # temporal attention over T at each spatial token
        xt = x.permute(0, 2, 1, 3).reshape(bsz * n_tokens, t_steps, dim)
        xt_norm = self.temporal_norm(xt)
        xt_attn, _ = self.temporal_attn(xt_norm, xt_norm, xt_norm, need_weights=False)
        xt = xt + xt_attn
        x = xt.reshape(bsz, n_tokens, t_steps, dim).permute(0, 2, 1, 3).contiguous()

        # token-wise MLP
        x = x + self.mlp(self.mlp_norm(x))
        return x


class FutureDecoderBlock(nn.Module):
    """
    Decode future-time query tokens using memory from observed tokens.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        qn = self.query_norm(query)
        cross, _ = self.cross_attn(qn, memory, memory, need_weights=False)
        query = query + cross
        sn = self.self_norm(query)
        self_out, _ = self.self_attn(sn, sn, sn, need_weights=False)
        query = query + self_out
        query = query + self.mlp(self.mlp_norm(query))
        return query


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        # times: (B, T)
        return self.net(times.unsqueeze(-1))


class GroundwaterEarthformerLite(nn.Module):
    def __init__(
        self,
        image_size: int = 320,
        patch_size: int = 40,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        y_bg: float = -12.0,
        y_cap: float = 2.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")

        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.y_bg = float(y_bg)
        self.y_cap = float(y_cap)

        self.frame_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.k_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.time_embed = ScalarTimeEmbedding(embed_dim)
        max_grid = max(int(image_size // patch_size), 16)
        self.row_pos = nn.Parameter(torch.zeros(1, max_grid, embed_dim))
        self.col_pos = nn.Parameter(torch.zeros(1, max_grid, embed_dim))

        self.encoder = nn.ModuleList(
            [FactorizedSTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.decoder = nn.ModuleList(
            [FutureDecoderBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(2)]
        )

        self.patch_head = nn.Linear(embed_dim, patch_size * patch_size)
        self.output_norm = nn.LayerNorm(embed_dim)

    def _patchify_frames(self, frames: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # frames: (B,T,H,W) -> (B,T,N,C), dynamic N
        bsz, t_steps, height, width = frames.shape
        x = frames.reshape(bsz * t_steps, 1, height, width)
        tok = self.frame_embed(x)                                  # (B*T,C,G,G)
        grid_h, grid_w = tok.shape[-2:]
        tok = tok.flatten(2).transpose(1, 2)                       # (B*T,N,C)
        return tok.reshape(bsz, t_steps, grid_h * grid_w, self.embed_dim), grid_h, grid_w

    def _patchify_k(self, k_field: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        embed = self.k_embed(k_field)
        tok = embed.flatten(2).transpose(1, 2)                     # (B,N,C)
        grid_h = embed.shape[-2]
        grid_w = embed.shape[-1]
        return tok, grid_h, grid_w

    def _spatial_pos(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if grid_h > self.row_pos.shape[1] or grid_w > self.col_pos.shape[1]:
            raise ValueError(
                f"Grid {grid_h}x{grid_w} exceeds positional capacity "
                f"{self.row_pos.shape[1]}x{self.col_pos.shape[1]}"
            )
        row = self.row_pos[:, :grid_h].unsqueeze(2)                # (1,H,1,C)
        col = self.col_pos[:, :grid_w].unsqueeze(1)                # (1,1,W,C)
        pos = row + col                                            # (1,H,W,C)
        return pos.reshape(1, grid_h * grid_w, self.embed_dim)

    def _add_embeddings(
        self,
        tokens: torch.Tensor,
        k_tokens: torch.Tensor,
        times: torch.Tensor,
        spatial_pos: torch.Tensor,
    ) -> torch.Tensor:
        # tokens: (B,T,N,C), k_tokens: (B,N,C), times: (B,T)
        time_tok = self.time_embed(times).unsqueeze(2)             # (B,T,1,C)
        return tokens + k_tokens.unsqueeze(1) + spatial_pos.unsqueeze(1) + time_tok

    def _build_memory(
        self,
        c_seq: torch.Tensor,
        k_field: torch.Tensor,
        times_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        tokens, grid_h, grid_w = self._patchify_frames(c_seq)
        k_tokens, k_grid_h, k_grid_w = self._patchify_k(k_field)
        if (grid_h, grid_w) != (k_grid_h, k_grid_w):
            raise ValueError(f"Mismatched token grids: C={grid_h}x{grid_w}, K={k_grid_h}x{k_grid_w}")
        spatial_pos = self._spatial_pos(grid_h, grid_w)
        x = self._add_embeddings(tokens, k_tokens, times_obs, spatial_pos)
        for block in self.encoder:
            x = block(x)
        memory = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])   # (B,T*N,C)
        last_tokens = x[:, -1]                                                 # (B,N,C)
        return memory, last_tokens + k_tokens + spatial_pos, grid_h, grid_w

    def _decode_queries(
        self,
        base_query: torch.Tensor,
        memory: torch.Tensor,
        times_future: torch.Tensor,
        num_patches: int,
    ) -> torch.Tensor:
        # base_query: (B,N,C), times_future: (B,Tf)
        bsz, t_future = times_future.shape
        time_tok = self.time_embed(times_future).unsqueeze(2)      # (B,Tf,1,C)
        query = base_query.unsqueeze(1) + time_tok                 # (B,Tf,N,C)
        query = query.reshape(bsz, t_future * num_patches, self.embed_dim)
        for block in self.decoder:
            query = block(query, memory)
        query = self.output_norm(query)
        return query.reshape(bsz, t_future, num_patches, self.embed_dim)

    def _unpatchify(self, patch_tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        # patch_tokens: (B,Tf,N,C) -> (B,Tf,H,W)
        bsz, t_future, _, _ = patch_tokens.shape
        patch_pixels = self.patch_head(patch_tokens)               # (B,Tf,N,P*P)
        patch_pixels = patch_pixels.reshape(
            bsz, t_future, grid_h, grid_w, self.patch_size, self.patch_size
        )
        frames = (
            patch_pixels.permute(0, 1, 2, 4, 3, 5)
            .contiguous()
            .reshape(bsz, t_future, grid_h * self.patch_size, grid_w * self.patch_size)
        )
        return frames

    def forward(
        self,
        c_seq: torch.Tensor,
        k_field: torch.Tensor,
        times_obs: torch.Tensor,
        times_future: torch.Tensor,
    ) -> torch.Tensor:
        if times_obs.ndim == 1:
            times_obs = times_obs.unsqueeze(0).expand(c_seq.shape[0], -1)
        if times_future.ndim == 1:
            times_future = times_future.unsqueeze(0).expand(c_seq.shape[0], -1)
        memory, base_query, grid_h, grid_w = self._build_memory(c_seq, k_field, times_obs)
        future_tokens = self._decode_queries(base_query, memory, times_future, grid_h * grid_w)
        return self._unpatchify(future_tokens, grid_h, grid_w).clamp(self.y_bg, self.y_cap)
