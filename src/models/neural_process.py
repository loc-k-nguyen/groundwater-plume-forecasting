"""
Neural Process style baseline for long-horizon groundwater plume forecasting.

This is a lightweight Attentive Neural Process inspired design:
  - encode each observed (C_t, K) frame pair into a spatial feature map
  - aggregate observed features into a spatial context map
  - build context tokens and attend target-time queries to them
  - infer a global latent z from context, and a posterior z during training
  - decode arbitrary query times directly without autoregressive rollout
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        return self.net(times.unsqueeze(-1))


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StatsHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int) -> None:
        super().__init__()
        hidden = max(in_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * latent_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.net(x).chunk(2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=8.0)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroundwaterNeuralProcess(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 32,
        token_dim: int = 64,
        latent_dim: int = 32,
        time_embed_dim: int = 32,
        num_heads: int = 4,
        kl_beta: float = 1e-4,
        y_bg: float = -12.0,
        y_cap: float = 2.0,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.token_dim = int(token_dim)
        self.latent_dim = int(latent_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.kl_beta = float(kl_beta)
        self.y_bg = float(y_bg)
        self.y_cap = float(y_cap)

        self.time_embed = ScalarTimeEmbedding(self.time_embed_dim)
        self.frame_encoder = ConvEncoder(2, self.hidden_channels)
        self.token_proj = nn.Linear(self.hidden_channels + self.time_embed_dim, self.token_dim)
        self.query_proj = nn.Linear(self.time_embed_dim, self.token_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.prior_head = StatsHead(self.token_dim, self.latent_dim)
        self.posterior_head = StatsHead(self.token_dim, self.latent_dim)
        decoder_in = self.hidden_channels + 1 + 1 + self.token_dim + self.latent_dim
        self.decoder = Decoder(decoder_in, self.hidden_channels)

    def _encode_sequence(
        self,
        c_seq: torch.Tensor,
        k_field: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, height, width = c_seq.shape
        k_rep = k_field.unsqueeze(1).expand(batch, steps, 1, height, width)
        enc_in = torch.cat(
            [
                c_seq.reshape(batch * steps, 1, height, width),
                k_rep.reshape(batch * steps, 1, height, width),
            ],
            dim=1,
        )
        feat = self.frame_encoder(enc_in).reshape(batch, steps, self.hidden_channels, height, width)
        pooled = feat.mean(dim=(-2, -1))
        time_emb = self.time_embed(times.reshape(batch * steps)).reshape(batch, steps, self.time_embed_dim)
        tokens = self.token_proj(torch.cat([pooled, time_emb], dim=-1))
        return feat, tokens

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool) -> torch.Tensor:
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(
        self,
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        post_var = torch.exp(post_logvar)
        prior_var = torch.exp(prior_logvar)
        kl = 0.5 * (
            prior_logvar
            - post_logvar
            + (post_var + (post_mu - prior_mu) ** 2) / prior_var
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    def forward(
        self,
        c_obs: torch.Tensor,
        k_field: torch.Tensor,
        times_obs: torch.Tensor,
        times_query: torch.Tensor,
        c_target: Optional[torch.Tensor] = None,
        times_target: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if times_obs.ndim == 1:
            times_obs = times_obs.unsqueeze(0).expand(c_obs.shape[0], -1)
        if times_query.ndim == 1:
            times_query = times_query.unsqueeze(0).expand(c_obs.shape[0], -1)

        obs_feat, obs_tokens = self._encode_sequence(c_obs, k_field, times_obs)
        context_map = obs_feat.mean(dim=1)
        prior_mu, prior_logvar = self.prior_head(obs_tokens.mean(dim=1))

        post_mu = prior_mu
        post_logvar = prior_logvar
        if c_target is not None and times_target is not None:
            if times_target.ndim == 1:
                times_target = times_target.unsqueeze(0).expand(c_target.shape[0], -1)
            _, target_tokens = self._encode_sequence(c_target, k_field, times_target)
            all_tokens = torch.cat([obs_tokens, target_tokens], dim=1)
            post_mu, post_logvar = self.posterior_head(all_tokens.mean(dim=1))

        z = self._sample(post_mu, post_logvar, deterministic=deterministic)

        query_time = self.time_embed(times_query.reshape(-1)).reshape(c_obs.shape[0], times_query.shape[1], self.time_embed_dim)
        query_tokens = self.query_proj(query_time)
        attn_tokens, _ = self.cross_attn(query_tokens, obs_tokens, obs_tokens, need_weights=False)

        batch, query_steps = times_query.shape
        height, width = c_obs.shape[-2:]
        context_rep = context_map.unsqueeze(1).expand(batch, query_steps, self.hidden_channels, height, width)
        k_rep = k_field.unsqueeze(1).expand(batch, query_steps, 1, height, width)
        time_map = times_query[:, :, None, None, None].expand(batch, query_steps, 1, height, width)
        attn_map = attn_tokens[:, :, :, None, None].expand(batch, query_steps, self.token_dim, height, width)
        z_map = z[:, None, :, None, None].expand(batch, query_steps, self.latent_dim, height, width)

        dec_in = torch.cat([context_rep, k_rep, time_map, attn_map, z_map], dim=2)
        preds = self.decoder(dec_in.reshape(batch * query_steps, dec_in.shape[2], height, width))
        preds = preds.reshape(batch, query_steps, height, width).clamp(self.y_bg, self.y_cap)
        return preds, {
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
        }
