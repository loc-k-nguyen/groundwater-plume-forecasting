"""
Transport-Aware Attentive Neural Process for long-horizon groundwater plume forecasting.

This extends the validated Neural Process baseline with:
  - residual prediction anchored to the last observed frame
  - optional K conditioning
  - optional centroid-velocity flow context

The output is always an absolute log10(C) prediction. When residual mode is
enabled, the decoder predicts a temporal residual and the last observed frame is
added back internally.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.neural_process import Decoder, ScalarTimeEmbedding, StatsHead


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


class GroundwaterTAANP(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 32,
        token_dim: int = 64,
        latent_dim: int = 32,
        time_embed_dim: int = 32,
        num_heads: int = 4,
        kl_beta: float = 1e-4,
        use_residual: bool = True,
        use_k: bool = True,
        use_flow: bool = True,
        k_fusion: str = "append",
        y_bg: float = -12.0,
        y_cap: float = 2.0,
        eps_c: float = 1e-12,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.token_dim = int(token_dim)
        self.latent_dim = int(latent_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.kl_beta = float(kl_beta)
        self.use_residual = bool(use_residual)
        self.use_k = bool(use_k)
        self.use_flow = bool(use_flow)
        # k_fusion: "append" (raw K channel, default/original) or "film"
        # (separate K encoder modulating the decoder context via FiLM).
        self.k_fusion = str(k_fusion)
        self._append_k = self.use_k and self.k_fusion == "append"
        self._film_k = self.use_k and self.k_fusion == "film"
        self.y_bg = float(y_bg)
        self.y_cap = float(y_cap)
        self.eps_c = float(eps_c)

        self.time_embed = ScalarTimeEmbedding(self.time_embed_dim)
        frame_in = 1 + (1 if self._append_k else 0) + (2 if self.use_flow else 0)
        self.frame_encoder = ConvEncoder(frame_in, self.hidden_channels)
        if self._film_k:
            # Separate K encoder producing per-pixel FiLM (gamma, beta) that
            # modulate the pooled context map before decoding.
            self.k_encoder = ConvEncoder(1, self.hidden_channels)
            self.k_film = nn.Conv2d(self.hidden_channels, 2 * self.hidden_channels, kernel_size=1)
        flow_token_dim = 2 if self.use_flow else 0
        self.token_proj = nn.Linear(
            self.hidden_channels + self.time_embed_dim + flow_token_dim,
            self.token_dim,
        )
        self.query_proj = nn.Linear(self.time_embed_dim, self.token_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.prior_head = StatsHead(self.token_dim, self.latent_dim)
        self.posterior_head = StatsHead(self.token_dim, self.latent_dim)

        decoder_in = self.hidden_channels + 1 + self.token_dim + self.latent_dim
        if self._append_k:
            decoder_in += 1
        if self.use_flow:
            decoder_in += 2
        self.decoder = Decoder(decoder_in, self.hidden_channels)

    def _flow_features(self, c_seq: torch.Tensor) -> torch.Tensor:
        batch, steps, height, width = c_seq.shape
        if steps <= 1:
            return c_seq.new_zeros(batch, steps, 2)

        c_phys = torch.clamp(torch.pow(10.0, c_seq.float()) - self.eps_c, min=0.0)
        y = torch.linspace(-1.0, 1.0, height, device=c_seq.device, dtype=c_phys.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=c_seq.device, dtype=c_phys.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        mass = c_phys.sum(dim=(-2, -1)).clamp_min(1e-8)
        cx = (c_phys * xx).sum(dim=(-2, -1)) / mass
        cy = (c_phys * yy).sum(dim=(-2, -1)) / mass
        centroid = torch.stack([cx, cy], dim=-1).to(dtype=c_seq.dtype)

        velocity = centroid.new_zeros(batch, steps, 2)
        velocity[:, 1:] = centroid[:, 1:] - centroid[:, :-1]
        return velocity

    def _encode_sequence(
        self,
        c_seq: torch.Tensor,
        k_field: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, height, width = c_seq.shape
        flow = self._flow_features(c_seq) if self.use_flow else c_seq.new_zeros(batch, steps, 0)

        parts = [c_seq.reshape(batch * steps, 1, height, width)]
        if self._append_k:
            k_rep = k_field.unsqueeze(1).expand(batch, steps, 1, height, width)
            parts.append(k_rep.reshape(batch * steps, 1, height, width))
        if self.use_flow:
            flow_map = flow[:, :, :, None, None].expand(batch, steps, 2, height, width)
            parts.append(flow_map.reshape(batch * steps, 2, height, width))

        feat = self.frame_encoder(torch.cat(parts, dim=1)).reshape(
            batch, steps, self.hidden_channels, height, width
        )
        pooled = feat.mean(dim=(-2, -1))
        time_emb = self.time_embed(times.reshape(batch * steps)).reshape(
            batch, steps, self.time_embed_dim
        )
        token_in = torch.cat([pooled, time_emb, flow], dim=-1)
        tokens = self.token_proj(token_in)
        return feat, tokens, flow

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

        obs_feat, obs_tokens, obs_flow = self._encode_sequence(c_obs, k_field, times_obs)
        context_map = obs_feat.mean(dim=1)
        if self._film_k:
            k_feat = self.k_encoder(k_field)
            gamma, beta = self.k_film(k_feat).chunk(2, dim=1)
            context_map = (1.0 + gamma) * context_map + beta
        prior_mu, prior_logvar = self.prior_head(obs_tokens.mean(dim=1))

        post_mu = prior_mu
        post_logvar = prior_logvar
        if c_target is not None and times_target is not None:
            if times_target.ndim == 1:
                times_target = times_target.unsqueeze(0).expand(c_target.shape[0], -1)
            _, target_tokens, _ = self._encode_sequence(c_target, k_field, times_target)
            all_tokens = torch.cat([obs_tokens, target_tokens], dim=1)
            post_mu, post_logvar = self.posterior_head(all_tokens.mean(dim=1))

        z = self._sample(post_mu, post_logvar, deterministic=deterministic)

        batch, query_steps = times_query.shape
        height, width = c_obs.shape[-2:]
        query_time = self.time_embed(times_query.reshape(-1)).reshape(
            batch, query_steps, self.time_embed_dim
        )
        query_tokens = self.query_proj(query_time)
        attn_tokens, _ = self.cross_attn(query_tokens, obs_tokens, obs_tokens, need_weights=False)

        context_rep = context_map.unsqueeze(1).expand(
            batch, query_steps, self.hidden_channels, height, width
        )
        time_map = times_query[:, :, None, None, None].expand(batch, query_steps, 1, height, width)
        attn_map = attn_tokens[:, :, :, None, None].expand(
            batch, query_steps, self.token_dim, height, width
        )
        z_map = z[:, None, :, None, None].expand(batch, query_steps, self.latent_dim, height, width)

        dec_parts = [context_rep]
        if self._append_k:
            dec_parts.append(k_field.unsqueeze(1).expand(batch, query_steps, 1, height, width))
        dec_parts.extend([time_map, attn_map, z_map])
        if self.use_flow:
            # obs_flow: (B, T_obs, 2). Broadcast the last observed velocity over the
            # query steps and spatial dimensions (correct for any batch size).
            last_flow = obs_flow[:, -1, :].view(batch, 1, 2, 1, 1)
            last_flow = last_flow.expand(batch, query_steps, 2, height, width)
            dec_parts.append(last_flow)

        dec_in = torch.cat(dec_parts, dim=2)
        raw = self.decoder(dec_in.reshape(batch * query_steps, dec_in.shape[2], height, width))
        raw = raw.reshape(batch, query_steps, height, width)

        if self.use_residual:
            last_obs = c_obs[:, -1:, :, :].expand(batch, query_steps, height, width)
            preds = last_obs + raw
        else:
            preds = raw
        return preds.clamp(self.y_bg, self.y_cap), {
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
        }
