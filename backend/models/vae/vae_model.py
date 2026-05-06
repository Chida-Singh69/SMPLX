"""Lightweight sequence VAE for SMPL-X motion."""

from __future__ import annotations

import torch
import torch.nn as nn


class SignLanguageVAE(nn.Module):
    """VAE for fixed-length sequences with per-frame SMPL-X parameters.

    Input shape: [B, T, D]
    Output shape: [B, T, D]
    """

    def __init__(self, seq_len: int = 300, pose_dim: int = 182, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.frame_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(hidden_dim, pose_dim)

    def encode(self, x: torch.Tensor):
        # x: [B, T, D] -> [B, T, H]
        h = self.frame_encoder(x)
        # Pool across time: [B, H]
        h_pooled = self.temporal_pool(h.transpose(1, 2)).squeeze(-1)
        mu = self.mu_layer(h_pooled)
        logvar = self.logvar_layer(h_pooled)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, Z]
        b = z.shape[0]
        device = z.device

        # Add normalized time channel so decoder can produce temporal variation.
        t = torch.linspace(0.0, 1.0, self.seq_len, device=device).view(1, self.seq_len, 1)
        t = t.expand(b, self.seq_len, 1)

        z_rep = z.unsqueeze(1).expand(b, self.seq_len, self.latent_dim)
        zt = torch.cat([z_rep, t], dim=-1)

        h = self.decoder_input(zt)
        out = self.out_layer(h)
        return out

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def blend_latents(self, z_list: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Blend latent vectors with normalized weights.

        z_list: [K, Z]
        weights: [K]
        """
        w = weights / (weights.sum() + 1e-8)
        return torch.sum(z_list * w.unsqueeze(-1), dim=0, keepdim=True)
