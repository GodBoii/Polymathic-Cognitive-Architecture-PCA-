from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from model_core.ffn import build_ffn
from model_core.norms import RMSNorm


class LatentImaginationBlock(nn.Module):
    """
    A first-step latent workspace:
    1. compress token states into a small latent array,
    2. iteratively refine that array,
    3. write the refined latent state back to tokens.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.num_latents = cfg.imagination_num_latents
        self.num_steps = cfg.imagination_steps
        self.anchor_alpha = cfg.imagination_anchor_alpha
        self.latent_seed = nn.Parameter(torch.randn(1, self.num_latents, cfg.d_model) * 0.02)

        self.input_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.latent_norm_1 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.latent_norm_2 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.latent_norm_3 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.output_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)

        self.token_to_latent = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.imagination_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.latent_self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.imagination_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.latent_ffn = build_ffn(
            kind=cfg.cognitive_ffn_kind,
            d_model=cfg.d_model,
            ffn_dim=cfg.imagination_ffn_dim,
            bias=False,
            dropout=cfg.dropout,
        )
        self.latent_to_token = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.imagination_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        source = self.input_norm(x)
        pooled = source.mean(dim=1, keepdim=True)
        anchor = pooled + self.latent_seed.expand(x.size(0), -1, -1)
        z = anchor
        avg_delta_norm = torch.zeros((), device=x.device, dtype=x.dtype)

        for _ in range(self.num_steps):
            prev = z
            z = z + self.token_to_latent(self.latent_norm_1(z), source, source, need_weights=False)[0]
            latent_norm = self.latent_norm_2(z)
            z = z + self.latent_self_attn(latent_norm, latent_norm, latent_norm, need_weights=False)[0]
            z = z + self.latent_ffn(self.latent_norm_3(z))
            z = (1.0 - self.anchor_alpha) * z + self.anchor_alpha * anchor
            avg_delta_norm = avg_delta_norm + (z - prev).norm(dim=-1).mean()

        token_update = self.latent_to_token(self.output_norm(x), z, z, need_weights=False)[0]
        out = x + token_update

        if not return_stats:
            return out

        stats = {
            "num_latents": float(self.num_latents),
            "num_steps": float(self.num_steps),
            "anchor_alpha": float(self.anchor_alpha),
            "avg_delta_norm": float((avg_delta_norm / max(self.num_steps, 1)).detach().item()),
            "final_latent_norm": float(z.norm(dim=-1).mean().detach().item()),
        }
        return out, stats
