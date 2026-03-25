from typing import Optional, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(start_dim=-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding dim must be even.")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.max_seq_cached = 0

    def _maybe_build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.max_seq_cached >= seq_len and self.cos_cached.device == device and self.cos_cached.dtype == dtype:
            return
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        self.cos_cached = cos[None, None, :, :]
        self.sin_cached = sin[None, None, :, :]
        self.max_seq_cached = seq_len

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._maybe_build_cache(seq_len=seq_len, device=device, dtype=dtype)
        if position_ids is None:
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
            return cos, sin

        # position_ids: [batch, seq_len]
        cos = self.cos_cached[0, 0].index_select(0, position_ids.reshape(-1))
        sin = self.sin_cached[0, 0].index_select(0, position_ids.reshape(-1))
        bsz, seqlen = position_ids.shape
        cos = cos.view(bsz, seqlen, self.dim).unsqueeze(1)
        sin = sin.view(bsz, seqlen, self.dim).unsqueeze(1)
        return cos, sin
