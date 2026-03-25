import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_modules.gqa_rope.rope import RotaryEmbedding, apply_rotary_pos_emb


class MLAAttention(nn.Module):
    """
    Skeleton MLA block.
    Compresses K/V into a latent space then up-projects for attention.
    """

    def __init__(self, cfg, bias: bool = False) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.latent_dim = cfg.mla_latent_dim

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=bias)

        kv_dim = self.n_kv_heads * self.head_dim
        q_dim = self.n_heads * self.head_dim
        self.k_down = nn.Linear(kv_dim, self.latent_dim, bias=bias)
        self.v_down = nn.Linear(kv_dim, self.latent_dim, bias=bias)
        self.k_up = nn.Linear(self.latent_dim, q_dim, bias=bias)
        self.v_up = nn.Linear(self.latent_dim, q_dim, bias=bias)

        self.o_proj = nn.Linear(q_dim, cfg.d_model, bias=bias)
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=cfg.max_seq_len,
            base=cfg.rope_theta,
        )
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1)
        return mask.to(dtype=dtype)[None, None, :, :]

    def forward(self, x: torch.Tensor, attention_mask=None, position_ids=None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        k_latent = self.k_down(k_raw)
        v_latent = self.v_down(v_raw)
        k = self.k_up(k_latent).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(v_latent).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len=seq_len, device=x.device, dtype=q.dtype, position_ids=position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos=cos, sin=sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + self._build_causal_mask(seq_len, device=x.device, dtype=attn_scores.dtype)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn_probs, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        out = self.o_proj(out)
        return self.dropout(out)
