import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F

import torch

from .rope import RotaryEmbedding, apply_rotary_pos_emb

if TYPE_CHECKING:
    from model_core.config import ModelConfig


class GQAAttention(nn.Module):
    def __init__(self, cfg: "ModelConfig", bias: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_repeat_factor = cfg.kv_repeat_factor

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=bias)

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=cfg.max_seq_len,
            base=cfg.rope_theta,
        )
        self.attn_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # x: [batch, n_kv_heads, seq_len, head_dim]
        if n_rep == 1:
            return x
        return x.repeat_interleave(n_rep, dim=1)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1)
        return mask.to(dtype=dtype)[None, None, :, :]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
        # x: [batch, seq_len, d_model]
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(
            seq_len=seq_len,
            device=x.device,
            dtype=q.dtype,
            position_ids=position_ids,
        )
        q, k = apply_rotary_pos_emb(q, k, cos=cos, sin=sin)

        k = self.repeat_kv(k, self.kv_repeat_factor)
        v = self.repeat_kv(v, self.kv_repeat_factor)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self._build_causal_mask(seq_len, device=x.device, dtype=attn_scores.dtype)
        attn_scores = attn_scores + causal_mask
        if attention_mask is not None:
            # attention_mask is additive and expected broadcastable to [batch, 1, seq_len, seq_len]
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(q.dtype)
        attn_probs = self.attn_dropout(attn_probs)
        out = torch.matmul(attn_probs, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        out = self.o_proj(out)

        if not return_debug:
            return out

        debug = {
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "attn_probs_shape": list(attn_probs.shape),
            "kv_repeat_factor": self.kv_repeat_factor,
            "causal_mask_shape": list(causal_mask.shape),
        }
        return out, debug
