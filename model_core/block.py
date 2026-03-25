from typing import Optional

import torch
import torch.nn as nn

from attention_modules import build_attention, get_attention_kind
from model_core.config import ModelConfig
from model_core.ffn import SwiGLUFFN
from model_core.norms import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_kind = get_attention_kind(cfg, layer_idx)
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = build_attention(cfg=cfg, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn = SwiGLUFFN(
            d_model=cfg.d_model,
            ffn_dim=cfg.ffn_dim,
            bias=False,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask, position_ids=position_ids)
        # Pre-norm FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x
