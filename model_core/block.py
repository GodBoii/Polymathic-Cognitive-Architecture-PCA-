from typing import Optional

import torch
import torch.nn as nn

from attention_modules import build_attention, get_attention_kind
from model_core.config import ModelConfig
from model_core.ffn import MoEFFN, build_ffn
from model_core.norms import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_kind = get_attention_kind(cfg, layer_idx)
        self.ffn_kind = cfg.ffn_kind_for_layer(layer_idx)
        self.ffn_dim = cfg.ffn_dim_for_layer(layer_idx)
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = build_attention(cfg=cfg, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        if self.ffn_kind == "moe":
            self.ffn = MoEFFN(
                d_model=cfg.d_model,
                ffn_dim=self.ffn_dim,
                num_experts=cfg.reasoning_moe_num_experts,
                top_k=cfg.reasoning_moe_top_k,
                num_groups=cfg.reasoning_moe_num_groups,
                groups_top_k=cfg.reasoning_moe_groups_top_k,
                gate_type=cfg.reasoning_moe_gate_type,
                expert_ffn_kind=cfg.reasoning_moe_expert_ffn_kind,
                bias=False,
                dropout=cfg.dropout,
            )
        else:
            self.ffn = build_ffn(
                kind=self.ffn_kind,
                d_model=cfg.d_model,
                ffn_dim=self.ffn_dim,
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
