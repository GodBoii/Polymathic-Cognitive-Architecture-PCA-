from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cognitive_router import RecursiveCognitiveBlock
from model_core.block import TransformerBlock
from model_core.config import ModelConfig
from model_core.norms import RMSNorm


class PCAModel(nn.Module):
    def __init__(self, cfg: ModelConfig, tie_embeddings: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.cognitive_block = RecursiveCognitiveBlock(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _prepare_attention_mask(attention_mask: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        # Accepts [batch, seq_len] with 1=keep, 0=mask and returns additive mask.
        if attention_mask.dim() == 2:
            mask = (1.0 - attention_mask.float()) * -1e9
            return mask[:, None, None, :].to(dtype=target_dtype)
        return attention_mask.to(dtype=target_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_router_stats: bool = False,
        return_aux_losses: bool = False,
        aux_alpha_override: Optional[float] = None,
    ) -> dict:
        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask, target_dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, position_ids=position_ids)

        x = self.final_norm(x)
        if return_router_stats or return_aux_losses:
            x, router_stats = self.cognitive_block(
                x,
                aux_alpha_override=aux_alpha_override,
                return_aux=True,
            )
        else:
            x = self.cognitive_block(x, aux_alpha_override=aux_alpha_override)
        logits = self.lm_head(x)

        out = {"logits": logits}
        if return_router_stats:
            out["router_stats"] = router_stats
        if return_aux_losses:
            out["aux_losses"] = {
                "moe_aux_loss": router_stats["moe_aux_loss"],
                "moe_load_balance_loss": router_stats["moe_load_balance_loss"],
                "moe_entropy_reg_loss": router_stats["moe_entropy_reg_loss"],
            }
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out
