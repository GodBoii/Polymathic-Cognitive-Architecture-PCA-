from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ) -> dict:
        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask, target_dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, position_ids=position_ids)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        out = {"logits": logits}
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
