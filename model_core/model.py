from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cognitive_router import LatentImaginationBlock, RecursiveCognitiveBlock
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
        self.imagination_block = LatentImaginationBlock(cfg) if cfg.use_imagination and cfg.imagination_steps > 0 else None
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

    @staticmethod
    def _maybe_checkpoint(fn, x: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not enabled:
            return fn(x)
        try:
            return checkpoint(fn, x, use_reentrant=False)
        except TypeError:
            return checkpoint(fn, x)

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
            def layer_forward(hidden_states: torch.Tensor, block: TransformerBlock = layer) -> torch.Tensor:
                return block(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

            x = self._maybe_checkpoint(layer_forward, x, enabled=self.cfg.gradient_checkpointing and self.training)

        x = self.final_norm(x)
        imagination_stats = None
        if self.imagination_block is not None:
            imagination_input = x
            if return_router_stats and not (self.cfg.gradient_checkpointing and self.training):
                x, imagination_stats = self.imagination_block(x, return_stats=True)
            elif return_router_stats:
                def imagination_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                    return self.imagination_block(hidden_states, return_stats=False)

                x = self._maybe_checkpoint(
                    imagination_forward,
                    x,
                    enabled=self.cfg.gradient_checkpointing and self.training,
                )
                with torch.no_grad():
                    _, imagination_stats = self.imagination_block(imagination_input.detach(), return_stats=True)
            else:
                def imagination_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                    return self.imagination_block(hidden_states, return_stats=False)

                x = self._maybe_checkpoint(
                    imagination_forward,
                    x,
                    enabled=self.cfg.gradient_checkpointing and self.training,
                )
        if return_router_stats or return_aux_losses:
            x, router_stats = self.cognitive_block(
                x,
                aux_alpha_override=aux_alpha_override,
                return_aux=True,
            )
        else:
            def cognitive_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                return self.cognitive_block(hidden_states, aux_alpha_override=aux_alpha_override, return_aux=False)

            x = self._maybe_checkpoint(
                cognitive_forward,
                x,
                enabled=self.cfg.gradient_checkpointing and self.training,
            )
        logits = self.lm_head(x)

        out = {"logits": logits}
        if return_router_stats:
            out["router_stats"] = router_stats
            if imagination_stats is not None:
                out["imagination_stats"] = imagination_stats
        if return_aux_losses:
            if router_stats is None:
                zero = logits.new_zeros(())
                out["aux_losses"] = {
                    "total_aux_loss": zero,
                    "moe_aux_loss": zero,
                    "moe_load_balance_loss": zero,
                    "moe_entropy_reg_loss": zero,
                }
            else:
                out["aux_losses"] = {
                    "total_aux_loss": router_stats["moe_aux_loss"],
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
