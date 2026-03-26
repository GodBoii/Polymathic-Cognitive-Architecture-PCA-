from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_core.ffn import SwiGLUFFN
from model_core.norms import RMSNorm


class RecursiveCognitiveBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cognitive_loops = cfg.cognitive_loops
        self.num_experts = cfg.cognitive_num_experts
        self.top_k = cfg.cognitive_top_k
        self.gate_type = cfg.cognitive_gate_type
        self.aux_alpha = cfg.cognitive_aux_alpha
        self.entropy_alpha = cfg.cognitive_entropy_alpha

        self.loop_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.router = nn.Linear(cfg.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLUFFN(
                    d_model=cfg.d_model,
                    ffn_dim=cfg.ffn_dim,
                    bias=False,
                    dropout=cfg.dropout,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.eps = 1e-9

    def _apply_experts(
        self,
        z: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
        collect_stats: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [B, S, D], topk_idx/topk_vals: [B, S, K]
        bsz, seq_len, d_model = z.shape
        tokens = bsz * seq_len
        mixed_flat = torch.zeros(tokens, d_model, device=z.device, dtype=z.dtype)
        usage = torch.zeros(self.num_experts, device=z.device, dtype=torch.long)
        flat_z = z.reshape(tokens, d_model)
        flat_idx = topk_idx.reshape(-1)
        flat_weight = topk_vals.reshape(-1, 1)
        token_index = (
            torch.arange(tokens, device=z.device, dtype=torch.long)
            .unsqueeze(-1)
            .expand(tokens, self.top_k)
            .reshape(-1)
        )
        # Group routed slots by expert id once to avoid per-expert boolean scans.
        sorted_expert, sort_order = torch.sort(flat_idx)
        sorted_token_index = token_index.index_select(0, sort_order)
        sorted_weight = flat_weight.index_select(0, sort_order)
        counts = torch.bincount(sorted_expert, minlength=self.num_experts)

        start = 0
        for expert_id, expert in enumerate(self.experts):
            cnt = int(counts[expert_id].item())
            if cnt == 0:
                continue
            end = start + cnt
            selected_token_idx = sorted_token_index[start:end]
            selected = flat_z.index_select(0, selected_token_idx)
            expert_out = expert(selected) * sorted_weight[start:end]
            mixed_flat.index_add_(0, selected_token_idx, expert_out)
            if collect_stats:
                usage[expert_id] += cnt
            start = end
        return mixed_flat.view(bsz, seq_len, d_model), usage

    def _router_scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "sigmoid":
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    def _compute_aux_losses(
        self,
        router_scores: torch.Tensor,
        topk_idx: torch.Tensor,
        aux_alpha_override: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # router_scores: [B, S, E]
        bsz, seq_len, num_experts = router_scores.shape
        dispatch = torch.zeros_like(router_scores)
        dispatch.scatter_(dim=-1, index=topk_idx, value=1.0)

        # Fraction of dispatched slots per expert, normalized to sum to 1.
        fi = dispatch.mean(dim=(0, 1)) / max(self.top_k, 1)
        fi = fi / (fi.sum() + self.eps)

        # Mean router score per expert; normalized for compatibility across gate types.
        pi = router_scores.mean(dim=(0, 1))
        pi = pi / (pi.sum() + self.eps)

        # Switch-style load balancing proxy.
        aux_alpha = self.aux_alpha if aux_alpha_override is None else aux_alpha_override
        lb = aux_alpha * num_experts * torch.sum(fi * pi)

        # Encourage non-collapsed routing distributions.
        if self.gate_type == "sigmoid":
            # Binary entropy averaged over experts.
            ent = -(router_scores * torch.log(router_scores + self.eps) + (1.0 - router_scores) * torch.log(1.0 - router_scores + self.eps))
            entropy = ent.mean()
        else:
            entropy = -(router_scores * torch.log(router_scores + self.eps)).sum(dim=-1).mean()
        entropy_reg = -self.entropy_alpha * entropy
        co_activation = torch.einsum("bse,bsf->ef", dispatch, dispatch)
        return lb, entropy_reg, entropy, co_activation

    def forward(
        self,
        z: torch.Tensor,
        aux_alpha_override: float | None = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
        usage_total = torch.zeros(self.num_experts, device=z.device, dtype=torch.long)
        entropy_total = torch.zeros((), device=z.device, dtype=z.dtype)
        lb_total = torch.zeros((), device=z.device, dtype=z.dtype)
        entropy_reg_total = torch.zeros((), device=z.device, dtype=z.dtype)
        co_activation_total = torch.zeros(
            (self.num_experts, self.num_experts), device=z.device, dtype=z.dtype
        )

        for _ in range(self.cognitive_loops):
            z_norm = self.loop_norm(z)
            router_logits = self.router(z_norm)
            router_scores = self._router_scores(router_logits)
            topk_vals, topk_idx = torch.topk(router_scores, k=self.top_k, dim=-1)
            topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + self.eps)

            mixed, usage = self._apply_experts(
                z=z_norm,
                topk_idx=topk_idx,
                topk_vals=topk_vals,
                collect_stats=return_aux,
            )
            z = z + mixed

            lb_loss, entropy_reg, entropy, co_activation = self._compute_aux_losses(
                router_scores=router_scores,
                topk_idx=topk_idx,
                aux_alpha_override=aux_alpha_override,
            )
            lb_total = lb_total + lb_loss
            entropy_reg_total = entropy_reg_total + entropy_reg
            co_activation_total = co_activation_total + co_activation.to(dtype=z.dtype)

            if return_aux:
                usage_total = usage_total + usage
                entropy_total = entropy_total + entropy

        if not return_aux:
            return z

        lb_avg = lb_total / max(self.cognitive_loops, 1)
        entropy_reg_avg = entropy_reg_total / max(self.cognitive_loops, 1)
        aux_loss = lb_avg + entropy_reg_avg
        aux = {
            "cognitive_loops": self.cognitive_loops,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "gate_type": self.gate_type,
            "aux_alpha": aux_alpha_override if aux_alpha_override is not None else self.aux_alpha,
            "avg_router_entropy": float((entropy_total / max(self.cognitive_loops, 1)).detach().item()),
            "moe_load_balance_loss": lb_avg,
            "moe_entropy_reg_loss": entropy_reg_avg,
            "moe_aux_loss": aux_loss,
            "expert_usage": usage_total.tolist(),
            "co_activation": co_activation_total.detach().cpu().tolist(),
        }
        return z, aux
