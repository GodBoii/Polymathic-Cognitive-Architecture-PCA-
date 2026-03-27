from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_core.ffn import build_ffn
from model_core.norms import RMSNorm


class RecursiveCognitiveBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cognitive_loops = cfg.cognitive_loops
        self.num_experts = cfg.cognitive_num_experts
        self.num_groups = cfg.cognitive_num_groups
        self.experts_per_group = cfg.experts_per_group
        self.groups_top_k = cfg.cognitive_groups_top_k
        self.top_k = cfg.cognitive_top_k
        self.gate_type = cfg.cognitive_gate_type
        self.aux_alpha = cfg.cognitive_aux_alpha
        self.entropy_alpha = cfg.cognitive_entropy_alpha
        self.active_experts_per_token = self.groups_top_k * self.top_k

        if self.top_k > self.experts_per_group:
            raise ValueError("cognitive_top_k cannot exceed experts per group")

        self.loop_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.macro_router = nn.Linear(cfg.d_model, self.num_groups, bias=False)
        self.micro_routers = nn.ModuleList(
            [nn.Linear(cfg.d_model, self.experts_per_group, bias=False) for _ in range(self.num_groups)]
        )
        self.experts = nn.ModuleList(
            [
                build_ffn(
                    kind=cfg.cognitive_ffn_kind,
                    d_model=cfg.d_model,
                    ffn_dim=cfg.cognitive_ffn_dim,
                    bias=False,
                    dropout=cfg.dropout,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.eps = 1e-9

    def _router_scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "sigmoid":
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    def _route_hierarchical(self, z_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, seq_len, _ = z_norm.shape
        tokens = bsz * seq_len
        flat_tokens = torch.arange(tokens, device=z_norm.device, dtype=torch.long)

        macro_scores = self._router_scores(self.macro_router(z_norm))  # [B, S, G]
        macro_top_vals, macro_top_idx = torch.topk(macro_scores, k=self.groups_top_k, dim=-1)
        macro_top_vals = macro_top_vals / (macro_top_vals.sum(dim=-1, keepdim=True) + self.eps)

        macro_idx_flat = macro_top_idx.reshape(tokens, self.groups_top_k)
        macro_w_flat = macro_top_vals.reshape(tokens, self.groups_top_k)

        dispatch_group = torch.zeros(tokens, self.num_groups, device=z_norm.device, dtype=z_norm.dtype)
        dispatch_group.scatter_(dim=1, index=macro_idx_flat, value=1.0)

        expert_scores = torch.zeros(bsz, seq_len, self.num_experts, device=z_norm.device, dtype=z_norm.dtype)
        assign_token_idx: list[torch.Tensor] = []
        assign_expert_idx: list[torch.Tensor] = []
        assign_weight: list[torch.Tensor] = []

        for group_id, micro_router in enumerate(self.micro_routers):
            start = group_id * self.experts_per_group
            end = start + self.experts_per_group

            micro_scores = self._router_scores(micro_router(z_norm))  # [B, S, E/G]
            expert_scores[:, :, start:end] = micro_scores * macro_scores[:, :, group_id : group_id + 1]

            micro_top_vals, micro_top_idx = torch.topk(micro_scores, k=self.top_k, dim=-1)
            micro_top_vals = micro_top_vals / (micro_top_vals.sum(dim=-1, keepdim=True) + self.eps)
            micro_top_idx_flat = micro_top_idx.reshape(tokens, self.top_k)
            micro_top_vals_flat = micro_top_vals.reshape(tokens, self.top_k)

            for slot in range(self.groups_top_k):
                mask = macro_idx_flat[:, slot] == group_id
                if not mask.any():
                    continue
                token_idx = flat_tokens[mask]
                group_w = macro_w_flat[mask, slot : slot + 1]
                local_idx = micro_top_idx_flat[mask]
                local_weight = micro_top_vals_flat[mask] * group_w
                assign_token_idx.append(token_idx[:, None].expand(-1, self.top_k).reshape(-1))
                assign_expert_idx.append((local_idx + start).reshape(-1))
                assign_weight.append(local_weight.reshape(-1))

        if assign_token_idx:
            flat_token_idx = torch.cat(assign_token_idx, dim=0)
            flat_expert_idx = torch.cat(assign_expert_idx, dim=0)
            flat_weight = torch.cat(assign_weight, dim=0)
        else:
            flat_token_idx = torch.empty(0, device=z_norm.device, dtype=torch.long)
            flat_expert_idx = torch.empty(0, device=z_norm.device, dtype=torch.long)
            flat_weight = torch.empty(0, device=z_norm.device, dtype=z_norm.dtype)

        dispatch_expert = torch.zeros(tokens, self.num_experts, device=z_norm.device, dtype=z_norm.dtype)
        if flat_token_idx.numel() > 0:
            dispatch_expert[flat_token_idx, flat_expert_idx] = 1.0

        return {
            "macro_scores": macro_scores,
            "expert_scores": expert_scores,
            "dispatch_group": dispatch_group.view(bsz, seq_len, self.num_groups),
            "dispatch_expert": dispatch_expert.view(bsz, seq_len, self.num_experts),
            "token_idx": flat_token_idx,
            "expert_idx": flat_expert_idx,
            "weight": flat_weight,
        }

    def _apply_experts(
        self,
        z: torch.Tensor,
        token_idx: torch.Tensor,
        expert_idx: torch.Tensor,
        weight: torch.Tensor,
        collect_stats: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, d_model = z.shape
        tokens = bsz * seq_len
        mixed_flat = torch.zeros(tokens, d_model, device=z.device, dtype=z.dtype)
        usage = torch.zeros(self.num_experts, device=z.device, dtype=torch.long)

        if token_idx.numel() == 0:
            return mixed_flat.view(bsz, seq_len, d_model), usage

        flat_z = z.reshape(tokens, d_model)
        sorted_expert, sort_order = torch.sort(expert_idx)
        sorted_token_idx = token_idx.index_select(0, sort_order)
        sorted_weight = weight.index_select(0, sort_order).unsqueeze(-1)
        counts = torch.bincount(sorted_expert, minlength=self.num_experts)

        start = 0
        for expert_id, expert in enumerate(self.experts):
            cnt = int(counts[expert_id].item())
            if cnt == 0:
                continue
            end = start + cnt
            selected_token_idx = sorted_token_idx[start:end]
            selected = flat_z.index_select(0, selected_token_idx)
            expert_out = expert(selected) * sorted_weight[start:end]
            mixed_flat.index_add_(0, selected_token_idx, expert_out)
            if collect_stats:
                usage[expert_id] += cnt
            start = end

        return mixed_flat.view(bsz, seq_len, d_model), usage

    def _compute_aux_losses(
        self,
        expert_scores: torch.Tensor,
        macro_scores: torch.Tensor,
        dispatch_expert: torch.Tensor,
        dispatch_group: torch.Tensor,
        aux_alpha_override: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aux_alpha = self.aux_alpha if aux_alpha_override is None else aux_alpha_override

        fi_expert = dispatch_expert.mean(dim=(0, 1)) / max(self.active_experts_per_token, 1)
        fi_expert = fi_expert / (fi_expert.sum() + self.eps)
        pi_expert = expert_scores.mean(dim=(0, 1))
        pi_expert = pi_expert / (pi_expert.sum() + self.eps)
        lb_expert = aux_alpha * self.num_experts * torch.sum(fi_expert * pi_expert)

        fi_group = dispatch_group.mean(dim=(0, 1)) / max(self.groups_top_k, 1)
        fi_group = fi_group / (fi_group.sum() + self.eps)
        pi_group = macro_scores.mean(dim=(0, 1))
        pi_group = pi_group / (pi_group.sum() + self.eps)
        lb_group = aux_alpha * self.num_groups * torch.sum(fi_group * pi_group)

        lb = 0.5 * (lb_expert + lb_group)

        if self.gate_type == "sigmoid":
            ent_expert = -(expert_scores * torch.log(expert_scores + self.eps) + (1.0 - expert_scores) * torch.log(1.0 - expert_scores + self.eps))
            ent_group = -(macro_scores * torch.log(macro_scores + self.eps) + (1.0 - macro_scores) * torch.log(1.0 - macro_scores + self.eps))
            entropy = 0.5 * (ent_expert.mean() + ent_group.mean())
        else:
            ent_expert = -(expert_scores * torch.log(expert_scores + self.eps)).sum(dim=-1).mean()
            ent_group = -(macro_scores * torch.log(macro_scores + self.eps)).sum(dim=-1).mean()
            entropy = 0.5 * (ent_expert + ent_group)

        entropy_reg = -self.entropy_alpha * entropy
        co_activation = torch.einsum("bse,bsf->ef", dispatch_expert, dispatch_expert)
        group_co_activation = torch.einsum("bsg,bsh->gh", dispatch_group, dispatch_group)
        return lb, entropy_reg, entropy, co_activation, group_co_activation

    def forward(
        self,
        z: torch.Tensor,
        aux_alpha_override: float | None = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
        usage_total = torch.zeros(self.num_experts, device=z.device, dtype=torch.long)
        group_usage_total = torch.zeros(self.num_groups, device=z.device, dtype=torch.long)
        entropy_total = torch.zeros((), device=z.device, dtype=z.dtype)
        lb_total = torch.zeros((), device=z.device, dtype=z.dtype)
        entropy_reg_total = torch.zeros((), device=z.device, dtype=z.dtype)
        co_activation_total = torch.zeros((self.num_experts, self.num_experts), device=z.device, dtype=z.dtype)
        group_co_activation_total = torch.zeros((self.num_groups, self.num_groups), device=z.device, dtype=z.dtype)

        for _ in range(self.cognitive_loops):
            z_norm = self.loop_norm(z)
            route = self._route_hierarchical(z_norm)
            mixed, usage = self._apply_experts(
                z=z_norm,
                token_idx=route["token_idx"],
                expert_idx=route["expert_idx"],
                weight=route["weight"],
                collect_stats=return_aux,
            )
            z = z + mixed

            lb_loss, entropy_reg, entropy, co_activation, group_co_activation = self._compute_aux_losses(
                expert_scores=route["expert_scores"],
                macro_scores=route["macro_scores"],
                dispatch_expert=route["dispatch_expert"],
                dispatch_group=route["dispatch_group"],
                aux_alpha_override=aux_alpha_override,
            )
            lb_total = lb_total + lb_loss
            entropy_reg_total = entropy_reg_total + entropy_reg
            co_activation_total = co_activation_total + co_activation.to(dtype=z.dtype)
            group_co_activation_total = group_co_activation_total + group_co_activation.to(dtype=z.dtype)

            if return_aux:
                usage_total = usage_total + usage
                group_usage_total = group_usage_total + route["dispatch_group"].sum(dim=(0, 1)).to(dtype=torch.long)
                entropy_total = entropy_total + entropy

        if not return_aux:
            return z

        lb_avg = lb_total / max(self.cognitive_loops, 1)
        entropy_reg_avg = entropy_reg_total / max(self.cognitive_loops, 1)
        aux_loss = lb_avg + entropy_reg_avg
        aux = {
            "cognitive_loops": self.cognitive_loops,
            "num_groups": self.num_groups,
            "groups_top_k": self.groups_top_k,
            "num_experts": self.num_experts,
            "experts_per_group": self.experts_per_group,
            "expert_top_k": self.top_k,
            "active_experts_per_token": self.active_experts_per_token,
            "gate_type": self.gate_type,
            "aux_alpha": aux_alpha_override if aux_alpha_override is not None else self.aux_alpha,
            "avg_router_entropy": float((entropy_total / max(self.cognitive_loops, 1)).detach().item()),
            "moe_load_balance_loss": lb_avg,
            "moe_entropy_reg_loss": entropy_reg_avg,
            "moe_aux_loss": aux_loss,
            "expert_usage": usage_total.tolist(),
            "group_usage": group_usage_total.tolist(),
            "co_activation": co_activation_total.detach().cpu().tolist(),
            "group_co_activation": group_co_activation_total.detach().cpu().tolist(),
        }
        return z, aux
