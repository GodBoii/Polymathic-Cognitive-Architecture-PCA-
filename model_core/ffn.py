import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.up_proj(x))
        x = self.down_proj(x)
        return self.dropout(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return self.dropout(x)


class MoEFFN(nn.Module):
    """
    Single-pass hierarchical MoE FFN for transformer blocks.
    Router chooses group top-k then expert top-k per selected group.
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        num_groups: int = 1,
        groups_top_k: int = 1,
        gate_type: str = "sigmoid",
        expert_ffn_kind: str = "swiglu",
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if num_groups <= 0:
            raise ValueError("num_groups must be > 0")
        if num_experts % num_groups != 0:
            raise ValueError("num_experts must be divisible by num_groups")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if groups_top_k <= 0:
            raise ValueError("groups_top_k must be > 0")
        experts_per_group = num_experts // num_groups
        if top_k > experts_per_group:
            raise ValueError("top_k must be <= experts_per_group")
        if groups_top_k > num_groups:
            raise ValueError("groups_top_k must be <= num_groups")
        if gate_type not in {"softmax", "sigmoid"}:
            raise ValueError("gate_type must be 'softmax' or 'sigmoid'")
        if expert_ffn_kind == "moe":
            raise ValueError("expert_ffn_kind cannot be 'moe' to avoid recursive MoE nesting")

        self.num_experts = num_experts
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.top_k = top_k
        self.groups_top_k = groups_top_k
        self.gate_type = gate_type
        self.eps = 1e-9
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.macro_router = nn.Linear(d_model, num_groups, bias=False)
        self.micro_routers = nn.ModuleList(
            [nn.Linear(d_model, self.experts_per_group, bias=False) for _ in range(self.num_groups)]
        )
        self.experts = nn.ModuleList(
            [
                build_ffn(
                    kind=expert_ffn_kind,
                    d_model=d_model,
                    ffn_dim=ffn_dim,
                    bias=bias,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

    def _router_scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "sigmoid":
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        bsz, seq_len, d_model = x.shape
        tokens = bsz * seq_len
        flat_tokens = torch.arange(tokens, device=x.device, dtype=torch.long)

        macro_scores = self._router_scores(self.macro_router(x))  # [B, S, G]
        macro_top_vals, macro_top_idx = torch.topk(macro_scores, k=self.groups_top_k, dim=-1)
        macro_top_vals = macro_top_vals / (macro_top_vals.sum(dim=-1, keepdim=True) + self.eps)

        macro_idx_flat = macro_top_idx.reshape(tokens, self.groups_top_k)
        macro_w_flat = macro_top_vals.reshape(tokens, self.groups_top_k)

        assign_token_idx: list[torch.Tensor] = []
        assign_expert_idx: list[torch.Tensor] = []
        assign_weight: list[torch.Tensor] = []

        for group_id, micro_router in enumerate(self.micro_routers):
            start = group_id * self.experts_per_group

            micro_scores = self._router_scores(micro_router(x))  # [B, S, E/G]
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

        mixed_flat = torch.zeros(tokens, d_model, device=x.device, dtype=x.dtype)
        if not assign_token_idx:
            return self.dropout(mixed_flat.view(bsz, seq_len, d_model))

        token_idx = torch.cat(assign_token_idx, dim=0)
        expert_idx = torch.cat(assign_expert_idx, dim=0)
        weight = torch.cat(assign_weight, dim=0)

        flat_x = x.reshape(tokens, d_model)
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
            selected = flat_x.index_select(0, selected_token_idx)
            expert_out = expert(selected) * sorted_weight[start:end]
            mixed_flat.index_add_(0, selected_token_idx, expert_out)
            start = end

        return self.dropout(mixed_flat.view(bsz, seq_len, d_model))


def build_ffn(kind: str, d_model: int, ffn_dim: int, bias: bool = False, dropout: float = 0.0) -> nn.Module:
    if kind == "standard":
        return StandardFFN(d_model=d_model, ffn_dim=ffn_dim, bias=bias, dropout=dropout)
    if kind == "swiglu":
        return SwiGLUFFN(d_model=d_model, ffn_dim=ffn_dim, bias=bias, dropout=dropout)
    raise ValueError(f"Unsupported FFN kind: {kind}")
