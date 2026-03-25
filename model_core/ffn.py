import torch
import torch.nn as nn
import torch.nn.functional as F


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
