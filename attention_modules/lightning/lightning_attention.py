import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_modules.gqa_rope.rope import RotaryEmbedding, apply_rotary_pos_emb


class LightningAttention(nn.Module):
    """
    Skeleton linear attention block.
    Uses causal prefix sums as a practical stand-in for advanced Lightning kernels.
    """

    def __init__(self, cfg, bias: bool = False) -> None:
        super().__init__()
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
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        return x.repeat_interleave(n_rep, dim=1)

    @staticmethod
    def _key_keep_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        # Supports additive masks from model: [batch, 1, 1, seq] with 0 keep and -1e9 masked.
        if attention_mask.dim() == 4:
            return (attention_mask[:, 0, 0, :] == 0).to(dtype=torch.float32)
        if attention_mask.dim() == 2:
            return attention_mask.float()
        return torch.ones(attention_mask.size(0), attention_mask.size(-1), device=attention_mask.device)

    def forward(self, x: torch.Tensor, attention_mask=None, position_ids=None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len=seq_len, device=x.device, dtype=q.dtype, position_ids=position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos=cos, sin=sin)

        k = self._repeat_kv(k, self.kv_repeat_factor)
        v = self._repeat_kv(v, self.kv_repeat_factor)

        qf = F.elu(q) + 1.0
        kf = F.elu(k) + 1.0

        if attention_mask is not None:
            keep = self._key_keep_mask(attention_mask).to(device=x.device, dtype=qf.dtype)  # [B, S]
            keep = keep[:, None, :, None]
            kf = kf * keep
            v = v * keep

        # Recurrent causal linear attention to avoid allocating [B, H, S, D, D].
        state = torch.zeros(
            bsz,
            self.n_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=qf.dtype,
        )
        z_state = torch.zeros(
            bsz,
            self.n_heads,
            self.head_dim,
            device=x.device,
            dtype=qf.dtype,
        )
        out = torch.empty(
            bsz,
            self.n_heads,
            seq_len,
            self.head_dim,
            device=x.device,
            dtype=qf.dtype,
        )
        for t in range(seq_len):
            k_t = kf[:, :, t, :]  # [B, H, D]
            v_t = v[:, :, t, :]   # [B, H, D]
            q_t = qf[:, :, t, :]  # [B, H, D]
            state = state + torch.einsum("bhe,bhd->bhed", k_t, v_t)
            z_state = z_state + k_t
            num_t = torch.einsum("bhe,bhed->bhd", q_t, state)
            den_t = torch.einsum("bhe,bhe->bh", q_t, z_state).unsqueeze(-1) + 1e-6
            out[:, :, t, :] = num_t / den_t

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        out = self.o_proj(out)
        return self.dropout(out)
