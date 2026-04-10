from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model_core.config import ModelConfig
from model_core.ffn import build_ffn
from model_core.norms import RMSNorm


def _build_causal_bool_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn = build_ffn(
            kind=cfg.ffn_kind if cfg.ffn_kind != "moe" else "swiglu",
            d_model=cfg.d_model,
            ffn_dim=int(cfg.ffn_dim),
            bias=False,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        norm_x = self.attn_norm(x)
        attn_out = self.attn(
            norm_x,
            norm_x,
            norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn = build_ffn(
            kind=cfg.ffn_kind if cfg.ffn_kind != "moe" else "swiglu",
            d_model=cfg.d_model,
            ffn_dim=int(cfg.ffn_dim),
            bias=False,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        self_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        norm_x = self.self_norm(x)
        causal_mask = _build_causal_bool_mask(norm_x.size(1), device=norm_x.device)
        self_out = self.self_attn(
            norm_x,
            norm_x,
            norm_x,
            attn_mask=causal_mask,
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self_out
        cross_out = self.cross_attn(
            self.cross_norm(x),
            latents,
            latents,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        x = x + cross_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ThreePhasePCAModel(nn.Module):
    def __init__(self, cfg: ModelConfig, tie_embeddings: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(int(cfg.encoder_layers))])
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(int(cfg.decoder_layers))])

        self.latent_seed = nn.Parameter(torch.randn(1, cfg.imagination_num_latents, cfg.d_model) * 0.02)
        self.token_to_latent = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.imagination_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.latent_norm_1 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.latent_self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.imagination_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.latent_norm_2 = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.latent_ffn = build_ffn(
            kind=cfg.cognitive_ffn_kind,
            d_model=cfg.d_model,
            ffn_dim=cfg.imagination_ffn_dim,
            bias=False,
            dropout=cfg.dropout,
        )
        self.summary_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.self_attn_gate = nn.Parameter(torch.tensor(float(cfg.imagination_update_scale)))
        self.ffn_gate = nn.Parameter(torch.tensor(float(cfg.imagination_update_scale)))
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _maybe_checkpoint(fn, x: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not enabled:
            return fn(x)
        try:
            return checkpoint(fn, x, use_reentrant=False)
        except TypeError:
            return checkpoint(fn, x)

    @staticmethod
    def _key_padding_mask(attention_mask: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            return None
        if attention_mask.size(1) != seq_len:
            return None
        return attention_mask == 0

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        return self.embed_tokens(input_ids) + self.position_embed(pos)

    def _run_encoder(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for block in self.encoder:
            def enc_forward(hidden_states: torch.Tensor, layer: EncoderBlock = block) -> torch.Tensor:
                return layer(hidden_states, key_padding_mask=key_padding_mask)

            x = self._maybe_checkpoint(enc_forward, x, enabled=self.cfg.gradient_checkpointing and self.training)
        return x

    def _run_latent_workspace(
        self,
        encoder_states: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict, dict]:
        latents = self.latent_seed.expand(encoder_states.size(0), -1, -1)
        latents = latents + self.token_to_latent(
            latents,
            encoder_states,
            encoder_states,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )[0]
        anchor = latents
        avg_delta = torch.zeros((), device=encoder_states.device, dtype=encoder_states.dtype)
        step_delta_norms: list[torch.Tensor] = []
        step_delta_squares: list[torch.Tensor] = []

        for step_idx in range(max(self.cfg.imagination_steps, 1)):
            prev = latents
            latent_norm = self.latent_norm_1(latents)
            step_scale = self.cfg.imagination_step_decay ** step_idx
            self_attn_out = self.latent_self_attn(
                latent_norm,
                latent_norm,
                latent_norm,
                need_weights=False,
            )[0]
            latents = latents + (torch.sigmoid(self.self_attn_gate) * step_scale) * self_attn_out
            ffn_out = self.latent_ffn(self.latent_norm_2(latents))
            latents = latents + (torch.sigmoid(self.ffn_gate) * step_scale) * ffn_out
            latents = (1.0 - self.cfg.imagination_anchor_alpha) * latents + self.cfg.imagination_anchor_alpha * anchor
            step_delta = latents - prev
            step_delta_norm = step_delta.norm(dim=-1).mean()
            avg_delta = avg_delta + step_delta_norm
            step_delta_norms.append(step_delta_norm.detach())
            step_delta_squares.append(step_delta.float().pow(2).mean())

        summary = encoder_states.mean(dim=1)
        summary_target = self.summary_proj(summary)
        latent_summary = latents.mean(dim=1)
        stability_loss = torch.stack(step_delta_squares).mean() if step_delta_squares else latents.new_zeros(())
        consistency_loss = F.mse_loss(latent_summary.float(), summary_target.float())
        norm_loss = latents.float().pow(2).mean()
        contraction_terms: list[torch.Tensor] = []
        for i in range(1, len(step_delta_squares)):
            contraction_terms.append(torch.relu(step_delta_squares[i] - step_delta_squares[i - 1]))
        contraction_loss = (
            torch.stack(contraction_terms).mean()
            if contraction_terms
            else latents.new_zeros(())
        )
        imagination_aux_loss = (
            self.cfg.imagination_stability_alpha * stability_loss
            + self.cfg.imagination_consistency_alpha * consistency_loss
            + self.cfg.imagination_norm_alpha * norm_loss
            + self.cfg.imagination_contraction_alpha * contraction_loss
        ) * self.cfg.imagination_aux_alpha

        stats = {
            "num_latents": float(self.cfg.imagination_num_latents),
            "num_steps": float(self.cfg.imagination_steps),
            "avg_delta_norm": float((avg_delta / max(self.cfg.imagination_steps, 1)).detach().item()),
            "final_latent_norm": float(latents.norm(dim=-1).mean().detach().item()),
            "self_attn_gate": float(torch.sigmoid(self.self_attn_gate).detach().item()),
            "ffn_gate": float(torch.sigmoid(self.ffn_gate).detach().item()),
            "step_decay": float(self.cfg.imagination_step_decay),
            "step_delta_norms": [float(v.item()) for v in step_delta_norms],
            "convergence_ratio": float(
                (step_delta_norms[-1] / (step_delta_norms[0] + 1e-8)).item()
                if step_delta_norms
                else 1.0
            ),
            "delta_monotonic_decrease": bool(
                all(step_delta_norms[i] <= step_delta_norms[i - 1] for i in range(1, len(step_delta_norms)))
            ),
        }
        aux = {
            "total_aux_loss": imagination_aux_loss.to(dtype=latents.dtype),
            "moe_aux_loss": latents.new_zeros(()),
            "moe_load_balance_loss": latents.new_zeros(()),
            "moe_entropy_reg_loss": latents.new_zeros(()),
            "imagination_aux_loss": imagination_aux_loss.to(dtype=latents.dtype),
            "imagination_stability_loss": stability_loss.to(dtype=latents.dtype),
            "imagination_consistency_loss": consistency_loss.to(dtype=latents.dtype),
            "imagination_norm_loss": norm_loss.to(dtype=latents.dtype),
            "imagination_contraction_loss": contraction_loss.to(dtype=latents.dtype),
        }
        return latents, stats, aux

    def _run_decoder(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        decoder_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for block in self.decoder:
            def dec_forward(hidden_states: torch.Tensor, layer: DecoderBlock = block) -> torch.Tensor:
                return layer(
                    hidden_states,
                    latents=latents,
                    self_key_padding_mask=decoder_padding_mask,
                    memory_key_padding_mask=None,
                )

            x = self._maybe_checkpoint(dec_forward, x, enabled=self.cfg.gradient_checkpointing and self.training)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_router_stats: bool = False,
        return_aux_losses: bool = False,
        aux_alpha_override: Optional[float] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        del position_ids, aux_alpha_override
        encoder_input_ids = input_ids
        decoder_input_ids = input_ids if decoder_input_ids is None else decoder_input_ids
        encoder_states = self._embed(encoder_input_ids)
        encoder_padding_mask = self._key_padding_mask(attention_mask, seq_len=encoder_input_ids.size(1))
        encoder_states = self._run_encoder(encoder_states, key_padding_mask=encoder_padding_mask)

        latents, imagination_stats, imagination_aux_losses = self._run_latent_workspace(
            encoder_states=encoder_states,
            encoder_padding_mask=encoder_padding_mask,
        )

        decoder_states = self._embed(decoder_input_ids)
        decoder_padding_mask = self._key_padding_mask(
            decoder_attention_mask if decoder_attention_mask is not None else attention_mask,
            seq_len=decoder_input_ids.size(1),
        )
        decoder_states = self._run_decoder(decoder_states, latents=latents, decoder_padding_mask=decoder_padding_mask)
        logits = self.lm_head(self.final_norm(decoder_states))

        out = {"logits": logits}
        if return_router_stats:
            out["router_stats"] = {
                "architecture_kind": self.cfg.architecture_kind,
                "imagination_stats": imagination_stats,
            }
            out["imagination_stats"] = imagination_stats
        if return_aux_losses:
            out["aux_losses"] = imagination_aux_losses
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            out["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out
