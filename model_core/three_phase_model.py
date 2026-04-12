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


def _build_cross_causal_bool_mask(query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
    q_idx = torch.arange(query_len, device=device, dtype=torch.long).unsqueeze(1)
    k_idx = torch.arange(key_len, device=device, dtype=torch.long).unsqueeze(0)
    return k_idx > q_idx


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

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        causal: bool = False,
    ) -> torch.Tensor:
        norm_x = self.attn_norm(x)
        attn_mask = _build_causal_bool_mask(norm_x.size(1), device=norm_x.device) if causal else None
        attn_out = self.attn(
            norm_x,
            norm_x,
            norm_x,
            attn_mask=attn_mask,
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
        cross_causal: bool = False,
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
        cross_mask = (
            _build_cross_causal_bool_mask(
                query_len=x.size(1),
                key_len=latents.size(1),
                device=x.device,
            )
            if cross_causal
            else None
        )
        cross_out = self.cross_attn(
            self.cross_norm(x),
            latents,
            latents,
            attn_mask=cross_mask,
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
                return layer(
                    hidden_states,
                    key_padding_mask=key_padding_mask,
                    causal=self.cfg.three_phase_encoder_causal,
                )

            x = self._maybe_checkpoint(enc_forward, x, enabled=self.cfg.gradient_checkpointing and self.training)
        return x

    def _refine_latents(
        self,
        latents: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
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
            step_delta_norms.append(step_delta.norm(dim=-1).mean())
            step_delta_squares.append(step_delta.float().pow(2).mean())
        return latents, step_delta_norms, step_delta_squares

    def _build_imagination_outputs(
        self,
        latent_memory: torch.Tensor,
        final_latents: torch.Tensor,
        step_delta_norms: list[torch.Tensor],
        step_delta_squares: list[torch.Tensor],
        encoder_states: torch.Tensor,
    ) -> tuple[dict, dict]:
        summary = encoder_states.mean(dim=1)
        summary_target = self.summary_proj(summary)
        latent_summary = latent_memory.mean(dim=1)
        latent_summary_norm = F.normalize(latent_summary.float(), dim=-1, eps=1e-6)
        summary_target_norm = F.normalize(summary_target.float(), dim=-1, eps=1e-6)
        stability_loss = torch.stack(step_delta_squares).mean() if step_delta_squares else latent_memory.new_zeros(())
        consistency_loss = F.mse_loss(latent_summary_norm, summary_target_norm)
        latent_rms_sq = latent_memory.float().pow(2).mean()
        latent_rms = torch.sqrt(latent_rms_sq + 1e-8)
        target_rms = float(self.cfg.imagination_target_rms)
        norm_loss = (latent_rms - target_rms) ** 2
        contraction_terms: list[torch.Tensor] = []
        for i in range(1, len(step_delta_squares)):
            contraction_terms.append(torch.relu(step_delta_squares[i] - step_delta_squares[i - 1]))
        contraction_loss = torch.stack(contraction_terms).mean() if contraction_terms else latent_memory.new_zeros(())
        imagination_aux_loss = (
            self.cfg.imagination_stability_alpha * stability_loss
            + self.cfg.imagination_consistency_alpha * consistency_loss
            + self.cfg.imagination_norm_alpha * norm_loss
            + self.cfg.imagination_contraction_alpha * contraction_loss
        ) * self.cfg.imagination_aux_alpha

        stats = {
            "num_latents": float(self.cfg.imagination_num_latents),
            "num_steps": float(self.cfg.imagination_steps),
            "avg_delta_norm": float(torch.stack(step_delta_norms).mean().detach().item()) if step_delta_norms else 0.0,
            "final_latent_norm": float(final_latents.norm(dim=-1).mean().detach().item()),
            "self_attn_gate": float(torch.sigmoid(self.self_attn_gate).detach().item()),
            "ffn_gate": float(torch.sigmoid(self.ffn_gate).detach().item()),
            "step_decay": float(self.cfg.imagination_step_decay),
            "step_delta_norms": [float(v.detach().item()) for v in step_delta_norms],
            "convergence_ratio": float(
                (step_delta_norms[-1] / (step_delta_norms[0] + 1e-8)).detach().item()
                if step_delta_norms
                else 1.0
            ),
            "latent_rms": float(latent_rms.detach().item()),
            "target_rms": float(target_rms),
            "delta_monotonic_decrease": bool(
                all(step_delta_norms[i] <= step_delta_norms[i - 1] for i in range(1, len(step_delta_norms)))
            ),
            "workspace_mode": "causal_scan" if self.cfg.three_phase_causal_latent_workspace else "global",
        }
        aux = {
            "total_aux_loss": imagination_aux_loss.to(dtype=latent_memory.dtype),
            "moe_aux_loss": latent_memory.new_zeros(()),
            "moe_load_balance_loss": latent_memory.new_zeros(()),
            "moe_entropy_reg_loss": latent_memory.new_zeros(()),
            "imagination_aux_loss": imagination_aux_loss.to(dtype=latent_memory.dtype),
            "imagination_stability_loss": stability_loss.to(dtype=latent_memory.dtype),
            "imagination_consistency_loss": consistency_loss.to(dtype=latent_memory.dtype),
            "imagination_norm_loss": norm_loss.to(dtype=latent_memory.dtype),
            "imagination_contraction_loss": contraction_loss.to(dtype=latent_memory.dtype),
        }
        return stats, aux

    def _run_latent_workspace(
        self,
        encoder_states: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict, dict]:
        if self.cfg.three_phase_causal_latent_workspace:
            return self._run_latent_workspace_causal(
                encoder_states=encoder_states,
                encoder_padding_mask=encoder_padding_mask,
            )

        latents = self.latent_seed.expand(encoder_states.size(0), -1, -1)
        latents = latents + self.token_to_latent(
            latents,
            encoder_states,
            encoder_states,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )[0]
        anchor = latents
        latents, step_delta_norms, step_delta_squares = self._refine_latents(latents=latents, anchor=anchor)
        stats, aux = self._build_imagination_outputs(
            latent_memory=latents,
            final_latents=latents,
            step_delta_norms=step_delta_norms,
            step_delta_squares=step_delta_squares,
            encoder_states=encoder_states,
        )
        return latents, stats, aux

    def _run_latent_workspace_causal(
        self,
        encoder_states: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict, dict]:
        bsz, seq_len, _ = encoder_states.shape
        latents = self.latent_seed.expand(bsz, -1, -1)
        latent_summaries: list[torch.Tensor] = []
        step_delta_norm_sums = [torch.zeros((), device=encoder_states.device, dtype=encoder_states.dtype) for _ in range(max(self.cfg.imagination_steps, 1))]
        step_delta_square_sums = [torch.zeros((), device=encoder_states.device, dtype=torch.float32) for _ in range(max(self.cfg.imagination_steps, 1))]
        total_step_deltas: list[torch.Tensor] = []
        total_step_squares: list[torch.Tensor] = []
        rms_align_scale_sum = torch.zeros((), device=encoder_states.device, dtype=torch.float32)
        rms_align_scale_count = 0

        for token_idx in range(seq_len):
            token_states = encoder_states[:, token_idx : token_idx + 1, :]
            token_padding_mask = (
                encoder_padding_mask[:, token_idx : token_idx + 1]
                if encoder_padding_mask is not None
                else None
            )
            latents = latents + self.token_to_latent(
                latents,
                token_states,
                token_states,
                key_padding_mask=token_padding_mask,
                need_weights=False,
            )[0]
            latents, step_delta_norms, step_delta_squares = self._refine_latents(latents=latents, anchor=latents)
            if self.cfg.imagination_rms_align_alpha > 0.0:
                latents_float = latents.float()
                curr_rms = torch.sqrt(latents_float.pow(2).mean() + 1e-8)
                target_rms = float(self.cfg.imagination_target_rms)
                scale = target_rms / (curr_rms + 1e-8)
                mixed_scale = (1.0 - self.cfg.imagination_rms_align_alpha) + self.cfg.imagination_rms_align_alpha * scale
                latents = latents * mixed_scale.to(dtype=latents.dtype)
                rms_align_scale_sum = rms_align_scale_sum + mixed_scale.detach().float()
                rms_align_scale_count += 1
            for step_idx, step_norm in enumerate(step_delta_norms):
                step_delta_norm_sums[step_idx] = step_delta_norm_sums[step_idx] + step_norm
                step_delta_square_sums[step_idx] = step_delta_square_sums[step_idx] + step_delta_squares[step_idx].float()
                total_step_deltas.append(step_norm)
                total_step_squares.append(step_delta_squares[step_idx])
            latent_summaries.append(latents.mean(dim=1, keepdim=True))

        latent_memory = torch.cat(latent_summaries, dim=1) if latent_summaries else encoder_states.new_zeros((bsz, 0, encoder_states.size(-1)))
        avg_step_deltas = [v / max(seq_len, 1) for v in step_delta_norm_sums]
        avg_step_squares = [v / max(seq_len, 1) for v in step_delta_square_sums]
        stats, aux = self._build_imagination_outputs(
            latent_memory=latent_memory,
            final_latents=latents,
            step_delta_norms=avg_step_deltas if avg_step_deltas else total_step_deltas,
            step_delta_squares=avg_step_squares if avg_step_squares else total_step_squares,
            encoder_states=encoder_states,
        )
        stats["rms_align_alpha"] = float(self.cfg.imagination_rms_align_alpha)
        stats["avg_rms_align_scale"] = float((rms_align_scale_sum / max(rms_align_scale_count, 1)).item())
        return latent_memory, stats, aux

    def _run_decoder(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        decoder_padding_mask: Optional[torch.Tensor],
        memory_padding_mask: Optional[torch.Tensor],
        cross_causal: bool,
    ) -> torch.Tensor:
        for block in self.decoder:
            def dec_forward(hidden_states: torch.Tensor, layer: DecoderBlock = block) -> torch.Tensor:
                return layer(
                    hidden_states,
                    latents=latents,
                    self_key_padding_mask=decoder_padding_mask,
                    memory_key_padding_mask=memory_padding_mask,
                    cross_causal=cross_causal,
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
        decoder_input_ids = input_ids if decoder_input_ids is None else decoder_input_ids
        encoder_input_ids = decoder_input_ids if self.cfg.three_phase_share_encoder_decoder else input_ids
        encoder_attention_mask = (
            decoder_attention_mask if (self.cfg.three_phase_share_encoder_decoder and decoder_attention_mask is not None) else attention_mask
        )
        encoder_states = self._embed(encoder_input_ids)
        encoder_padding_mask = self._key_padding_mask(encoder_attention_mask, seq_len=encoder_input_ids.size(1))
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
        cross_causal = self.cfg.three_phase_causal_latent_workspace and (latents.size(1) == decoder_states.size(1))
        memory_padding_mask = decoder_padding_mask if cross_causal else None
        decoder_states = self._run_decoder(
            decoder_states,
            latents=latents,
            decoder_padding_mask=decoder_padding_mask,
            memory_padding_mask=memory_padding_mask,
            cross_causal=cross_causal,
        )
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
