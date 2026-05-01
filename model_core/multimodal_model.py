from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_core.config import ModelConfig
from model_core.norms import RMSNorm
from model_core.three_phase_model import DecoderBlock, EncoderBlock


TEXT_MODALITY = 0
IMAGE_MODALITY = 1
AUDIO_MODALITY = 2
VIDEO_MODALITY = 3


class MultimodalPCAModel(nn.Module):
    """
    Multimodal PCA scaffold.

    This is a runnable architecture boundary, not a finished foundation model:
    image/audio/video heads currently reconstruct patch features. A production
    system should replace those heads with learned discrete codecs or diffusion
    decoders trained on aligned data.
    """

    def __init__(self, cfg: ModelConfig, tie_embeddings: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.modality_embed = nn.Embedding(cfg.modality_vocab_size, cfg.d_model)

        self.image_encoder = nn.Conv2d(
            cfg.image_channels,
            cfg.d_model,
            kernel_size=cfg.image_patch_size,
            stride=cfg.image_patch_size,
            bias=False,
        )
        self.audio_encoder = nn.Conv1d(
            cfg.audio_channels,
            cfg.d_model,
            kernel_size=cfg.audio_patch_size,
            stride=cfg.audio_patch_size,
            bias=False,
        )
        self.video_frame_encoder = nn.Conv2d(
            cfg.video_channels,
            cfg.d_model,
            kernel_size=cfg.video_patch_size,
            stride=cfg.video_patch_size,
            bias=False,
        )
        self.video_time_embed = nn.Embedding(cfg.max_video_frames, cfg.d_model)

        self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(int(cfg.encoder_layers))])
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(int(cfg.decoder_layers))])

        self.latent_seed = nn.Parameter(torch.randn(1, cfg.imagination_num_latents, cfg.d_model) * 0.02)
        self.source_to_latent = nn.MultiheadAttention(
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
        self.latent_ffn = nn.Sequential(
            nn.Linear(cfg.d_model, int(cfg.imagination_ffn_dim), bias=False),
            nn.SiLU(),
            nn.Linear(int(cfg.imagination_ffn_dim), cfg.d_model, bias=False),
        )
        self.self_attn_gate = nn.Parameter(torch.tensor(float(cfg.imagination_update_scale)))
        self.ffn_gate = nn.Parameter(torch.tensor(float(cfg.imagination_update_scale)))

        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.image_head = nn.Linear(cfg.d_model, cfg.image_channels * cfg.image_patch_size * cfg.image_patch_size)
        self.audio_head = nn.Linear(cfg.d_model, cfg.audio_channels * cfg.audio_patch_size)
        self.video_head = nn.Linear(cfg.d_model, cfg.video_channels * cfg.video_patch_size * cfg.video_patch_size)

        if tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _add_positions_and_modality(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"modality sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
        modality = torch.full((x.size(0), seq_len), modality_id, device=x.device, dtype=torch.long)
        return x + self.position_embed(pos) + self.modality_embed(modality)

    def _encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._add_positions_and_modality(self.embed_tokens(input_ids), TEXT_MODALITY)

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.image_encoder(pixel_values).flatten(2).transpose(1, 2)
        if patches.size(1) > self.cfg.multimodal_max_image_patches:
            patches = patches[:, : self.cfg.multimodal_max_image_patches, :]
        return self._add_positions_and_modality(patches, IMAGE_MODALITY)

    def _encode_audio(self, audio_values: torch.Tensor) -> torch.Tensor:
        patches = self.audio_encoder(audio_values).transpose(1, 2)
        if patches.size(1) > self.cfg.multimodal_max_audio_patches:
            patches = patches[:, : self.cfg.multimodal_max_audio_patches, :]
        return self._add_positions_and_modality(patches, AUDIO_MODALITY)

    def _encode_video(self, video_values: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video_values.shape
        if frames > self.cfg.max_video_frames:
            video_values = video_values[:, : self.cfg.max_video_frames]
            frames = self.cfg.max_video_frames
        flat = video_values.reshape(batch * frames, channels, height, width)
        patches = self.video_frame_encoder(flat).flatten(2).transpose(1, 2)
        patches = patches.reshape(batch, frames, patches.size(1), patches.size(2))
        time_ids = torch.arange(frames, device=video_values.device, dtype=torch.long)
        patches = patches + self.video_time_embed(time_ids).view(1, frames, 1, -1)
        patches = patches.reshape(batch, frames * patches.size(2), patches.size(3))
        if patches.size(1) > self.cfg.multimodal_max_video_patches:
            patches = patches[:, : self.cfg.multimodal_max_video_patches, :]
        return self._add_positions_and_modality(patches, VIDEO_MODALITY)

    def _encode_sources(
        self,
        input_ids: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        audio_values: Optional[torch.Tensor],
        video_values: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pieces = []
        if input_ids is not None:
            pieces.append(self._encode_text(input_ids))
        if pixel_values is not None:
            pieces.append(self._encode_image(pixel_values))
        if audio_values is not None:
            pieces.append(self._encode_audio(audio_values))
        if video_values is not None:
            pieces.append(self._encode_video(video_values))
        if not pieces:
            raise ValueError("at least one modality input must be provided")
        return torch.cat(pieces, dim=1)

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.encoder:
            x = block(x, key_padding_mask=None, causal=False)
        return x

    def _run_latents(self, source_states: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
        latents = self.latent_seed.expand(source_states.size(0), -1, -1)
        latents = latents + self.source_to_latent(latents, source_states, source_states, need_weights=False)[0]
        anchor = latents
        step_delta_squares = []
        step_delta_norms = []
        for step_idx in range(max(self.cfg.imagination_steps, 1)):
            prev = latents
            scale = self.cfg.imagination_step_decay ** step_idx
            z_norm = self.latent_norm_1(latents)
            attn_out = self.latent_self_attn(z_norm, z_norm, z_norm, need_weights=False)[0]
            latents = latents + (torch.sigmoid(self.self_attn_gate) * scale) * attn_out
            ffn_out = self.latent_ffn(self.latent_norm_2(latents))
            latents = latents + (torch.sigmoid(self.ffn_gate) * scale) * ffn_out
            latents = (1.0 - self.cfg.imagination_anchor_alpha) * latents + self.cfg.imagination_anchor_alpha * anchor
            delta = latents - prev
            step_delta_norms.append(delta.norm(dim=-1).mean())
            step_delta_squares.append(delta.float().pow(2).mean())

        stability_loss = torch.stack(step_delta_squares).mean() if step_delta_squares else latents.new_zeros(())
        latent_rms = torch.sqrt(latents.float().pow(2).mean() + 1e-8)
        norm_loss = (latent_rms - float(self.cfg.imagination_target_rms)) ** 2
        contraction_terms = [
            torch.relu(step_delta_squares[i] - step_delta_squares[i - 1])
            for i in range(1, len(step_delta_squares))
        ]
        contraction_loss = torch.stack(contraction_terms).mean() if contraction_terms else latents.new_zeros(())
        imagination_aux_loss = (
            self.cfg.imagination_stability_alpha * stability_loss
            + self.cfg.imagination_norm_alpha * norm_loss
            + self.cfg.imagination_contraction_alpha * contraction_loss
        ) * self.cfg.imagination_aux_alpha
        stats = {
            "architecture_kind": self.cfg.architecture_kind,
            "num_latents": float(self.cfg.imagination_num_latents),
            "num_steps": float(self.cfg.imagination_steps),
            "avg_delta_norm": float(torch.stack(step_delta_norms).mean().detach().item()) if step_delta_norms else 0.0,
            "final_latent_norm": float(latents.norm(dim=-1).mean().detach().item()),
            "latent_rms": float(latent_rms.detach().item()),
        }
        aux = {
            "total_aux_loss": imagination_aux_loss.to(dtype=latents.dtype),
            "moe_aux_loss": latents.new_zeros(()),
            "moe_load_balance_loss": latents.new_zeros(()),
            "moe_entropy_reg_loss": latents.new_zeros(()),
            "imagination_aux_loss": imagination_aux_loss.to(dtype=latents.dtype),
            "imagination_stability_loss": stability_loss.to(dtype=latents.dtype),
            "imagination_consistency_loss": latents.new_zeros(()),
            "imagination_norm_loss": norm_loss.to(dtype=latents.dtype),
            "imagination_contraction_loss": contraction_loss.to(dtype=latents.dtype),
        }
        return latents, stats, aux

    def _decode_text(self, decoder_input_ids: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self._encode_text(decoder_input_ids)
        for block in self.decoder:
            x = block(
                x,
                latents=latents,
                self_key_padding_mask=None,
                memory_key_padding_mask=None,
                cross_causal=False,
            )
        return self.lm_head(self.final_norm(x))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_router_stats: bool = False,
        return_aux_losses: bool = False,
        aux_alpha_override: Optional[float] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
        output_modalities: tuple[str, ...] = ("text",),
    ) -> dict:
        del attention_mask, position_ids, aux_alpha_override, decoder_attention_mask
        source_states = self._encode_sources(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            video_values=video_values,
        )
        source_states = self._run_encoder(source_states)
        latents, stats, aux_losses = self._run_latents(source_states)

        out = {}
        if "text" in output_modalities:
            text_ids = decoder_input_ids if decoder_input_ids is not None else input_ids
            if text_ids is None:
                raise ValueError("text output requires input_ids or decoder_input_ids")
            logits = self._decode_text(text_ids, latents)
            out["logits"] = logits
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                out["loss"] = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        if "image" in output_modalities:
            out["image_patch_logits"] = self.image_head(latents)
        if "audio" in output_modalities:
            out["audio_patch_logits"] = self.audio_head(latents)
        if "video" in output_modalities:
            out["video_patch_logits"] = self.video_head(latents)

        if "loss" not in out:
            out["loss"] = latents.new_zeros(())
        if return_router_stats:
            out["router_stats"] = stats
            out["imagination_stats"] = stats
        if return_aux_losses:
            out["aux_losses"] = aux_losses
        return out
