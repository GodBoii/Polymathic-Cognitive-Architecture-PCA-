from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def _is_multiple(value: int, factor: int) -> bool:
    return value % factor == 0


@dataclass
class ModelConfig:
    architecture_kind: str = "autoregressive_pca"
    vocab_size: int = 32000
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 8192
    encoder_layers: Optional[int] = None
    decoder_layers: Optional[int] = None

    ffn_dim: Optional[int] = None
    ffn_kind: str = "swiglu"
    reasoning_start_layer: Optional[int] = None
    reasoning_ffn_dim: Optional[int] = None
    reasoning_ffn_kind: Optional[str] = None
    reasoning_moe_num_experts: int = 16
    reasoning_moe_top_k: int = 2
    reasoning_moe_num_groups: int = 4
    reasoning_moe_groups_top_k: int = 1
    reasoning_moe_gate_type: str = "sigmoid"
    reasoning_moe_expert_ffn_kind: str = "swiglu"
    ffn_multiple_of: int = 256
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3
    unk_token_id: int = 0

    cognitive_loops: int = 5
    cognitive_num_experts: int = 8
    cognitive_top_k: int = 4
    cognitive_num_groups: int = 1
    cognitive_groups_top_k: int = 1
    cognitive_gate_type: str = "sigmoid"
    cognitive_aux_alpha: float = 0.01
    cognitive_entropy_alpha: float = 0.001
    cognitive_ffn_kind: str = "swiglu"
    cognitive_ffn_dim: Optional[int] = None
    gqa_layers: int = 4
    lightning_end_layer: int = 16
    mla_latent_dim: int = 512
    use_imagination: bool = False
    imagination_num_latents: int = 8
    imagination_steps: int = 2
    imagination_heads: Optional[int] = None
    imagination_ffn_dim: Optional[int] = None
    imagination_anchor_alpha: float = 0.1
    imagination_aux_alpha: float = 0.05
    imagination_stability_alpha: float = 0.025
    imagination_consistency_alpha: float = 0.025
    imagination_norm_alpha: float = 0.01
    imagination_update_scale: float = 0.25
    imagination_step_decay: float = 0.85
    imagination_contraction_alpha: float = 0.02
    imagination_target_rms: float = 1.0
    imagination_rms_align_alpha: float = 0.1
    three_phase_encoder_causal: bool = True
    three_phase_causal_latent_workspace: bool = True
    three_phase_share_encoder_decoder: bool = True

    def __post_init__(self) -> None:
        if self.ffn_dim is None:
            # SwiGLU usually uses a slightly smaller expansion than 4x MLP.
            approx = int((8 * self.d_model) / 3)
            self.ffn_dim = ((approx + self.ffn_multiple_of - 1) // self.ffn_multiple_of) * self.ffn_multiple_of
        if self.cognitive_ffn_dim is None:
            self.cognitive_ffn_dim = int(self.ffn_dim)
        if self.imagination_ffn_dim is None:
            self.imagination_ffn_dim = int(self.ffn_dim)
        if self.imagination_heads is None:
            self.imagination_heads = self.n_heads
        if self.encoder_layers is None or self.decoder_layers is None:
            encoder_layers = max(1, self.n_layers // 2)
            decoder_layers = max(1, self.n_layers - encoder_layers)
            self.encoder_layers = encoder_layers if self.encoder_layers is None else self.encoder_layers
            self.decoder_layers = decoder_layers if self.decoder_layers is None else self.decoder_layers
        if self.architecture_kind == "three_phase":
            # Hybrid-attention boundaries are irrelevant for the dedicated three-phase path.
            self.gqa_layers = min(self.gqa_layers, self.n_layers)
            self.lightning_end_layer = min(self.lightning_end_layer, self.n_layers)
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_repeat_factor(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def experts_per_group(self) -> int:
        return self.cognitive_num_experts // self.cognitive_num_groups

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.architecture_kind not in {"autoregressive_pca", "three_phase"}:
            raise ValueError("architecture_kind must be 'autoregressive_pca' or 'three_phase'")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0")
        if not _is_multiple(self.d_model, self.n_heads):
            raise ValueError("d_model must be divisible by n_heads")
        if not _is_multiple(self.n_heads, self.n_kv_heads):
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.encoder_layers is None or self.encoder_layers <= 0:
            raise ValueError("encoder_layers must be > 0")
        if self.decoder_layers is None or self.decoder_layers <= 0:
            raise ValueError("decoder_layers must be > 0")
        if self.architecture_kind == "three_phase" and (self.encoder_layers + self.decoder_layers) != self.n_layers:
            raise ValueError("for three_phase, encoder_layers + decoder_layers must equal n_layers")
        if self.ffn_multiple_of <= 0:
            raise ValueError("ffn_multiple_of must be > 0")
        if self.ffn_dim is not None and self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be > 0 when provided")
        if self.ffn_kind not in {"swiglu", "standard", "moe"}:
            raise ValueError("ffn_kind must be 'swiglu', 'standard', or 'moe'")
        if self.reasoning_start_layer is not None:
            if self.reasoning_start_layer < 0 or self.reasoning_start_layer > self.n_layers:
                raise ValueError("reasoning_start_layer must be in [0, n_layers] when provided")
        if self.reasoning_ffn_dim is not None and self.reasoning_ffn_dim <= 0:
            raise ValueError("reasoning_ffn_dim must be > 0 when provided")
        if self.reasoning_ffn_kind is not None and self.reasoning_ffn_kind not in {"swiglu", "standard", "moe"}:
            raise ValueError("reasoning_ffn_kind must be 'swiglu', 'standard', or 'moe' when provided")
        if self.reasoning_moe_num_experts <= 0:
            raise ValueError("reasoning_moe_num_experts must be > 0")
        if self.reasoning_moe_top_k <= 0:
            raise ValueError("reasoning_moe_top_k must be > 0")
        if self.reasoning_moe_num_groups <= 0:
            raise ValueError("reasoning_moe_num_groups must be > 0")
        if self.reasoning_moe_num_experts % self.reasoning_moe_num_groups != 0:
            raise ValueError("reasoning_moe_num_experts must be divisible by reasoning_moe_num_groups")
        if self.reasoning_moe_groups_top_k <= 0:
            raise ValueError("reasoning_moe_groups_top_k must be > 0")
        if self.reasoning_moe_groups_top_k > self.reasoning_moe_num_groups:
            raise ValueError("reasoning_moe_groups_top_k must be <= reasoning_moe_num_groups")
        if self.reasoning_moe_top_k > (self.reasoning_moe_num_experts // self.reasoning_moe_num_groups):
            raise ValueError("reasoning_moe_top_k must be <= experts_per_reasoning_group")
        if self.reasoning_moe_gate_type not in {"softmax", "sigmoid"}:
            raise ValueError("reasoning_moe_gate_type must be 'softmax' or 'sigmoid'")
        if self.reasoning_moe_expert_ffn_kind not in {"swiglu", "standard"}:
            raise ValueError("reasoning_moe_expert_ffn_kind must be 'swiglu' or 'standard'")
        if self.rms_eps <= 0.0:
            raise ValueError("rms_eps must be > 0")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be a bool")
        if self.cognitive_loops <= 0:
            raise ValueError("cognitive_loops must be > 0")
        if self.cognitive_num_experts <= 0:
            raise ValueError("cognitive_num_experts must be > 0")
        if self.cognitive_num_groups <= 0:
            raise ValueError("cognitive_num_groups must be > 0")
        if self.cognitive_num_experts % self.cognitive_num_groups != 0:
            raise ValueError("cognitive_num_experts must be divisible by cognitive_num_groups")
        if self.cognitive_top_k <= 0:
            raise ValueError("cognitive_top_k must be > 0")
        experts_per_group = self.cognitive_num_experts // self.cognitive_num_groups
        if self.cognitive_top_k > experts_per_group:
            raise ValueError("cognitive_top_k must be <= experts per group")
        if self.cognitive_groups_top_k <= 0:
            raise ValueError("cognitive_groups_top_k must be > 0")
        if self.cognitive_groups_top_k > self.cognitive_num_groups:
            raise ValueError("cognitive_groups_top_k must be <= cognitive_num_groups")
        if self.cognitive_gate_type not in {"softmax", "sigmoid"}:
            raise ValueError("cognitive_gate_type must be 'softmax' or 'sigmoid'")
        if self.cognitive_aux_alpha < 0.0:
            raise ValueError("cognitive_aux_alpha must be >= 0")
        if self.cognitive_entropy_alpha < 0.0:
            raise ValueError("cognitive_entropy_alpha must be >= 0")
        if self.cognitive_ffn_kind not in {"swiglu", "standard"}:
            raise ValueError("cognitive_ffn_kind must be 'swiglu' or 'standard'")
        if self.cognitive_ffn_dim is not None and self.cognitive_ffn_dim <= 0:
            raise ValueError("cognitive_ffn_dim must be > 0 when provided")
        if self.gqa_layers < 0:
            raise ValueError("gqa_layers must be >= 0")
        if self.lightning_end_layer < 0:
            raise ValueError("lightning_end_layer must be >= 0")
        if self.gqa_layers > self.lightning_end_layer:
            raise ValueError("gqa_layers must be <= lightning_end_layer")
        if self.lightning_end_layer > self.n_layers:
            raise ValueError("lightning_end_layer must be <= n_layers")
        if self.mla_latent_dim <= 0:
            raise ValueError("mla_latent_dim must be > 0")
        if self.imagination_num_latents <= 0:
            raise ValueError("imagination_num_latents must be > 0")
        if self.imagination_steps < 0:
            raise ValueError("imagination_steps must be >= 0")
        if self.imagination_heads is None or self.imagination_heads <= 0:
            raise ValueError("imagination_heads must be > 0")
        if not _is_multiple(self.d_model, self.imagination_heads):
            raise ValueError("d_model must be divisible by imagination_heads")
        if self.imagination_ffn_dim is not None and self.imagination_ffn_dim <= 0:
            raise ValueError("imagination_ffn_dim must be > 0 when provided")
        if self.imagination_anchor_alpha < 0.0 or self.imagination_anchor_alpha > 1.0:
            raise ValueError("imagination_anchor_alpha must be in [0.0, 1.0]")
        if self.imagination_aux_alpha < 0.0:
            raise ValueError("imagination_aux_alpha must be >= 0.0")
        if self.imagination_stability_alpha < 0.0:
            raise ValueError("imagination_stability_alpha must be >= 0.0")
        if self.imagination_consistency_alpha < 0.0:
            raise ValueError("imagination_consistency_alpha must be >= 0.0")
        if self.imagination_norm_alpha < 0.0:
            raise ValueError("imagination_norm_alpha must be >= 0.0")
        if self.imagination_update_scale <= 0.0 or self.imagination_update_scale > 1.0:
            raise ValueError("imagination_update_scale must be in (0.0, 1.0]")
        if self.imagination_step_decay <= 0.0 or self.imagination_step_decay > 1.0:
            raise ValueError("imagination_step_decay must be in (0.0, 1.0]")
        if self.imagination_contraction_alpha < 0.0:
            raise ValueError("imagination_contraction_alpha must be >= 0.0")
        if self.imagination_target_rms <= 0.0:
            raise ValueError("imagination_target_rms must be > 0.0")
        if self.imagination_rms_align_alpha < 0.0 or self.imagination_rms_align_alpha > 1.0:
            raise ValueError("imagination_rms_align_alpha must be in [0.0, 1.0]")
        if not isinstance(self.three_phase_encoder_causal, bool):
            raise ValueError("three_phase_encoder_causal must be a bool")
        if not isinstance(self.three_phase_causal_latent_workspace, bool):
            raise ValueError("three_phase_causal_latent_workspace must be a bool")
        if not isinstance(self.three_phase_share_encoder_decoder, bool):
            raise ValueError("three_phase_share_encoder_decoder must be a bool")

    def ffn_kind_for_layer(self, layer_idx: int) -> str:
        if self.reasoning_start_layer is not None and layer_idx >= self.reasoning_start_layer:
            return self.reasoning_ffn_kind or self.ffn_kind
        return self.ffn_kind

    def ffn_dim_for_layer(self, layer_idx: int) -> int:
        if self.reasoning_start_layer is not None and layer_idx >= self.reasoning_start_layer:
            return self.reasoning_ffn_dim or int(self.ffn_dim)
        return int(self.ffn_dim)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["head_dim"] = self.head_dim
        payload["kv_repeat_factor"] = self.kv_repeat_factor
        payload["experts_per_group"] = self.experts_per_group
        return payload
