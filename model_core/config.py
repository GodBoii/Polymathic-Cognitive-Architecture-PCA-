from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def _is_multiple(value: int, factor: int) -> bool:
    return value % factor == 0


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 8192

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

    def __post_init__(self) -> None:
        if self.ffn_dim is None:
            # SwiGLU usually uses a slightly smaller expansion than 4x MLP.
            approx = int((8 * self.d_model) / 3)
            self.ffn_dim = ((approx + self.ffn_multiple_of - 1) // self.ffn_multiple_of) * self.ffn_multiple_of
        if self.cognitive_ffn_dim is None:
            self.cognitive_ffn_dim = int(self.ffn_dim)
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
        if self.mla_latent_dim <= 0:
            raise ValueError("mla_latent_dim must be > 0")

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
