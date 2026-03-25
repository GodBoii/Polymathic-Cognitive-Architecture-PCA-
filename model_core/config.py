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
    ffn_multiple_of: int = 256
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0

    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3
    unk_token_id: int = 0

    cognitive_loops: int = 5
    gqa_layers: int = 4
    lightning_end_layer: int = 16
    mla_latent_dim: int = 512

    def __post_init__(self) -> None:
        self.validate()
        if self.ffn_dim is None:
            # SwiGLU usually uses a slightly smaller expansion than 4x MLP.
            approx = int((8 * self.d_model) / 3)
            self.ffn_dim = ((approx + self.ffn_multiple_of - 1) // self.ffn_multiple_of) * self.ffn_multiple_of

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_repeat_factor(self) -> int:
        return self.n_heads // self.n_kv_heads

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
        if self.rms_eps <= 0.0:
            raise ValueError("rms_eps must be > 0")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.cognitive_loops <= 0:
            raise ValueError("cognitive_loops must be > 0")
        if self.gqa_layers < 0:
            raise ValueError("gqa_layers must be >= 0")
        if self.lightning_end_layer < 0:
            raise ValueError("lightning_end_layer must be >= 0")
        if self.gqa_layers > self.lightning_end_layer:
            raise ValueError("gqa_layers must be <= lightning_end_layer")
        if self.mla_latent_dim <= 0:
            raise ValueError("mla_latent_dim must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
