from .config import ModelConfig
from .ffn import MoEFFN, StandardFFN, SwiGLUFFN, build_ffn
from .model import PCAModel
from .norms import RMSNorm
from .three_phase_model import ThreePhasePCAModel


def build_model(cfg: ModelConfig):
    if cfg.architecture_kind == "three_phase":
        return ThreePhasePCAModel(cfg)
    return PCAModel(cfg)


__all__ = [
    "ModelConfig",
    "RMSNorm",
    "SwiGLUFFN",
    "StandardFFN",
    "MoEFFN",
    "build_ffn",
    "PCAModel",
    "ThreePhasePCAModel",
    "build_model",
]
