import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core import ModelConfig, RMSNorm, SwiGLUFFN


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig()

    norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps).to(device)
    ffn = SwiGLUFFN(cfg.d_model, cfg.ffn_dim, bias=False, dropout=cfg.dropout).to(device)

    x = torch.randn(2, 128, cfg.d_model, device=device)
    y_norm = norm(x)
    y_ffn = ffn(y_norm)

    report = {
        "device": device,
        "config": cfg.to_dict(),
        "head_dim": cfg.head_dim,
        "kv_repeat_factor": cfg.kv_repeat_factor,
        "x_shape": list(x.shape),
        "norm_shape": list(y_norm.shape),
        "ffn_shape": list(y_ffn.shape),
        "norm_finite": bool(torch.isfinite(y_norm).all().item()),
        "ffn_finite": bool(torch.isfinite(y_ffn).all().item()),
        "norm_mean_abs": float(y_norm.abs().mean().item()),
        "ffn_mean_abs": float(y_ffn.abs().mean().item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
