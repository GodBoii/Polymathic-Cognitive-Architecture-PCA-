import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_modules.gqa_rope import GQAAttention
from model_core import ModelConfig


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig()
    attn = GQAAttention(cfg).to(device)

    x = torch.randn(2, 128, cfg.d_model, device=device)
    out, debug = attn(x, return_debug=True)

    report = {
        "device": device,
        "input_shape": list(x.shape),
        "output_shape": list(out.shape),
        "output_finite": bool(torch.isfinite(out).all().item()),
        "head_dim": cfg.head_dim,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "kv_repeat_factor": cfg.kv_repeat_factor,
        "debug": debug,
        "output_mean_abs": float(out.abs().mean().item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
