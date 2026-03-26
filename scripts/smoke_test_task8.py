import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core.config import ModelConfig
from model_core.model import PCAModel


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=32000,
        d_model=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=512,
        cognitive_loops=3,
        cognitive_num_experts=8,
        cognitive_top_k=4,
        cognitive_gate_type="sigmoid",
        cognitive_aux_alpha=0.01,
        cognitive_entropy_alpha=0.001,
        gqa_layers=2,
        lightning_end_layer=4,
        mla_latent_dim=128,
    )
    model = PCAModel(cfg).to(device).eval()

    with torch.no_grad():
        input_ids = torch.randint(0, cfg.vocab_size, (2, 64), device=device)
        attention_mask = torch.ones(2, 64, device=device)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_router_stats=True,
            return_aux_losses=True,
        )

    report = {
        "device": device,
        "logits_shape": list(out["logits"].shape),
        "logits_finite": bool(torch.isfinite(out["logits"]).all().item()),
        "aux_losses": {
            "moe_aux_loss": float(out["aux_losses"]["moe_aux_loss"].item()),
            "moe_load_balance_loss": float(out["aux_losses"]["moe_load_balance_loss"].item()),
            "moe_entropy_reg_loss": float(out["aux_losses"]["moe_entropy_reg_loss"].item()),
        },
        "router_stats": to_jsonable(out["router_stats"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
