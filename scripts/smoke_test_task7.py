import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core.config import ModelConfig
from model_core.model import PCAModel


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=32000,
        d_model=256,
        n_layers=18,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=512,
        gqa_layers=4,
        lightning_end_layer=12,
        mla_latent_dim=128,
    )
    model = PCAModel(cfg).to(device).eval()

    layout = [layer.attention_kind for layer in model.layers]
    counts = Counter(layout)

    with torch.no_grad():
        input_ids = torch.randint(0, cfg.vocab_size, (2, 64), device=device)
        attention_mask = torch.ones(2, 64, device=device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    report = {
        "device": device,
        "n_layers": cfg.n_layers,
        "attention_layout": layout,
        "attention_counts": dict(counts),
        "output_shape": list(out["logits"].shape),
        "output_finite": bool(torch.isfinite(out["logits"]).all().item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
