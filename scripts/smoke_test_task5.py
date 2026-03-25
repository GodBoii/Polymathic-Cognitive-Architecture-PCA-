import json
import sys
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
        d_model=2048,
        n_layers=4,  # reduced for smoke-test speed
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=512,
    )
    model = PCAModel(cfg).to(device)
    model.eval()

    batch, seq = 2, 64
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    attention_mask = torch.ones(batch, seq, device=device)
    labels = input_ids.clone()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    tied = model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    report = {
        "device": device,
        "config": cfg.to_dict(),
        "attention_layout": [layer.attention_kind for layer in model.layers],
        "input_shape": list(input_ids.shape),
        "logits_shape": list(out["logits"].shape),
        "loss": float(out["loss"].item()),
        "logits_finite": bool(torch.isfinite(out["logits"]).all().item()),
        "embeddings_tied": tied,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
