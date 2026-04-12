import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core import build_model
from model_core.config import ModelConfig


def _run_logits(model, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(
            input_ids=ids,
            decoder_input_ids=ids,
        )
    return out["logits"]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)

    cfg = ModelConfig(
        architecture_kind="three_phase",
        vocab_size=512,
        d_model=64,
        n_layers=4,
        encoder_layers=2,
        decoder_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        ffn_dim=128,
        dropout=0.0,
        gradient_checkpointing=False,
        imagination_num_latents=4,
        imagination_steps=2,
        imagination_heads=4,
        imagination_ffn_dim=128,
        three_phase_encoder_causal=True,
        three_phase_causal_latent_workspace=True,
        three_phase_share_encoder_decoder=True,
    )

    model = build_model(cfg).to(device).eval()

    batch = 1
    seq_len = 16
    prefix_len = 8
    if prefix_len >= seq_len:
        raise RuntimeError("prefix_len must be < seq_len")

    base_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    alt_ids = base_ids.clone()
    alt_ids[:, prefix_len:] = torch.randint(0, cfg.vocab_size, (batch, seq_len - prefix_len), device=device)

    logits_base = _run_logits(model, base_ids)
    logits_alt = _run_logits(model, alt_ids)

    prefix_diff = (logits_base[:, :prefix_len, :] - logits_alt[:, :prefix_len, :]).abs().max().item()
    suffix_diff = (logits_base[:, prefix_len:, :] - logits_alt[:, prefix_len:, :]).abs().max().item()

    passed = prefix_diff < 1e-5
    report = {
        "device": device,
        "prefix_len": prefix_len,
        "seq_len": seq_len,
        "prefix_max_abs_diff": prefix_diff,
        "suffix_max_abs_diff": suffix_diff,
        "passed": passed,
        "config": {
            "three_phase_encoder_causal": cfg.three_phase_encoder_causal,
            "three_phase_causal_latent_workspace": cfg.three_phase_causal_latent_workspace,
            "three_phase_share_encoder_decoder": cfg.three_phase_share_encoder_decoder,
        },
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Causality smoke test failed: prefix logits changed when only suffix tokens were modified.")


if __name__ == "__main__":
    main()
