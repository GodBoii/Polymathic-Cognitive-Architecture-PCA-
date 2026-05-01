import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core import build_model
from model_core.config import ModelConfig


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    cfg = ModelConfig(
        architecture_kind="multimodal_pca",
        vocab_size=512,
        d_model=64,
        n_layers=4,
        encoder_layers=2,
        decoder_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        ffn_dim=128,
        imagination_num_latents=6,
        imagination_steps=2,
        imagination_heads=4,
        imagination_ffn_dim=128,
        image_patch_size=8,
        audio_patch_size=32,
        video_patch_size=8,
        max_video_frames=3,
    )
    model = build_model(cfg).to(device)
    model.train()

    batch = 2
    text = torch.randint(0, cfg.vocab_size, (batch, 12), device=device)
    image = torch.randn(batch, cfg.image_channels, 32, 32, device=device)
    audio = torch.randn(batch, cfg.audio_channels, 256, device=device)
    video = torch.randn(batch, 3, cfg.video_channels, 32, 32, device=device)

    out = model(
        input_ids=text,
        labels=text,
        pixel_values=image,
        audio_values=audio,
        video_values=video,
        output_modalities=("text", "image", "audio", "video"),
        return_aux_losses=True,
        return_router_stats=True,
    )
    total_loss = out["loss"] + out["aux_losses"]["total_aux_loss"]
    total_loss.backward()

    report = {
        "device": device,
        "text_logits_shape": list(out["logits"].shape),
        "image_patch_logits_shape": list(out["image_patch_logits"].shape),
        "audio_patch_logits_shape": list(out["audio_patch_logits"].shape),
        "video_patch_logits_shape": list(out["video_patch_logits"].shape),
        "loss": float(out["loss"].detach().item()),
        "aux_loss": float(out["aux_losses"]["total_aux_loss"].detach().item()),
        "imagination_stats": out["imagination_stats"],
        "passed": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
