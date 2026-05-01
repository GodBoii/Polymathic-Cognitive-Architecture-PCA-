import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.multimodal_dataset_loader import create_multimodal_dataloader
from model_core import build_model
from model_core.config import ModelConfig
from train.muon import Muon
from train.train_step import autocast_context, parameter_counts, resolve_runtime_device_and_precision, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small multimodal PCA training harness.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("train/checkpoints_multimodal"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--audio-samples", type=int, default=16000)
    parser.add_argument("--video-frames", type=int, default=4)
    parser.add_argument("--video-frame-size", type=int, default=128)
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)

    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--imagination-num-latents", type=int, default=8)
    parser.add_argument("--imagination-steps", type=int, default=2)
    parser.add_argument("--imagination-heads", type=int, default=8)
    parser.add_argument("--imagination-ffn-dim", type=int, default=1024)
    parser.add_argument("--image-patch-size", type=int, default=16)
    parser.add_argument("--audio-patch-size", type=int, default=320)
    parser.add_argument("--video-patch-size", type=int, default=16)
    parser.add_argument("--max-video-frames", type=int, default=4)
    return parser.parse_args()


def to_device(batch: dict, device: str) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def build_cfg(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        architecture_kind="multimodal_pca",
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
        ffn_dim=args.ffn_dim,
        imagination_num_latents=args.imagination_num_latents,
        imagination_steps=args.imagination_steps,
        imagination_heads=args.imagination_heads,
        imagination_ffn_dim=args.imagination_ffn_dim,
        image_patch_size=args.image_patch_size,
        audio_patch_size=args.audio_patch_size,
        video_patch_size=args.video_patch_size,
        max_video_frames=args.max_video_frames,
    )


def main() -> None:
    args = parse_args()
    args.device, args.precision = resolve_runtime_device_and_precision(args.device, args.precision)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device

    cfg = build_cfg(args)
    model = build_model(cfg).to(device)
    trainable, total = parameter_counts(model)
    print(json.dumps({"event": "trainable_params", "trainable": trainable, "total": total}))

    loader = create_multimodal_dataloader(
        manifest_path=args.manifest,
        tokenizer_model=args.tokenizer_model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        image_size=args.image_size,
        audio_samples=args.audio_samples,
        video_frames=args.video_frames,
        video_frame_size=args.video_frame_size,
        shuffle=True,
        num_workers=args.data_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    data_iter = iter(loader)
    optimizer = Muon(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and args.precision == "fp16")
    total_tokens = 0
    start = time.perf_counter()

    model.train()
    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device=device, precision=args.precision):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                labels=batch["labels"],
                pixel_values=batch.get("pixel_values"),
                audio_values=batch.get("audio_values"),
                video_values=batch.get("video_values"),
                return_aux_losses=True,
                return_router_stats=True,
            )
            loss = out["loss"] + out["aux_losses"]["total_aux_loss"]

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        if torch.isfinite(grad_norm):
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            print(json.dumps({"event": "non_finite_gradients", "step": step + 1, "grad_norm": float(grad_norm.item())}))

        total_tokens += int(batch["labels"].ne(-100).sum().item())
        if (step + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach().item()),
                        "main_loss": float(out["loss"].detach().item()),
                        "aux_loss": float(out["aux_losses"]["total_aux_loss"].detach().item()),
                        "grad_norm": float(grad_norm.detach().item()),
                        "tokens_per_sec": total_tokens / max(elapsed, 1e-9),
                        "imagination_stats": out.get("imagination_stats", {}),
                    }
                )
            )

        if (step + 1) % args.save_every == 0:
            save_checkpoint(
                path=args.checkpoint_dir / f"step_{step + 1}.pt",
                model=model,
                optimizers=[optimizer],
                schedulers=[],
                scaler=scaler,
                step=step + 1,
                total_tokens=total_tokens,
                micro_batches_seen=step + 1,
                cfg=cfg,
                current_aux_alpha=0.0,
                base_aux_alpha=0.0,
            )

    final = args.checkpoint_dir / "last.pt"
    save_checkpoint(
        path=final,
        model=model,
        optimizers=[optimizer],
        schedulers=[],
        scaler=scaler,
        step=args.steps,
        total_tokens=total_tokens,
        micro_batches_seen=args.steps,
        cfg=cfg,
        current_aux_alpha=0.0,
        base_aux_alpha=0.0,
    )
    print(json.dumps({"status": "done", "steps": args.steps, "checkpoint": str(final)}))


if __name__ == "__main__":
    main()
