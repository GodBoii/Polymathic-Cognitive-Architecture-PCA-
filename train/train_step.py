import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core.config import ModelConfig
from model_core.model import PCAModel
from train.muon import Muon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 6 training harness with Muon + mixed precision.")

    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("train/checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")

    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ffn-multiple-of", type=int, default=256)

    return parser.parse_args()


def make_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        ffn_multiple_of=args.ffn_multiple_of,
    )


def build_optimizer(model: PCAModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "muon":
        return Muon(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )


def autocast_context(device: str, precision: str):
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if precision == "fp32":
        return torch.autocast(device_type=device_type, enabled=False)
    if precision == "fp16":
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


def save_checkpoint(
    path: Path,
    model: PCAModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    total_tokens: int,
    cfg: ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "total_tokens": total_tokens,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_config": cfg.to_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: PCAModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
) -> Tuple[int, int]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload["scaler_state"])
    return int(payload["step"]), int(payload["total_tokens"])


def make_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = args.device
    cfg = make_model_config(args)
    model = PCAModel(cfg).to(device)
    optimizer = build_optimizer(model, args)

    use_scaler = device.startswith("cuda") and args.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_step = 0
    total_tokens = 0
    if args.resume is not None:
        start_step, total_tokens = load_checkpoint(args.resume, model, optimizer, scaler, device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    run_start = time.perf_counter()

    for step in range(start_step, args.steps):
        step_start = time.perf_counter()
        running_loss = 0.0

        for _ in range(args.grad_accum_steps):
            batch = make_batch(
                batch_size=args.micro_batch_size,
                seq_len=args.seq_len,
                vocab_size=cfg.vocab_size,
                device=device,
            )
            with autocast_context(device=device, precision=args.precision):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out["loss"] / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.item()) * args.grad_accum_steps

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_step = args.micro_batch_size * args.seq_len * args.grad_accum_steps
        total_tokens += tokens_step
        dt = time.perf_counter() - step_start
        tokens_per_sec = tokens_step / max(dt, 1e-9)

        if step % args.log_every == 0:
            payload = {
                "step": step + 1,
                "loss": running_loss,
                "grad_norm": float(grad_norm.item()),
                "lr": args.lr,
                "tokens_per_sec": tokens_per_sec,
                "total_tokens": total_tokens,
                "step_time_sec": dt,
                "gpu_mem_gb": (
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
            print(json.dumps(payload))

        if (step + 1) % args.save_every == 0:
            ckpt_path = args.checkpoint_dir / f"step_{step + 1}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step + 1,
                total_tokens=total_tokens,
                cfg=cfg,
            )

    final_ckpt = args.checkpoint_dir / "last.pt"
    save_checkpoint(
        path=final_ckpt,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=args.steps,
        total_tokens=total_tokens,
        cfg=cfg,
    )

    elapsed = time.perf_counter() - run_start
    summary = {
        "status": "done",
        "steps": args.steps,
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed,
        "checkpoint": str(final_ckpt),
        "optimizer": args.optimizer,
        "precision": args.precision,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
