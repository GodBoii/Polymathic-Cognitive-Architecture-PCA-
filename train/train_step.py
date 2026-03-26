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
from data_pipeline.dataset_loader import create_packed_dataloader
from train.muon import Muon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 6 training harness with Muon + mixed precision.")

    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-router-stats-every", type=int, default=0, help="0 disables router telemetry in logs.")
    parser.add_argument("--collapse-cv-threshold", type=float, default=0.5)
    parser.add_argument("--collapse-entropy-threshold", type=float, default=0.25)
    parser.add_argument("--adaptive-aux-alpha", action="store_true", help="Enable dynamic MoE aux alpha feedback control.")
    parser.add_argument("--aux-alpha-max", type=float, default=0.2)
    parser.add_argument("--aux-alpha-up-mult", type=float, default=1.5)
    parser.add_argument("--aux-alpha-decay", type=float, default=0.95)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("train/checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")

    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--train-manifest", type=Path, default=None, help="Use packed dataset manifest instead of random synthetic batches.")
    parser.add_argument("--data-workers", type=int, default=0)
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
    parser.add_argument("--cognitive-loops", type=int, default=5)
    parser.add_argument("--cognitive-num-experts", type=int, default=8)
    parser.add_argument("--cognitive-top-k", type=int, default=4)
    parser.add_argument("--cognitive-gate-type", choices=["softmax", "sigmoid"], default="sigmoid")
    parser.add_argument("--cognitive-aux-alpha", type=float, default=0.01)
    parser.add_argument("--cognitive-entropy-alpha", type=float, default=0.001)

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
        cognitive_loops=args.cognitive_loops,
        cognitive_num_experts=args.cognitive_num_experts,
        cognitive_top_k=args.cognitive_top_k,
        cognitive_gate_type=args.cognitive_gate_type,
        cognitive_aux_alpha=args.cognitive_aux_alpha,
        cognitive_entropy_alpha=args.cognitive_entropy_alpha,
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
    current_aux_alpha: float,
    base_aux_alpha: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "total_tokens": total_tokens,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_config": cfg.to_dict(),
        "current_aux_alpha": current_aux_alpha,
        "base_aux_alpha": base_aux_alpha,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: PCAModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
) -> Tuple[int, int, float | None, float | None]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload["scaler_state"])
    return (
        int(payload["step"]),
        int(payload["total_tokens"]),
        payload.get("current_aux_alpha"),
        payload.get("base_aux_alpha"),
    )


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


def to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


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
    base_aux_alpha = float(cfg.cognitive_aux_alpha)
    current_aux_alpha = float(cfg.cognitive_aux_alpha)

    use_scaler = device.startswith("cuda") and args.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_step = 0
    total_tokens = 0
    if args.resume is not None:
        start_step, total_tokens, ckpt_current_aux_alpha, ckpt_base_aux_alpha = load_checkpoint(
            args.resume, model, optimizer, scaler, device=device
        )
        if ckpt_base_aux_alpha is not None:
            base_aux_alpha = float(ckpt_base_aux_alpha)
        if ckpt_current_aux_alpha is not None:
            current_aux_alpha = float(ckpt_current_aux_alpha)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    run_start = time.perf_counter()

    train_loader = None
    train_iter = None
    if args.train_manifest is not None:
        train_loader = create_packed_dataloader(
            manifest_path=args.train_manifest,
            batch_size=args.micro_batch_size,
            shuffle=True,
            num_workers=args.data_workers,
            pin_memory=device.startswith("cuda"),
            drop_last=True,
        )
        train_iter = iter(train_loader)

    for step in range(start_step, args.steps):
        step_start = time.perf_counter()
        running_loss = 0.0
        running_main_loss = 0.0
        running_moe_aux_loss = 0.0
        running_moe_lb_loss = 0.0
        running_moe_entropy_reg_loss = 0.0
        running_router_entropy = 0.0
        agg_expert_usage = None
        agg_co_activation = None

        for _ in range(args.grad_accum_steps):
            batch = make_batch(
                batch_size=args.micro_batch_size,
                seq_len=args.seq_len,
                vocab_size=cfg.vocab_size,
                device=device,
            ) if train_iter is None else None
            if train_iter is not None:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                batch = to_device(batch, device=device)
            want_router_stats = args.log_router_stats_every > 0 and ((step + 1) % args.log_router_stats_every == 0)
            with autocast_context(device=device, precision=args.precision):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_aux_losses=True,
                    return_router_stats=want_router_stats,
                    aux_alpha_override=current_aux_alpha,
                )
                main_loss = out["loss"]
                moe_aux_loss = out["aux_losses"]["moe_aux_loss"]
                total_loss = main_loss + moe_aux_loss
                loss = total_loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(total_loss.item())
            running_main_loss += float(main_loss.item())
            running_moe_aux_loss += float(moe_aux_loss.item())
            running_moe_lb_loss += float(out["aux_losses"]["moe_load_balance_loss"].item())
            running_moe_entropy_reg_loss += float(out["aux_losses"]["moe_entropy_reg_loss"].item())
            if want_router_stats:
                running_router_entropy += float(out["router_stats"]["avg_router_entropy"])
                usage = torch.tensor(out["router_stats"]["expert_usage"], dtype=torch.float64)
                co_act = torch.tensor(out["router_stats"]["co_activation"], dtype=torch.float64)
                agg_expert_usage = usage if agg_expert_usage is None else (agg_expert_usage + usage)
                agg_co_activation = co_act if agg_co_activation is None else (agg_co_activation + co_act)

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
                "main_loss": running_main_loss,
                "moe_aux_loss": running_moe_aux_loss,
                "moe_load_balance_loss": running_moe_lb_loss,
                "moe_entropy_reg_loss": running_moe_entropy_reg_loss,
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
                "current_aux_alpha": current_aux_alpha,
            }
            if agg_expert_usage is not None:
                mean_entropy = running_router_entropy / max(args.grad_accum_steps, 1)
                usage_mean = float(agg_expert_usage.mean().item())
                usage_std = float(agg_expert_usage.std(unbiased=False).item())
                usage_cv = usage_std / (usage_mean + 1e-12)

                payload["router_entropy"] = mean_entropy
                payload["expert_usage"] = agg_expert_usage.tolist()
                payload["co_activation"] = agg_co_activation.tolist() if agg_co_activation is not None else None
                payload["expert_usage_cv"] = usage_cv

                warnings = []
                if mean_entropy < args.collapse_entropy_threshold:
                    warnings.append(
                        f"router_entropy={mean_entropy:.4f} < threshold={args.collapse_entropy_threshold:.4f}"
                    )
                if usage_cv > args.collapse_cv_threshold:
                    warnings.append(
                        f"expert_usage_cv={usage_cv:.4f} > threshold={args.collapse_cv_threshold:.4f}"
                    )
                if warnings:
                    warning_text = "; ".join(warnings)
                    print(f"[WARNING: EXPERT COLLAPSE IMMINENT] step={step + 1} {warning_text}")
                    payload["warning"] = warning_text

                if args.adaptive_aux_alpha:
                    prev_alpha = current_aux_alpha
                    if usage_cv > args.collapse_cv_threshold:
                        current_aux_alpha = min(current_aux_alpha * args.aux_alpha_up_mult, args.aux_alpha_max)
                        payload["aux_alpha_action"] = "increase"
                    else:
                        current_aux_alpha = max(current_aux_alpha * args.aux_alpha_decay, base_aux_alpha)
                        payload["aux_alpha_action"] = "decay"
                    payload["current_aux_alpha"] = current_aux_alpha
                    payload["prev_aux_alpha"] = prev_alpha
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
                current_aux_alpha=current_aux_alpha,
                base_aux_alpha=base_aux_alpha,
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
        current_aux_alpha=current_aux_alpha,
        base_aux_alpha=base_aux_alpha,
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
