import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core.config import ModelConfig
from model_core.model import PCAModel
from data_pipeline.dataset_loader import CUDAPrefetchLoader, create_packed_dataloader
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
    parser.add_argument(
        "--fast-forward-on-resume",
        action="store_true",
        help="When using packed manifests, skip already-consumed micro-batches after resume.",
    )

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")

    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--train-manifest", type=Path, default=None, help="Use packed dataset manifest instead of random synthetic batches.")
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=0, help="0 disables eval.")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-to-device", action="store_true", help="Use CUDA stream prefetch wrapper for dataloader batches.")
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="Final LR ratio after cosine decay.")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Used when warmup-steps is 0.")
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--ffn-kind", choices=["swiglu", "standard", "moe"], default="swiglu")
    parser.add_argument("--reasoning-start-layer", type=int, default=None)
    parser.add_argument("--reasoning-ffn-dim", type=int, default=None)
    parser.add_argument("--reasoning-ffn-kind", choices=["swiglu", "standard", "moe"], default=None)
    parser.add_argument("--reasoning-moe-num-experts", type=int, default=16)
    parser.add_argument("--reasoning-moe-top-k", type=int, default=2)
    parser.add_argument("--reasoning-moe-num-groups", type=int, default=4)
    parser.add_argument("--reasoning-moe-groups-top-k", type=int, default=1)
    parser.add_argument("--reasoning-moe-gate-type", choices=["softmax", "sigmoid"], default="sigmoid")
    parser.add_argument("--reasoning-moe-expert-ffn-kind", choices=["swiglu", "standard"], default="swiglu")
    parser.add_argument("--gqa-layers", type=int, default=4)
    parser.add_argument("--lightning-end-layer", type=int, default=16)
    parser.add_argument("--mla-latent-dim", type=int, default=512)
    parser.add_argument("--ffn-multiple-of", type=int, default=256)
    parser.add_argument("--cognitive-loops", type=int, default=5)
    parser.add_argument("--cognitive-num-experts", type=int, default=8)
    parser.add_argument("--cognitive-num-groups", type=int, default=1)
    parser.add_argument("--cognitive-groups-top-k", type=int, default=1)
    parser.add_argument("--cognitive-top-k", type=int, default=4)
    parser.add_argument("--cognitive-gate-type", choices=["softmax", "sigmoid"], default="sigmoid")
    parser.add_argument("--cognitive-aux-alpha", type=float, default=0.01)
    parser.add_argument("--cognitive-entropy-alpha", type=float, default=0.001)
    parser.add_argument("--cognitive-ffn-dim", type=int, default=None)
    parser.add_argument("--cognitive-ffn-kind", choices=["swiglu", "standard"], default="swiglu")

    parser.add_argument("--freeze-layers-below", type=int, default=0, help="Freeze transformer layers [0, N-1].")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--freeze-final-norm", action="store_true")
    parser.add_argument("--freeze-cognitive-block", action="store_true")
    parser.add_argument("--resume-model-only", action="store_true", help="Load only model weights from checkpoint and reset optimizer/scheduler/step state.")

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
        ffn_dim=args.ffn_dim,
        ffn_kind=args.ffn_kind,
        reasoning_start_layer=args.reasoning_start_layer,
        reasoning_ffn_dim=args.reasoning_ffn_dim,
        reasoning_ffn_kind=args.reasoning_ffn_kind,
        reasoning_moe_num_experts=args.reasoning_moe_num_experts,
        reasoning_moe_top_k=args.reasoning_moe_top_k,
        reasoning_moe_num_groups=args.reasoning_moe_num_groups,
        reasoning_moe_groups_top_k=args.reasoning_moe_groups_top_k,
        reasoning_moe_gate_type=args.reasoning_moe_gate_type,
        reasoning_moe_expert_ffn_kind=args.reasoning_moe_expert_ffn_kind,
        gqa_layers=args.gqa_layers,
        lightning_end_layer=args.lightning_end_layer,
        mla_latent_dim=args.mla_latent_dim,
        ffn_multiple_of=args.ffn_multiple_of,
        cognitive_loops=args.cognitive_loops,
        cognitive_num_experts=args.cognitive_num_experts,
        cognitive_num_groups=args.cognitive_num_groups,
        cognitive_groups_top_k=args.cognitive_groups_top_k,
        cognitive_top_k=args.cognitive_top_k,
        cognitive_gate_type=args.cognitive_gate_type,
        cognitive_aux_alpha=args.cognitive_aux_alpha,
        cognitive_entropy_alpha=args.cognitive_entropy_alpha,
        cognitive_ffn_dim=args.cognitive_ffn_dim,
        cognitive_ffn_kind=args.cognitive_ffn_kind,
    )


def _set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def apply_freeze_policy(model: PCAModel, args: argparse.Namespace) -> None:
    if args.freeze_layers_below < 0:
        raise ValueError("--freeze-layers-below must be >= 0")
    if args.freeze_layers_below > len(model.layers):
        raise ValueError("--freeze-layers-below cannot exceed model n_layers")

    for idx, layer in enumerate(model.layers):
        _set_trainable(layer, idx >= args.freeze_layers_below)

    if args.freeze_embeddings:
        _set_trainable(model.embed_tokens, False)
        # lm_head can be tied to embeddings; this naturally freezes both in tied mode.
        if model.lm_head.weight is not model.embed_tokens.weight:
            _set_trainable(model.lm_head, False)

    if args.freeze_final_norm:
        _set_trainable(model.final_norm, False)

    if args.freeze_cognitive_block:
        _set_trainable(model.cognitive_block, False)


def parameter_counts(model: PCAModel) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _split_muon_param_groups(model: PCAModel) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_embedding = "embed_tokens" in name or "lm_head" in name
        if param.ndim == 2 and not is_embedding:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def build_optimizers(model: PCAModel, args: argparse.Namespace) -> List[torch.optim.Optimizer]:
    if args.optimizer == "muon":
        muon_params, adamw_params = _split_muon_param_groups(model)
        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                Muon(
                    muon_params,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
            )
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_params,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    betas=(0.9, 0.95),
                )
            )
        return optimizers
    return [
        torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
    ]


def build_schedulers(optimizers: List[torch.optim.Optimizer], args: argparse.Namespace) -> List[torch.optim.lr_scheduler.LambdaLR]:
    total_steps = max(int(args.steps), 1)
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0:
        warmup_steps = int(total_steps * float(args.warmup_ratio))
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    min_lr_ratio = float(args.min_lr_ratio)

    def lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(max(warmup_steps, 1))
        if total_steps <= warmup_steps:
            return min_lr_ratio
        progress = float(step_idx - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda) for opt in optimizers]


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
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[torch.optim.lr_scheduler.LambdaLR],
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    total_tokens: int,
    micro_batches_seen: int,
    cfg: ModelConfig,
    current_aux_alpha: float,
    base_aux_alpha: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "total_tokens": total_tokens,
        "micro_batches_seen": micro_batches_seen,
        "model_state": model.state_dict(),
        "optimizer_states": [opt.state_dict() for opt in optimizers],
        "scheduler_states": [sch.state_dict() for sch in schedulers],
        "scaler_state": scaler.state_dict(),
        "model_config": cfg.to_dict(),
        "current_aux_alpha": current_aux_alpha,
        "base_aux_alpha": base_aux_alpha,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: PCAModel,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[torch.optim.lr_scheduler.LambdaLR],
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    load_optimizer_state: bool = True,
) -> Tuple[int, int, int, float | None, float | None]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"])
    if load_optimizer_state:
        states = payload.get("optimizer_states")
        if states is None and "optimizer_state" in payload:
            states = [payload["optimizer_state"]]
        if states is not None:
            try:
                for opt, state in zip(optimizers, states):
                    opt.load_state_dict(state)
            except ValueError as exc:
                raise ValueError(
                    "Optimizer state restore failed, likely due to a changed trainable-parameter set. "
                    "Use --resume-model-only when transitioning to a new freeze policy."
                ) from exc
        scheduler_states = payload.get("scheduler_states")
        if scheduler_states is not None:
            for scheduler, state in zip(schedulers, scheduler_states):
                scheduler.load_state_dict(state)
        scaler.load_state_dict(payload["scaler_state"])
    return (
        int(payload["step"]),
        int(payload["total_tokens"]),
        int(payload.get("micro_batches_seen", int(payload["step"]))),
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


@torch.no_grad()
def run_eval(
    model: PCAModel,
    eval_loader,
    device: str,
    precision: str,
    max_batches: int,
) -> tuple[float, float]:
    if max_batches <= 0 or len(eval_loader) == 0:
        return float("nan"), float("nan")
    model.eval()
    losses: list[float] = []
    eval_iter = iter(eval_loader)
    for _ in range(max_batches):
        try:
            batch = next(eval_iter)
        except StopIteration:
            eval_iter = iter(eval_loader)
            try:
                batch = next(eval_iter)
            except StopIteration:
                break
        batch = to_device(batch, device=device)
        with autocast_context(device=device, precision=precision):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        losses.append(float(out["loss"].item()))
    model.train()
    eval_loss = sum(losses) / max(len(losses), 1)
    eval_ppl = math.exp(min(eval_loss, 20.0))
    return eval_loss, eval_ppl


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
    apply_freeze_policy(model, args)
    trainable_params, total_params = parameter_counts(model)
    print(
        json.dumps(
            {
                "event": "trainable_params",
                "trainable": trainable_params,
                "total": total_params,
                "frozen": total_params - trainable_params,
                "freeze_layers_below": args.freeze_layers_below,
                "freeze_embeddings": args.freeze_embeddings,
                "freeze_final_norm": args.freeze_final_norm,
                "freeze_cognitive_block": args.freeze_cognitive_block,
            }
        )
    )
    optimizers = build_optimizers(model, args)
    schedulers = build_schedulers(optimizers, args)
    base_aux_alpha = float(cfg.cognitive_aux_alpha)
    current_aux_alpha = float(cfg.cognitive_aux_alpha)

    use_scaler = device.startswith("cuda") and args.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_step = 0
    total_tokens = 0
    micro_batches_seen = 0
    if args.resume is not None:
        start_step, total_tokens, micro_batches_seen, ckpt_current_aux_alpha, ckpt_base_aux_alpha = load_checkpoint(
            args.resume,
            model,
            optimizers,
            schedulers,
            scaler,
            device=device,
            load_optimizer_state=not args.resume_model_only,
        )
        if args.resume_model_only:
            start_step = 0
            total_tokens = 0
            micro_batches_seen = 0
            print(json.dumps({"event": "resume_model_only", "checkpoint": str(args.resume)}))
        else:
            if micro_batches_seen <= start_step:
                micro_batches_seen = start_step * args.grad_accum_steps
            if ckpt_base_aux_alpha is not None:
                base_aux_alpha = float(ckpt_base_aux_alpha)
            if ckpt_current_aux_alpha is not None:
                current_aux_alpha = float(ckpt_current_aux_alpha)

    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    run_start = time.perf_counter()

    train_loader = None
    train_iter = None
    batches_on_device = False
    train_seq_len = args.seq_len
    if args.train_manifest is not None:
        train_manifest = json.loads(args.train_manifest.read_text(encoding="utf-8"))
        train_seq_len = int(train_manifest["config"]["seq_len"])
        train_loader = create_packed_dataloader(
            manifest_path=args.train_manifest,
            batch_size=args.micro_batch_size,
            shuffle=True,
            num_workers=args.data_workers,
            pin_memory=device.startswith("cuda"),
            drop_last=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
        if args.prefetch_to_device and device.startswith("cuda"):
            train_loader = CUDAPrefetchLoader(train_loader, device=device)
            batches_on_device = True
        train_iter = iter(train_loader)
        if args.fast_forward_on_resume and micro_batches_seen > 0:
            print(f"[INFO] Fast-forwarding dataloader by {micro_batches_seen} micro-batches...")
            skipped = 0
            while skipped < micro_batches_seen:
                try:
                    _ = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    _ = next(train_iter)
                skipped += 1

    eval_loader = None
    if args.eval_manifest is not None and args.eval_every > 0 and args.eval_batches > 0:
        eval_loader = create_packed_dataloader(
            manifest_path=args.eval_manifest,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.startswith("cuda"),
            drop_last=False,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=False,
        )

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
        running_data_wait_sec = 0.0
        running_compute_sec = 0.0
        active_seq_len = train_seq_len if train_iter is not None else args.seq_len

        for _ in range(args.grad_accum_steps):
            t_data0 = time.perf_counter()
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
                if not batches_on_device:
                    batch = to_device(batch, device=device)
                active_seq_len = int(batch["input_ids"].shape[1])
            t_data1 = time.perf_counter()
            running_data_wait_sec += t_data1 - t_data0

            want_router_stats = args.log_router_stats_every > 0 and ((step + 1) % args.log_router_stats_every == 0)
            t_comp0 = time.perf_counter()
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
            t_comp1 = time.perf_counter()
            running_compute_sec += t_comp1 - t_comp0
            micro_batches_seen += 1

            running_loss += float(total_loss.item()) / args.grad_accum_steps
            running_main_loss += float(main_loss.item()) / args.grad_accum_steps
            running_moe_aux_loss += float(moe_aux_loss.item()) / args.grad_accum_steps
            running_moe_lb_loss += float(out["aux_losses"]["moe_load_balance_loss"].item()) / args.grad_accum_steps
            running_moe_entropy_reg_loss += float(out["aux_losses"]["moe_entropy_reg_loss"].item()) / args.grad_accum_steps
            if want_router_stats:
                running_router_entropy += float(out["router_stats"]["avg_router_entropy"])
                usage = torch.tensor(out["router_stats"]["expert_usage"], dtype=torch.float64)
                co_act = torch.tensor(out["router_stats"]["co_activation"], dtype=torch.float64)
                agg_expert_usage = usage if agg_expert_usage is None else (agg_expert_usage + usage)
                agg_co_activation = co_act if agg_co_activation is None else (agg_co_activation + co_act)

        if scaler.is_enabled():
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)

        if scaler.is_enabled():
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()
        else:
            for optimizer in optimizers:
                optimizer.step()
        for scheduler in schedulers:
            scheduler.step()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        tokens_step = args.micro_batch_size * active_seq_len * args.grad_accum_steps
        total_tokens += tokens_step
        dt = time.perf_counter() - step_start
        tokens_per_sec = tokens_step / max(dt, 1e-9)

        if (step + 1) % args.log_every == 0:
            payload = {
                "step": step + 1,
                "loss": running_loss,
                "main_loss": running_main_loss,
                "moe_aux_loss": running_moe_aux_loss,
                "moe_load_balance_loss": running_moe_lb_loss,
                "moe_entropy_reg_loss": running_moe_entropy_reg_loss,
                "grad_norm": float(grad_norm.item()),
                "lr": float(optimizers[0].param_groups[0]["lr"]),
                "tokens_per_sec": tokens_per_sec,
                "total_tokens": total_tokens,
                "step_time_sec": dt,
                "data_wait_sec": running_data_wait_sec,
                "compute_sec": running_compute_sec,
                "data_wait_pct": (running_data_wait_sec / max(dt, 1e-9)) * 100.0,
                "gpu_mem_gb": (
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "current_aux_alpha": current_aux_alpha,
            }
            if eval_loader is not None and (step + 1) % args.eval_every == 0:
                eval_loss, eval_perplexity = run_eval(
                    model=model,
                    eval_loader=eval_loader,
                    device=device,
                    precision=args.precision,
                    max_batches=args.eval_batches,
                )
                payload["eval_loss"] = eval_loss
                payload["eval_perplexity"] = eval_perplexity
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
                optimizers=optimizers,
                schedulers=schedulers,
                scaler=scaler,
                step=step + 1,
                total_tokens=total_tokens,
                micro_batches_seen=micro_batches_seen,
                cfg=cfg,
                current_aux_alpha=current_aux_alpha,
                base_aux_alpha=base_aux_alpha,
            )

    final_ckpt = args.checkpoint_dir / "last.pt"
    save_checkpoint(
        path=final_ckpt,
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        scaler=scaler,
        step=args.steps,
        total_tokens=total_tokens,
        micro_batches_seen=micro_batches_seen,
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
