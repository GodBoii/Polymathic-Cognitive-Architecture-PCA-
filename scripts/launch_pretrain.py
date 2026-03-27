import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch long-run pretraining from a JSON preset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_add(cmd: list[str], flag: str, value: Any | None) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def maybe_add_bool(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def parse_step_from_name(path: Path) -> int:
    # step_123.pt -> 123
    stem = path.stem
    if not stem.startswith("step_"):
        return -1
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return -1


def enforce_retention(checkpoint_dir: Path, keep_last: int) -> None:
    step_files = sorted(
        [p for p in checkpoint_dir.glob("step_*.pt") if parse_step_from_name(p) >= 0],
        key=parse_step_from_name,
    )
    if len(step_files) <= keep_last + 1:
        return

    first = step_files[0]
    keep = {first}
    keep.update(step_files[-keep_last:])
    for p in step_files:
        if p not in keep:
            p.unlink(missing_ok=True)


def build_train_cmd(
    train_script: Path,
    target_step: int,
    cfg: dict[str, Any],
    resume_path: Path | None,
) -> list[str]:
    hw = cfg["Hardware_Profile"]
    model = cfg["Model_Architecture"]
    train = cfg["Training_Hyperparams"]
    alerts = cfg["Telemetry_Alerts"]
    paths = cfg["Paths"]

    cmd = [sys.executable, str(train_script)]

    # Core run controls
    maybe_add(cmd, "--steps", target_step)
    maybe_add(cmd, "--log-every", train.get("log_every", 1))
    maybe_add(cmd, "--save-every", train.get("save_every", 500))
    maybe_add(cmd, "--checkpoint-dir", paths["checkpoint_dir"])
    if resume_path is not None:
        maybe_add(cmd, "--resume", resume_path)
        maybe_add_bool(cmd, "--fast-forward-on-resume", True)

    # Data
    maybe_add(cmd, "--train-manifest", paths["train_manifest"])
    maybe_add(cmd, "--eval-manifest", paths.get("eval_manifest"))
    maybe_add(cmd, "--eval-every", train.get("eval_every", 0))
    maybe_add(cmd, "--eval-batches", train.get("eval_batches", 50))
    maybe_add(cmd, "--eval-batch-size", train.get("eval_batch_size", train.get("micro_batch_size", 1)))

    # Hardware
    maybe_add(cmd, "--device", hw.get("device", "cuda"))
    maybe_add(cmd, "--precision", hw.get("precision", "bf16"))
    maybe_add(cmd, "--data-workers", hw.get("data_workers", 0))
    maybe_add(cmd, "--prefetch-factor", hw.get("prefetch_factor", 4))
    maybe_add_bool(cmd, "--persistent-workers", bool(hw.get("persistent_workers", False)))
    maybe_add_bool(cmd, "--prefetch-to-device", bool(hw.get("prefetch_to_device", False)))

    # Model architecture
    maybe_add(cmd, "--vocab-size", model.get("vocab_size", 32000))
    maybe_add(cmd, "--d-model", model.get("d_model", 2048))
    maybe_add(cmd, "--n-layers", model.get("n_layers", 24))
    maybe_add(cmd, "--n-heads", model.get("n_heads", 16))
    maybe_add(cmd, "--n-kv-heads", model.get("n_kv_heads", 4))
    maybe_add(cmd, "--max-seq-len", model.get("max_seq_len", 8192))
    maybe_add(cmd, "--dropout", model.get("dropout", 0.0))
    maybe_add(cmd, "--ffn-dim", model.get("ffn_dim"))
    maybe_add(cmd, "--ffn-kind", model.get("ffn_kind", "swiglu"))
    maybe_add(cmd, "--reasoning-start-layer", model.get("reasoning_start_layer"))
    maybe_add(cmd, "--reasoning-ffn-dim", model.get("reasoning_ffn_dim"))
    maybe_add(cmd, "--reasoning-ffn-kind", model.get("reasoning_ffn_kind"))
    maybe_add(cmd, "--reasoning-moe-num-experts", model.get("reasoning_moe_num_experts", 16))
    maybe_add(cmd, "--reasoning-moe-top-k", model.get("reasoning_moe_top_k", 2))
    maybe_add(cmd, "--reasoning-moe-num-groups", model.get("reasoning_moe_num_groups", 4))
    maybe_add(cmd, "--reasoning-moe-groups-top-k", model.get("reasoning_moe_groups_top_k", 1))
    maybe_add(cmd, "--reasoning-moe-gate-type", model.get("reasoning_moe_gate_type", "sigmoid"))
    maybe_add(cmd, "--reasoning-moe-expert-ffn-kind", model.get("reasoning_moe_expert_ffn_kind", "swiglu"))
    maybe_add(cmd, "--gqa-layers", model.get("gqa_layers", 4))
    maybe_add(cmd, "--lightning-end-layer", model.get("lightning_end_layer", 16))
    maybe_add(cmd, "--mla-latent-dim", model.get("mla_latent_dim", 512))
    maybe_add(cmd, "--ffn-multiple-of", model.get("ffn_multiple_of", 256))
    maybe_add(cmd, "--cognitive-loops", model.get("cognitive_loops", 5))
    maybe_add(cmd, "--cognitive-num-experts", model.get("cognitive_num_experts", 8))
    maybe_add(cmd, "--cognitive-num-groups", model.get("cognitive_num_groups", 1))
    maybe_add(cmd, "--cognitive-groups-top-k", model.get("cognitive_groups_top_k", 1))
    maybe_add(cmd, "--cognitive-top-k", model.get("cognitive_top_k", 4))
    maybe_add(cmd, "--cognitive-gate-type", model.get("cognitive_gate_type", "sigmoid"))
    maybe_add(cmd, "--cognitive-aux-alpha", model.get("cognitive_aux_alpha", 0.01))
    maybe_add(cmd, "--cognitive-entropy-alpha", model.get("cognitive_entropy_alpha", 0.001))
    maybe_add(cmd, "--cognitive-ffn-dim", model.get("cognitive_ffn_dim"))
    maybe_add(cmd, "--cognitive-ffn-kind", model.get("cognitive_ffn_kind", "swiglu"))

    # Training hyperparams
    maybe_add(cmd, "--optimizer", train.get("optimizer", "muon"))
    maybe_add(cmd, "--micro-batch-size", train.get("micro_batch_size", 1))
    maybe_add(cmd, "--grad-accum-steps", train.get("grad_accum_steps", 1))
    maybe_add(cmd, "--lr", train.get("lr", 3e-4))
    maybe_add(cmd, "--warmup-steps", train.get("warmup_steps"))
    maybe_add(cmd, "--warmup-ratio", train.get("warmup_ratio", 0.05))
    maybe_add(cmd, "--min-lr-ratio", train.get("min_lr_ratio", 0.1))
    maybe_add(cmd, "--weight-decay", train.get("weight_decay", 0.01))
    maybe_add(cmd, "--clip-grad-norm", train.get("clip_grad_norm", 1.0))
    maybe_add(cmd, "--seed", train.get("seed", 42))
    maybe_add(cmd, "--freeze-layers-below", train.get("freeze_layers_below", 0))
    maybe_add_bool(cmd, "--freeze-embeddings", bool(train.get("freeze_embeddings", False)))
    maybe_add_bool(cmd, "--freeze-final-norm", bool(train.get("freeze_final_norm", False)))
    maybe_add_bool(cmd, "--freeze-cognitive-block", bool(train.get("freeze_cognitive_block", False)))
    maybe_add_bool(cmd, "--resume-model-only", bool(train.get("resume_model_only", False)))

    # Alert settings passed through to train loop
    maybe_add(cmd, "--log-router-stats-every", alerts.get("log_router_stats_every", 0))
    maybe_add(cmd, "--collapse-cv-threshold", alerts.get("collapse_cv_threshold", 0.5))
    maybe_add(cmd, "--collapse-entropy-threshold", alerts.get("collapse_entropy_threshold", 0.25))
    maybe_add_bool(cmd, "--adaptive-aux-alpha", bool(alerts.get("adaptive_aux_alpha", False)))
    maybe_add(cmd, "--aux-alpha-max", alerts.get("aux_alpha_max", 0.2))
    maybe_add(cmd, "--aux-alpha-up-mult", alerts.get("aux_alpha_up_mult", 1.5))
    maybe_add(cmd, "--aux-alpha-decay", alerts.get("aux_alpha_decay", 0.95))

    return cmd


def launch_chunk(cmd: list[str], log_path: Path) -> tuple[int, dict[str, Any] | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_step_payload = None

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc, log_path.open("a", encoding="utf-8") as lf:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            print(line)
            lf.write(line + "\n")
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "step" in payload:
                    last_step_payload = payload
        return proc.wait(), last_step_payload


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg["Paths"]
    train = cfg["Training_Hyperparams"]
    alerts = cfg["Telemetry_Alerts"]

    checkpoint_dir = Path(paths["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(paths.get("run_log", "train/pretrain_run.jsonl"))
    train_script = Path(paths.get("train_script", "train/train_step.py"))

    total_steps = int(train["total_steps"])
    chunk_steps = int(train.get("chunk_steps", 1000))
    keep_last = int(train.get("checkpoint_keep_last", 5))

    usage_cv_threshold = float(alerts.get("early_stop_usage_cv_threshold", 0.85))
    usage_cv_patience = int(alerts.get("early_stop_usage_cv_patience", 5))
    stop_on_nan_ppl = bool(alerts.get("stop_on_nan_eval_perplexity", True))

    high_cv_streak = 0
    current_step = 0
    resume_path = Path(paths["resume_from"]) if paths.get("resume_from") else None

    if resume_path is None:
        last_ckpt = checkpoint_dir / "last.pt"
        if last_ckpt.exists():
            resume_path = last_ckpt

    while current_step < total_steps:
        target_step = min(total_steps, current_step + chunk_steps)
        cmd = build_train_cmd(
            train_script=train_script,
            target_step=target_step,
            cfg=cfg,
            resume_path=resume_path,
        )
        rc, payload = launch_chunk(cmd, log_path=log_path)
        if rc != 0:
            print(f"[LAUNCHER] Child run failed with code {rc}.")
            break
        if payload is None:
            print("[LAUNCHER] No step payload found; aborting.")
            break

        current_step = int(payload["step"])
        resume_path = checkpoint_dir / "last.pt"
        enforce_retention(checkpoint_dir=checkpoint_dir, keep_last=keep_last)

        if "eval_perplexity" in payload:
            ppl = float(payload["eval_perplexity"])
            if stop_on_nan_ppl and (math.isnan(ppl) or math.isinf(ppl)):
                emergency = checkpoint_dir / f"emergency_stop_step_{current_step}.pt"
                if resume_path.exists():
                    shutil.copy2(resume_path, emergency)
                print(f"[LAUNCHER] Early stop: eval perplexity invalid at step {current_step}.")
                break

            usage_cv = float(payload.get("expert_usage_cv", 0.0))
            if usage_cv > usage_cv_threshold:
                high_cv_streak += 1
            else:
                high_cv_streak = 0

            if high_cv_streak >= usage_cv_patience:
                emergency = checkpoint_dir / f"emergency_stop_step_{current_step}.pt"
                if resume_path.exists():
                    shutil.copy2(resume_path, emergency)
                print(
                    f"[LAUNCHER] Early stop: usage_cv>{usage_cv_threshold} for "
                    f"{high_cv_streak} consecutive eval checks."
                )
                break

        if current_step >= total_steps:
            break

    print(json.dumps({"status": "launcher_done", "final_step": current_step, "total_steps": total_steps}, indent=2))


if __name__ == "__main__":
    main()
