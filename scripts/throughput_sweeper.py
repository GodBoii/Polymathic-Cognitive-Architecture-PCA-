import argparse
import itertools
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_int_grid(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("Grid cannot be empty.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep throughput configs and rank results.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--micro-batch-grid", type=str, default="1,2,4")
    parser.add_argument("--grad-accum-grid", type=str, default="1,2,4")
    parser.add_argument("--workers-grid", type=str, default="0,2,4")
    parser.add_argument("--prefetch-grid", type=str, default="2,4")
    parser.add_argument("--prefetch-to-device", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--output-json", type=Path, default=Path("train/sweeps/throughput_sweep.json"))
    return parser.parse_args()


def run_trial(args: argparse.Namespace, micro_batch: int, grad_accum: int, workers: int, prefetch_factor: int) -> dict:
    checkpoint_dir = Path("train/checkpoints_sweep_tmp") / f"mb{micro_batch}_ga{grad_accum}_w{workers}_pf{prefetch_factor}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path("train/train_step.py")),
        "--steps",
        str(args.steps),
        "--log-every",
        "1",
        "--save-every",
        str(args.steps),
        "--log-router-stats-every",
        "0",
        "--optimizer",
        args.optimizer,
        "--precision",
        args.precision,
        "--device",
        args.device,
        "--d-model",
        str(args.d_model),
        "--n-layers",
        str(args.n_layers),
        "--n-heads",
        str(args.n_heads),
        "--n-kv-heads",
        str(args.n_kv_heads),
        "--max-seq-len",
        str(args.max_seq_len),
        "--micro-batch-size",
        str(micro_batch),
        "--grad-accum-steps",
        str(grad_accum),
        "--train-manifest",
        str(args.train_manifest),
        "--data-workers",
        str(workers),
        "--prefetch-factor",
        str(prefetch_factor),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if workers > 0:
        cmd.append("--persistent-workers")
    if args.prefetch_to_device:
        cmd.append("--prefetch-to-device")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    step_rows = []
    for ln in lines:
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "step" in row and "tokens_per_sec" in row:
            step_rows.append(row)

    step_rows = [r for r in step_rows if int(r.get("step", 0)) > args.warmup_steps]
    tokens = [float(r["tokens_per_sec"]) for r in step_rows]
    waits = [float(r.get("data_wait_pct", 0.0)) for r in step_rows]

    return {
        "config": {
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "workers": workers,
            "prefetch_factor": prefetch_factor,
            "prefetch_to_device": args.prefetch_to_device,
        },
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "samples": len(step_rows),
        "mean_tokens_per_sec": statistics.mean(tokens) if tokens else 0.0,
        "median_tokens_per_sec": statistics.median(tokens) if tokens else 0.0,
        "mean_data_wait_pct": statistics.mean(waits) if waits else 0.0,
        "stderr_tail": proc.stderr.splitlines()[-5:],
    }


def main() -> None:
    args = parse_args()
    micro_grid = parse_int_grid(args.micro_batch_grid)
    ga_grid = parse_int_grid(args.grad_accum_grid)
    worker_grid = parse_int_grid(args.workers_grid)
    prefetch_grid = parse_int_grid(args.prefetch_grid)

    combos = list(itertools.product(micro_grid, ga_grid, worker_grid, prefetch_grid))
    results = []
    for mb, ga, w, pf in combos:
        print(f"[SWEEP] mb={mb} ga={ga} workers={w} prefetch={pf}")
        results.append(run_trial(args, mb, ga, w, pf))

    ranked = sorted(results, key=lambda r: (r["mean_tokens_per_sec"], -r["mean_data_wait_pct"]), reverse=True)
    payload = {
        "num_trials": len(results),
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
