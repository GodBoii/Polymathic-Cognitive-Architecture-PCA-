import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MoE telemetry from JSONL train logs.")
    parser.add_argument("--log-file", type=Path, required=True, help="Path to training log file (one JSON object per line).")
    parser.add_argument("--cv-warn-threshold", type=float, default=0.5, help="Warn if expert utilization CV exceeds this.")
    parser.add_argument(
        "--dead-expert-min-ratio",
        type=float,
        default=0.15,
        help="Warn if an expert receives less than this ratio of uniform expected traffic.",
    )
    parser.add_argument("--entropy-warn-threshold", type=float, default=0.25, help="Warn if mean router entropy falls below this.")
    return parser.parse_args()


def load_json_lines(path: Path) -> list[dict]:
    text = None
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode log file with supported encodings.")

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> None:
    args = parse_args()
    rows = load_json_lines(args.log_file)
    if not rows:
        raise SystemExit("No JSON rows found in log file.")

    moe_rows = [r for r in rows if "expert_usage" in r]
    if not moe_rows:
        raise SystemExit("No MoE telemetry rows found. Run train_step with --log-router-stats-every > 0.")

    num_experts = len(moe_rows[0]["expert_usage"])
    usage_total = torch.zeros(num_experts, dtype=torch.float64)
    co_total = torch.zeros((num_experts, num_experts), dtype=torch.float64)
    entropy_vals = []
    aux_vals = []

    for row in moe_rows:
        usage_total += torch.tensor(row["expert_usage"], dtype=torch.float64)
        if "co_activation" in row:
            co_total += torch.tensor(row["co_activation"], dtype=torch.float64)
        if "router_entropy" in row:
            entropy_vals.append(float(row["router_entropy"]))
        if "moe_aux_loss" in row:
            aux_vals.append(float(row["moe_aux_loss"]))

    total_usage = float(usage_total.sum().item())
    mean_usage = float(usage_total.mean().item()) if num_experts > 0 else 0.0
    std_usage = float(usage_total.std(unbiased=False).item()) if num_experts > 0 else 0.0
    cv = std_usage / (mean_usage + 1e-12)

    util_frac = (usage_total / (usage_total.sum() + 1e-12)).tolist()
    dead_threshold = args.dead_expert_min_ratio * mean_usage
    dead_experts = [i for i, v in enumerate(usage_total.tolist()) if v < dead_threshold]

    diag = torch.diag(co_total)
    denom = torch.sqrt(torch.outer(diag + 1e-12, diag + 1e-12))
    overlap = (co_total / denom).clamp(min=0.0, max=1.0)

    warnings = []
    if cv > args.cv_warn_threshold:
        warnings.append(f"High utilization CV ({cv:.4f}) indicates expert imbalance.")
    if dead_experts:
        warnings.append(f"Dead/underused experts detected: {dead_experts}")
    if entropy_vals:
        entropy_mean = sum(entropy_vals) / len(entropy_vals)
        if entropy_mean < args.entropy_warn_threshold:
            warnings.append(f"Low router entropy ({entropy_mean:.4f}) suggests routing collapse.")
    else:
        entropy_mean = None
    aux_mean = (sum(aux_vals) / len(aux_vals)) if aux_vals else None

    payload = {
        "log_file": str(args.log_file),
        "num_rows": len(rows),
        "num_moe_rows": len(moe_rows),
        "num_experts": num_experts,
        "total_usage": total_usage,
        "usage_per_expert": usage_total.tolist(),
        "utilization_fraction": util_frac,
        "utilization_cv": cv,
        "dead_expert_threshold": dead_threshold,
        "dead_experts": dead_experts,
        "router_entropy_mean": entropy_mean,
        "moe_aux_loss_mean": aux_mean,
        "topk_overlap_matrix": overlap.tolist(),
        "warnings": warnings,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
