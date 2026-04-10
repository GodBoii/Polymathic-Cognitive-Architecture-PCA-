import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_PRESETS = [
    "train/presets/ablation_three_phase_v5_base.json",
    "train/presets/ablation_three_phase_v5_low_aux.json",
    "train/presets/ablation_three_phase_v5_low_norm.json",
    "train/presets/ablation_three_phase_v5_higher_update.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local three-phase ablation grid.")
    parser.add_argument("--fresh-start", action="store_true", help="Run launcher with --fresh-start.")
    parser.add_argument("--presets", nargs="*", default=DEFAULT_PRESETS)
    return parser.parse_args()


def extract_last_step_payload(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    last_step = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in payload:
            last_step = payload
    return last_step


def main() -> None:
    args = parse_args()
    summary = []
    for preset in args.presets:
        preset_path = Path(preset)
        if not preset_path.exists():
            print(json.dumps({"event": "skip_missing_preset", "preset": str(preset_path)}))
            continue

        cmd = [sys.executable, "scripts/launch_pretrain.py", "--config", str(preset_path)]
        if args.fresh_start:
            cmd.append("--fresh-start")
        print(json.dumps({"event": "ablation_run_start", "preset": str(preset_path), "command": cmd}))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(json.dumps({"event": "ablation_run_failed", "preset": str(preset_path), "exit_code": rc}))
            continue

        preset_payload = json.loads(preset_path.read_text(encoding="utf-8"))
        run_log = Path(preset_payload["Paths"]["run_log"])
        step_payload = extract_last_step_payload(run_log)
        summary.append(
            {
                "preset": str(preset_path),
                "run_log": str(run_log),
                "last_step": step_payload,
            }
        )

    print(json.dumps({"event": "ablation_summary", "results": summary}, indent=2))


if __name__ == "__main__":
    main()
