import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import shutil


DEFAULT_PRESETS = [
    "train/presets/ablation_three_phase_v5_base.json",
    "train/presets/ablation_three_phase_v5_low_aux.json",
    "train/presets/ablation_three_phase_v5_low_norm.json",
    "train/presets/ablation_three_phase_v5_higher_update.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local three-phase ablation grid.")
    parser.add_argument("--fresh-start", action="store_true", help="Run launcher with --fresh-start.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Archive existing ablation run logs/checkpoints before launching so comparisons stay clean.",
    )
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


def archive_previous_run_artifacts(preset_path: Path) -> dict[str, str] | None:
    preset_payload = json.loads(preset_path.read_text(encoding="utf-8"))
    paths = preset_payload.get("Paths", {})
    run_log = Path(paths["run_log"])
    checkpoint_dir = Path(paths["checkpoint_dir"])

    existing = []
    if run_log.exists():
        existing.append(run_log)
    if checkpoint_dir.exists():
        existing.append(checkpoint_dir)
    if not existing:
        return None

    archive_root = Path("tmp") / "ablation_archive" / datetime.now().strftime("%Y%m%d_%H%M%S") / preset_path.stem
    archive_root.mkdir(parents=True, exist_ok=True)

    archived_items: list[str] = []
    for item in existing:
        target = archive_root / item.name
        if target.exists():
            suffix = 1
            while True:
                candidate = archive_root / f"{item.stem}_{suffix}{item.suffix}" if item.is_file() else archive_root / f"{item.name}_{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                suffix += 1
        shutil.move(str(item), str(target))
        archived_items.append(str(target))

    return {
        "preset": str(preset_path),
        "archive_root": str(archive_root),
        "archived_items": archived_items,
    }


def main() -> None:
    args = parse_args()
    summary = []
    for preset in args.presets:
        preset_path = Path(preset)
        if not preset_path.exists():
            print(json.dumps({"event": "skip_missing_preset", "preset": str(preset_path)}))
            continue

        if args.clean:
            archive_info = archive_previous_run_artifacts(preset_path)
            if archive_info is not None:
                print(json.dumps({"event": "ablation_run_archive", **archive_info}))

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
