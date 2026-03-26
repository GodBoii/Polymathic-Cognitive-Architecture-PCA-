import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.dataset_loader import create_packed_dataloader


def main() -> None:
    manifest_path = PROJECT_ROOT / "data_pipeline" / "artifacts" / "phase1_smoke" / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    loader = create_packed_dataloader(
        manifest_path=manifest_path,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    report = {
        "manifest": str(manifest_path),
        "input_shape": list(batch["input_ids"].shape),
        "labels_shape": list(batch["labels"].shape),
        "attention_shape": list(batch["attention_mask"].shape),
        "input_dtype": str(batch["input_ids"].dtype),
        "labels_dtype": str(batch["labels"].dtype),
        "attention_dtype": str(batch["attention_mask"].dtype),
        "input_finite": bool(torch.isfinite(batch["input_ids"].float()).all().item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
