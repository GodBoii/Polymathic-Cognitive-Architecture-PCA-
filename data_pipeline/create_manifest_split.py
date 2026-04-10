import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/eval packed-manifest splits from an existing packed manifest.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-output-dir", type=Path, required=True)
    parser.add_argument("--eval-output-dir", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-shard-sequences", type=int, default=None)
    parser.add_argument("--eval-shard-sequences", type=int, default=None)
    return parser.parse_args()


def resolve_source_shard(manifest_path: Path, shard_path: str) -> Path:
    raw = Path(shard_path)
    if raw.exists():
        return raw
    candidate = manifest_path.parent / raw.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve shard '{shard_path}' from manifest '{manifest_path}'.")


def write_split(
    sequences: list[np.ndarray],
    out_dir: Path,
    config: dict,
    shard_sequences: int,
    split_name: str,
    source_manifest_path: Path,
    eval_ratio: float,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_index: list[dict] = []
    seq_len = int(config["seq_len"])
    dtype = str(sequences[0].dtype) if sequences else "uint16"

    for shard_id, start in enumerate(range(0, len(sequences), shard_sequences)):
        chunk = sequences[start : start + shard_sequences]
        if not chunk:
            continue
        arr = np.stack(chunk, axis=0)
        shard_path = out_dir / f"{split_name}_shard_{shard_id:05d}.npy"
        np.save(shard_path, arr, allow_pickle=False)
        shard_index.append(
            {
                "shard_id": shard_id,
                "path": str(shard_path),
                "num_sequences": int(arr.shape[0]),
                "seq_len": seq_len,
                "dtype": str(arr.dtype),
            }
        )

    manifest = {
        "config": {
            **config,
            "output_dir": str(out_dir),
            "split_name": split_name,
            "source_manifest": str(source_manifest_path),
            "eval_ratio": eval_ratio,
            "seed": seed,
        },
        "stats": {
            "num_sequences": len(sequences),
            "tokens_packed": len(sequences) * seq_len,
            "dtype": dtype,
        },
        "num_shards": len(shard_index),
        "shards": shard_index,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    args = parse_args()
    if not 0.0 < args.eval_ratio < 1.0:
        raise SystemExit("--eval-ratio must be in (0, 1)")

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_config = source_manifest["config"]
    seq_len = int(source_config["seq_len"])
    rng = np.random.default_rng(args.seed)

    all_sequences: list[np.ndarray] = []
    for shard_info in source_manifest["shards"]:
        shard_path = resolve_source_shard(args.source_manifest, shard_info["path"])
        shard = np.load(shard_path, mmap_mode="r")
        for i in range(int(shard.shape[0])):
            all_sequences.append(np.asarray(shard[i]).copy())

    if not all_sequences:
        raise SystemExit("Source manifest contains no sequences.")

    indices = np.arange(len(all_sequences))
    rng.shuffle(indices)
    eval_count = max(1, int(round(len(indices) * args.eval_ratio)))
    eval_idx = set(indices[:eval_count].tolist())

    train_sequences = [all_sequences[i] for i in range(len(all_sequences)) if i not in eval_idx]
    eval_sequences = [all_sequences[i] for i in range(len(all_sequences)) if i in eval_idx]

    if not train_sequences or not eval_sequences:
        raise SystemExit("Split resulted in an empty train or eval set.")

    default_shard_sequences = int(source_config.get("shard_sequences", 1024))
    train_manifest = write_split(
        sequences=train_sequences,
        out_dir=args.train_output_dir,
        config=source_config,
        shard_sequences=args.train_shard_sequences or default_shard_sequences,
        split_name="train",
        source_manifest_path=args.source_manifest,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    eval_manifest = write_split(
        sequences=eval_sequences,
        out_dir=args.eval_output_dir,
        config=source_config,
        shard_sequences=args.eval_shard_sequences or max(1, min(default_shard_sequences, len(eval_sequences))),
        split_name="eval",
        source_manifest_path=args.source_manifest,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    summary = {
        "source_manifest": str(args.source_manifest),
        "seq_len": seq_len,
        "total_sequences": len(all_sequences),
        "train_sequences": len(train_sequences),
        "eval_sequences": len(eval_sequences),
        "train_manifest": str(args.train_output_dir / "manifest.json"),
        "eval_manifest": str(args.eval_output_dir / "manifest.json"),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
    }
    print(json.dumps(summary, indent=2))
