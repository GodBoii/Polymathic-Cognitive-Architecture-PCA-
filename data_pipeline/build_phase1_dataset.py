import argparse
import concurrent.futures
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.filters import (
    DEFAULT_BLOCKED_PATH_KEYWORDS,
    FACTUAL_PATTERNS,
    line_is_probably_factual,
    should_skip_file,
)


def iter_text_files(root: Path, extensions: tuple[str, ...]) -> Iterable[Path]:
    ext_set = {e.lower() for e in extensions}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            yield path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase-1 packed dataset shards for PCA pretraining.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data_pipeline/artifacts/phase1"))
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--shard-sequences", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1, help="Number of CPU workers for parallel pretokenization.")
    parser.add_argument("--min-line-len", type=int, default=8)
    parser.add_argument("--factual-filter", action="store_true", help="Enable heuristic factual stripping.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".txt", ".md", ".json", ".csv"],
    )
    return parser.parse_args()


def process_file_worker(
    file_path_str: str,
    tokenizer_model: str,
    min_line_len: int,
    factual_filter: bool,
) -> dict:
    file_path = Path(file_path_str)
    stats = {
        "lines_seen": 0,
        "lines_dropped_short": 0,
        "lines_dropped_factual": 0,
        "lines_kept": 0,
        "tokens_encoded": 0,
    }
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 2

    try:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"encoded": [], "stats": stats}

    encoded: list[list[int]] = []
    for line in normalize_text(raw).split("\n"):
        stats["lines_seen"] += 1
        if len(line) < min_line_len:
            stats["lines_dropped_short"] += 1
            continue
        if factual_filter and line_is_probably_factual(line, FACTUAL_PATTERNS):
            stats["lines_dropped_factual"] += 1
            continue
        ids = sp.encode(line, out_type=int)
        if not ids:
            continue
        ids.append(eos_id)
        encoded.append(ids)
        stats["lines_kept"] += 1
        stats["tokens_encoded"] += len(ids)
    return {"encoded": encoded, "stats": stats}


def save_shard(
    out_dir: Path,
    shard_id: int,
    seq_len: int,
    sequences: list[np.ndarray],
) -> dict:
    arr = np.stack(sequences, axis=0).astype(np.uint16, copy=False)
    out_path = out_dir / f"phase1_shard_{shard_id:05d}.npy"
    np.save(out_path, arr, allow_pickle=False)
    return {
        "shard_id": shard_id,
        "path": str(out_path),
        "num_sequences": int(arr.shape[0]),
        "seq_len": seq_len,
        "dtype": str(arr.dtype),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    buffer: list[int] = []
    shard_sequences: list[np.ndarray] = []
    shard_index: list[dict] = []
    shard_id = 0

    stats = {
        "files_seen": 0,
        "files_skipped_path_filter": 0,
        "lines_seen": 0,
        "lines_dropped_short": 0,
        "lines_dropped_factual": 0,
        "lines_kept": 0,
        "tokens_encoded": 0,
        "tokens_packed": 0,
    }

    extensions = tuple(args.extensions)
    candidate_files = []
    for file_path in iter_text_files(args.input_dir, extensions=extensions):
        stats["files_seen"] += 1
        if args.factual_filter and should_skip_file(file_path, DEFAULT_BLOCKED_PATH_KEYWORDS):
            stats["files_skipped_path_filter"] += 1
            continue
        candidate_files.append(file_path)

    worker_results: Iterable[dict]
    if args.workers > 1 and len(candidate_files) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            worker_results = ex.map(
                process_file_worker,
                [str(p) for p in candidate_files],
                [str(args.tokenizer_model)] * len(candidate_files),
                [args.min_line_len] * len(candidate_files),
                [args.factual_filter] * len(candidate_files),
                chunksize=max(1, len(candidate_files) // (args.workers * 4)),
            )
            worker_results = list(worker_results)
    else:
        worker_results = [
            process_file_worker(
                file_path_str=str(p),
                tokenizer_model=str(args.tokenizer_model),
                min_line_len=args.min_line_len,
                factual_filter=args.factual_filter,
            )
            for p in candidate_files
        ]

    for result in worker_results:
        file_stats = result["stats"]
        for key in file_stats:
            stats[key] += file_stats[key]
        for ids in result["encoded"]:
            buffer.extend(ids)
            while len(buffer) >= args.seq_len:
                seq = np.asarray(buffer[: args.seq_len], dtype=np.uint16)
                del buffer[: args.seq_len]
                shard_sequences.append(seq)
                stats["tokens_packed"] += args.seq_len
                if len(shard_sequences) >= args.shard_sequences:
                    shard_meta = save_shard(
                        out_dir=args.output_dir,
                        shard_id=shard_id,
                        seq_len=args.seq_len,
                        sequences=shard_sequences,
                    )
                    shard_index.append(shard_meta)
                    shard_id += 1
                    shard_sequences = []

    if shard_sequences:
        shard_meta = save_shard(
            out_dir=args.output_dir,
            shard_id=shard_id,
            seq_len=args.seq_len,
            sequences=shard_sequences,
        )
        shard_index.append(shard_meta)

    manifest = {
        "config": {
            "input_dir": str(args.input_dir),
            "tokenizer_model": str(args.tokenizer_model),
            "output_dir": str(args.output_dir),
            "seq_len": args.seq_len,
            "shard_sequences": args.shard_sequences,
            "min_line_len": args.min_line_len,
            "workers": args.workers,
            "factual_filter": args.factual_filter,
            "extensions": args.extensions,
        },
        "stats": stats,
        "num_shards": len(shard_index),
        "shards": shard_index,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
