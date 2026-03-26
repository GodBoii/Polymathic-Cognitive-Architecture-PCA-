import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PackedSequenceDataset(Dataset):
    def __init__(self, manifest_path: Path | str) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.shard_infos = self.manifest["shards"]
        self.seq_len = int(self.manifest["config"]["seq_len"])

        self._shards = [np.load(info["path"], mmap_mode="r") for info in self.shard_infos]
        self._offsets = []
        total = 0
        for shard in self._shards:
            self._offsets.append(total)
            total += int(shard.shape[0])
        self.total_sequences = total

    def __len__(self) -> int:
        return self.total_sequences

    def _locate(self, index: int) -> tuple[int, int]:
        # Binary search over shard offsets.
        lo, hi = 0, len(self._offsets) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self._offsets[mid]
            end = self._offsets[mid + 1] if mid + 1 < len(self._offsets) else self.total_sequences
            if start <= index < end:
                return mid, index - start
            if index < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.total_sequences:
            raise IndexError(index)

        shard_idx, local_idx = self._locate(index)
        tokens = self._shards[shard_idx][local_idx].astype(np.int64, copy=False)
        x = torch.from_numpy(tokens[:-1].copy())
        y = torch.from_numpy(tokens[1:].copy())
        attn = torch.ones_like(x, dtype=torch.float32)
        return {"input_ids": x, "labels": y, "attention_mask": attn}


def create_packed_dataloader(
    manifest_path: Path | str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = PackedSequenceDataset(manifest_path=manifest_path)
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs,
    )


class CUDAPrefetchLoader:
    def __init__(self, loader: DataLoader, device: str = "cuda") -> None:
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        it = iter(self.loader)
        next_batch = None

        def _move(batch):
            return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        with torch.cuda.stream(self.stream):
            try:
                next_batch = _move(next(it))
            except StopIteration:
                next_batch = None

        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            with torch.cuda.stream(self.stream):
                try:
                    next_batch = _move(next(it))
                except StopIteration:
                    next_batch = None
            yield batch
