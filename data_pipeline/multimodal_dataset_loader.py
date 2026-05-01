import json
import wave
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav"}
VIDEO_EXTS = {".npy"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row {line_no} in {path}") from exc
    return rows


def _resolve_path(path: str | None, base_dir: Path) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return p


def _fit_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    if values.shape[-1] == target_len:
        return values
    if values.shape[-1] > target_len:
        return values[..., :target_len]
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (0, target_len - values.shape[-1])
    return np.pad(values, pad_width, mode="constant")


def _load_image(path: Path, size: int) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image loading: pip install pillow") from exc

    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size))
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_wav(path: Path, target_samples: int) -> torch.Tensor:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width {sample_width} in {path}")
    audio = audio.reshape(-1, channels).mean(axis=1, keepdims=True).T
    return torch.from_numpy(_fit_1d(audio, target_samples).copy())


def _load_video_npy(path: Path, frames: int, size: int) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 4:
        raise ValueError(f"video npy must have shape [frames, channels, height, width] or [frames, height, width, channels]: {path}")
    if arr.shape[1] in {1, 3}:
        video = arr.astype(np.float32, copy=False)
    elif arr.shape[-1] in {1, 3}:
        video = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)
    else:
        raise ValueError(f"cannot infer video channel dimension for {path}")
    if video.max() > 2.0:
        video = video / 127.5 - 1.0
    video = video[:frames]
    if video.shape[0] < frames:
        pad = np.zeros((frames - video.shape[0], video.shape[1], video.shape[2], video.shape[3]), dtype=np.float32)
        video = np.concatenate([video, pad], axis=0)
    if video.shape[-2:] != (size, size):
        # Avoid introducing a hard torchvision dependency for the smoke path.
        resized = []
        for frame in video:
            tensor = torch.from_numpy(frame).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
            resized.append(tensor.squeeze(0).numpy())
        video = np.stack(resized, axis=0)
    return torch.from_numpy(video.copy())


class MultimodalJSONLDataset(Dataset):
    """JSONL dataset for PCA multimodal smoke and small real-data runs."""

    def __init__(
        self,
        manifest_path: Path | str,
        tokenizer_model: Path | str,
        seq_len: int = 128,
        image_size: int = 224,
        audio_samples: int = 16000,
        video_frames: int = 8,
        video_frame_size: int = 224,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.rows = _read_jsonl(self.manifest_path)
        if not self.rows:
            raise ValueError(f"empty multimodal manifest: {self.manifest_path}")
        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.audio_samples = int(audio_samples)
        self.video_frames = int(video_frames)
        self.video_frame_size = int(video_frame_size)
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else 1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 2
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 3

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, text: str) -> torch.Tensor:
        ids = [self.bos_id] + self.sp.encode(text or "", out_type=int)[: max(self.seq_len - 2, 0)] + [self.eos_id]
        if len(ids) < self.seq_len:
            ids.extend([self.pad_id] * (self.seq_len - len(ids)))
        return torch.tensor(ids[: self.seq_len], dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source_text = row.get("text") or row.get("prompt") or row.get("input_text") or ""
        target_text = row.get("target_text") or row.get("answer") or row.get("caption") or source_text
        input_ids = self._encode(str(source_text))
        decoder_input_ids = self._encode(str(target_text))
        labels = decoder_input_ids.clone()
        labels[labels == self.pad_id] = -100

        item: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_id).float(),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": (decoder_input_ids != self.pad_id).float(),
            "labels": labels,
            "task": row.get("task", "mixed"),
        }

        image_path = _resolve_path(row.get("image_path"), self.base_dir)
        audio_path = _resolve_path(row.get("audio_path"), self.base_dir)
        video_path = _resolve_path(row.get("video_path"), self.base_dir)
        item["pixel_values"] = _load_image(image_path, self.image_size) if image_path else None
        item["audio_values"] = _load_wav(audio_path, self.audio_samples) if audio_path else None
        item["video_values"] = _load_video_npy(video_path, self.video_frames, self.video_frame_size) if video_path else None
        return item


def multimodal_collate(batch: Iterable[dict[str, Any]]) -> dict[str, Any]:
    batch = list(batch)
    out: dict[str, Any] = {}
    for key in ("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"):
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    for key in ("pixel_values", "audio_values", "video_values"):
        values = [item[key] for item in batch]
        present = [v for v in values if v is not None]
        if not present:
            out[key] = None
            continue
        prototype = present[0]
        filled = [v if v is not None else torch.zeros_like(prototype) for v in values]
        out[key] = torch.stack(filled, dim=0)
    out["tasks"] = [item.get("task", "mixed") for item in batch]
    return out


def create_multimodal_dataloader(
    manifest_path: Path | str,
    tokenizer_model: Path | str,
    batch_size: int,
    seq_len: int,
    image_size: int = 224,
    audio_samples: int = 16000,
    video_frames: int = 8,
    video_frame_size: int = 224,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    dataset = MultimodalJSONLDataset(
        manifest_path=manifest_path,
        tokenizer_model=tokenizer_model,
        seq_len=seq_len,
        image_size=image_size,
        audio_samples=audio_samples,
        video_frames=video_frames,
        video_frame_size=video_frame_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=multimodal_collate,
    )
