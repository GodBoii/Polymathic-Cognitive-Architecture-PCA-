import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav"}
VIDEO_EXTS = {".npy"}
TEXT_EXTS = {".txt", ".md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simple PCA multimodal JSONL manifest.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-task", type=str, default="caption")
    parser.add_argument("--include-text-only", action="store_true")
    return parser.parse_args()


def read_sidecar_text(path: Path) -> str:
    for suffix in (".txt", ".md"):
        sidecar = path.with_suffix(suffix)
        if sidecar.exists():
            return "\n".join(sidecar.read_text(encoding="utf-8", errors="ignore").splitlines()).strip()
    return ""


def relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen_sidecars: set[Path] = set()

    for path in sorted(p for p in args.input_root.rglob("*") if p.is_file()):
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS:
            text = read_sidecar_text(path)
            for suffix in (".txt", ".md"):
                sidecar = path.with_suffix(suffix)
                if sidecar.exists():
                    seen_sidecars.add(sidecar.resolve())
            row = {
                "id": path.stem,
                "task": args.default_task,
                "text": text,
                "target_text": text,
            }
            if ext in IMAGE_EXTS:
                row["image_path"] = relative(path, args.output.parent)
            elif ext in AUDIO_EXTS:
                row["audio_path"] = relative(path, args.output.parent)
            else:
                row["video_path"] = relative(path, args.output.parent)
            rows.append(row)
        elif args.include_text_only and ext in TEXT_EXTS and path.resolve() not in seen_sidecars:
            text = "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()).strip()
            if text:
                rows.append(
                    {
                        "id": path.stem,
                        "task": "text",
                        "text": text,
                        "target_text": text,
                    }
                )

    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"manifest": str(args.output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
