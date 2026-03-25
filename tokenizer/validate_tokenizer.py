import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trained SentencePiece tokenizer artifacts.")
    parser.add_argument("--model", type=Path, required=True, help="Path to .model file")
    parser.add_argument("--sample-text", type=str, default="Evaluate this API plan and return strict JSON output.")
    parser.add_argument("--max-preview-pieces", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    sp = spm.SentencePieceProcessor(model_file=str(args.model))

    piece_size = sp.get_piece_size()
    token_ids = sp.encode(args.sample_text, out_type=int)
    pieces = sp.encode(args.sample_text, out_type=str)
    decoded = sp.decode(token_ids)

    preview = [sp.id_to_piece(i) for i in range(min(piece_size, args.max_preview_pieces))]

    report = {
        "model": str(args.model),
        "vocab_size": piece_size,
        "sample_text": args.sample_text,
        "encoded_ids": token_ids,
        "encoded_pieces": pieces,
        "decoded_text": decoded,
        "roundtrip_match": decoded == args.sample_text,
        "preview_pieces": preview,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
