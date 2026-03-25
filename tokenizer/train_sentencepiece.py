import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List

import sentencepiece as spm


ALLOWED_PUNCT = set(
    r""" !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
)


def iter_text_files(root: Path, extensions: Iterable[str]) -> Iterable[Path]:
    ext_set = {ext.lower() for ext in extensions}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            yield path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive whitespace while keeping line boundaries.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def english_ratio(line: str) -> float:
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return 1.0
    latin = [ch for ch in letters if "LATIN" in unicodedata.name(ch, "")]
    return len(latin) / len(letters)


def filter_line(line: str, min_english_ratio: float, ascii_only: bool) -> bool:
    if not line:
        return False
    if english_ratio(line) < min_english_ratio:
        return False
    if ascii_only:
        for ch in line:
            if ch == "\n":
                continue
            if ord(ch) < 128:
                continue
            return False
    return True


def sanitize_for_english_logic(line: str) -> str:
    # Keep a constrained symbol space for English + JSON/API/code syntax.
    out = []
    for ch in line:
        if ch.isalnum() or ch.isspace() or ch in ALLOWED_PUNCT:
            out.append(ch)
    return "".join(out).strip()


def build_corpus(
    input_dir: Path,
    corpus_out: Path,
    extensions: List[str],
    min_english_ratio: float,
    ascii_only: bool,
    min_line_len: int,
) -> dict:
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    files_seen = 0
    lines_written = 0
    lines_dropped = 0

    with corpus_out.open("w", encoding="utf-8") as out_f:
        for file_path in iter_text_files(input_dir, extensions):
            files_seen += 1
            try:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            normalized = normalize_text(raw)
            for line in normalized.split("\n"):
                line = sanitize_for_english_logic(line)
                if len(line) < min_line_len:
                    lines_dropped += 1
                    continue
                if not filter_line(line, min_english_ratio=min_english_ratio, ascii_only=ascii_only):
                    lines_dropped += 1
                    continue
                out_f.write(line + "\n")
                lines_written += 1

    return {
        "files_seen": files_seen,
        "lines_written": lines_written,
        "lines_dropped": lines_dropped,
        "corpus_path": str(corpus_out),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an English-first SentencePiece BPE tokenizer.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw text files.")
    parser.add_argument("--output-dir", type=Path, default=Path("tokenizer/artifacts"), help="Output directory.")
    parser.add_argument("--model-prefix", type=str, default="pca_tokenizer", help="Tokenizer model prefix.")
    parser.add_argument("--vocab-size", type=int, default=32000, help="SentencePiece vocabulary size.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".txt", ".md", ".json", ".csv"],
        help="File extensions to include.",
    )
    parser.add_argument("--min-english-ratio", type=float, default=0.85, help="Min fraction of Latin letters per line.")
    parser.add_argument("--ascii-only", action="store_true", help="Keep only ASCII lines.")
    parser.add_argument("--min-line-len", type=int, default=8, help="Drop lines shorter than this.")
    parser.add_argument("--character-coverage", type=float, default=1.0, help="SentencePiece character coverage.")
    parser.add_argument("--input-sentence-size", type=int, default=2000000, help="Max sampled lines for training.")
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=[
            "<tool_call>",
            "</tool_call>",
            "<tool_result>",
            "</tool_result>",
            "<json>",
            "</json>",
        ],
        help="User-defined symbols for tool and schema routing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = args.output_dir / f"{args.model_prefix}.corpus.txt"
    stats = build_corpus(
        input_dir=args.input_dir,
        corpus_out=corpus_path,
        extensions=args.extensions,
        min_english_ratio=args.min_english_ratio,
        ascii_only=args.ascii_only,
        min_line_len=args.min_line_len,
    )

    model_prefix_path = args.output_dir / args.model_prefix
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix_path),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        split_digits=True,
        byte_fallback=True,
        normalization_rule_name="nfkc",
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
        user_defined_symbols=args.special_tokens,
    )

    metadata = {
        "config": {
            "vocab_size": args.vocab_size,
            "model_type": "bpe",
            "character_coverage": args.character_coverage,
            "min_english_ratio": args.min_english_ratio,
            "ascii_only": args.ascii_only,
            "min_line_len": args.min_line_len,
            "extensions": args.extensions,
            "special_tokens": args.special_tokens,
        },
        "corpus_stats": stats,
        "artifacts": {
            "model": str(model_prefix_path.with_suffix(".model")),
            "vocab": str(model_prefix_path.with_suffix(".vocab")),
            "corpus": str(corpus_path),
        },
    }

    metadata_path = args.output_dir / f"{args.model_prefix}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
