import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core.config import ModelConfig
from model_core.model import PCAModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a PCA checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--tokenizer-model", type=Path, required=True, help="Path to SentencePiece .model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _cfg_from_payload(payload: dict) -> ModelConfig:
    raw = payload["model_config"]
    allowed = {
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "max_seq_len",
        "ffn_dim",
        "ffn_kind",
        "reasoning_start_layer",
        "reasoning_ffn_dim",
        "reasoning_ffn_kind",
        "reasoning_moe_num_experts",
        "reasoning_moe_top_k",
        "reasoning_moe_num_groups",
        "reasoning_moe_groups_top_k",
        "reasoning_moe_gate_type",
        "reasoning_moe_expert_ffn_kind",
        "ffn_multiple_of",
        "rms_eps",
        "rope_theta",
        "dropout",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "cognitive_loops",
        "cognitive_num_experts",
        "cognitive_num_groups",
        "cognitive_groups_top_k",
        "cognitive_top_k",
        "cognitive_gate_type",
        "cognitive_aux_alpha",
        "cognitive_entropy_alpha",
        "cognitive_ffn_kind",
        "cognitive_ffn_dim",
        "gqa_layers",
        "lightning_end_layer",
        "mla_latent_dim",
    }
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return ModelConfig(**filtered)


@torch.no_grad()
def generate(
    model: PCAModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_id: int,
) -> torch.Tensor:
    out = input_ids
    for _ in range(max_new_tokens):
        logits = model(input_ids=out)["logits"][:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                vals, idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                probs = torch.softmax(vals, dim=-1)
                sampled_local = torch.multinomial(probs, num_samples=1)
                next_id = idx.gather(-1, sampled_local)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)
        if int(next_id.item()) == eos_id:
            break
    return out


def main() -> None:
    args = parse_args()

    try:
        payload = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location=args.device)

    cfg = _cfg_from_payload(payload)
    model = PCAModel(cfg).to(args.device).eval()
    model.load_state_dict(payload["model_state"])

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    ids = sp.encode(args.prompt, out_type=int)
    if not ids:
        raise SystemExit("Prompt encoded to empty token list.")

    input_ids = torch.tensor([ids], device=args.device, dtype=torch.long)
    out_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=sp.eos_id() if sp.eos_id() >= 0 else 2,
    )
    text = sp.decode(out_ids[0].tolist())
    report = {
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "generated_text": text,
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": int(out_ids.shape[1]),
        "new_tokens": int(out_ids.shape[1] - input_ids.shape[1]),
        "device": args.device,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
