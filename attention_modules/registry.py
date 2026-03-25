from attention_modules.gqa_rope import GQAAttention
from attention_modules.lightning import LightningAttention
from attention_modules.mla import MLAAttention


def get_attention_kind(cfg, layer_idx: int) -> str:
    if layer_idx < cfg.gqa_layers:
        return "gqa"
    if layer_idx < cfg.lightning_end_layer:
        return "lightning"
    return "mla"


def build_attention(cfg, layer_idx: int):
    kind = get_attention_kind(cfg, layer_idx)
    if kind == "gqa":
        return GQAAttention(cfg=cfg, bias=False)
    if kind == "lightning":
        return LightningAttention(cfg=cfg, bias=False)
    return MLAAttention(cfg=cfg, bias=False)
