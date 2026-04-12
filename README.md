# PCA: Progressive Cognitive Architecture

A research-oriented transformer language model implementation featuring multiple attention mechanisms, hierarchical mixture-of-experts routing, and a recursive cognitive processing block.

## Architecture Overview

PCA implements a hybrid transformer architecture with the following key components:

### Multi-Attention Strategy
The model uses different attention mechanisms across layers:
- **GQA (Grouped Query Attention) with RoPE**: Standard attention with rotary position embeddings for early layers
- **Lightning Attention**: Linear attention variant using causal prefix sums for middle layers
- **MLA (Multi-Latent Attention)**: Compressed KV cache attention with shared latent projections for later layers

### Recursive Cognitive Block
A novel post-transformer processing stage that applies hierarchical mixture-of-experts routing through multiple cognitive loops:
- Hierarchical two-level routing (macro groups → micro experts)
- Configurable number of recursive processing loops
- Load balancing and entropy regularization
- Expert usage tracking and co-activation analysis

### Feed-Forward Networks
- Standard MLP with configurable activation
- SwiGLU activation (default)
- Optional MoE layers for reasoning-focused layers

## Project Structure

```
├── attention_modules/          # Multiple attention mechanism implementations
│   ├── gqa_rope/              # Grouped Query Attention with RoPE
│   ├── lightning/             # Linear attention variant
│   ├── mla/                   # Multi-Latent Attention
│   └── registry.py            # Attention layer selection logic
├── cognitive_router/          # Recursive cognitive processing with MoE
│   └── recursive_cognitive_block.py
├── model_core/                # Core transformer components
│   ├── model.py              # Main PCAModel class
│   ├── config.py             # ModelConfig dataclass
│   ├── block.py              # TransformerBlock implementation
│   ├── ffn.py                # Feed-forward network variants
│   └── norms.py              # RMSNorm implementation
├── data_pipeline/            # Dataset preparation and loading
│   ├── build_phase1_dataset.py  # Tokenize and pack sequences
│   ├── dataset_loader.py        # PyTorch dataset for packed sequences
│   ├── filters.py               # Content filtering utilities
│   └── artifacts/               # Generated dataset shards
├── train/                    # Training infrastructure
│   ├── train_step.py         # Main training loop
│   ├── muon.py              # Muon optimizer implementation
│   └── pretrain_phase1_52l_v5.jsonl  # Training configuration
├── tokenizer/               # SentencePiece tokenizer training
│   ├── train_sentencepiece.py
│   ├── validate_tokenizer.py
│   └── README.md
├── scripts/                 # Utility scripts
│   ├── launch_pretrain.py   # Training launcher with checkpointing
│   ├── sample_generate.py   # Text generation
│   └── smoke_test_*.py      # Component tests
└── raw_data_corpus/         # Training corpus
    ├── code_and_systems/    # Programming and technical docs
    └── language_structure/  # Natural language examples
```

## Key Features

### 1. Hybrid Attention Mechanisms
Different attention types are selected based on layer index:
- Layers 0-3: GQA with RoPE (standard attention)
- Layers 4-15: Lightning Attention (linear complexity)
- Layers 16+: MLA (compressed KV cache)

### 2. Hierarchical MoE Routing
The cognitive block uses two-level routing:
- **Macro router**: Selects top-k expert groups
- **Micro routers**: Select top-k experts within each group
- Supports both softmax and sigmoid gating
- Includes load balancing and entropy regularization losses

### 3. Packed Sequence Training
Efficient training on variable-length documents:
- Documents tokenized and packed into fixed-length sequences
- No padding waste
- Sharded storage for large datasets
- Manifest-based indexing

### 4. Flexible Configuration
Extensive configuration options via `ModelConfig`:
- Model dimensions and layer counts
- Attention mechanism boundaries
- MoE parameters (experts, groups, top-k)
- Cognitive loop iterations
- FFN dimensions and activation types

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- SentencePiece >= 0.1.99

## Quick Start

### 1. Train Tokenizer

```bash
python tokenizer/train_sentencepiece.py \
  --input-dir ./raw_data_corpus \
  --output-dir ./tokenizer/artifacts \
  --model-prefix pca_tokenizer \
  --vocab-size 32000 \
  --min-english-ratio 0.85 \
  --ascii-only
```

### 2. Build Training Dataset

```bash
python data_pipeline/build_phase1_dataset.py \
  --input-dir ./raw_data_corpus \
  --tokenizer-model ./tokenizer/artifacts/pca_tokenizer.model \
  --output-dir ./data_pipeline/artifacts/phase1 \
  --seq-len 8192 \
  --shard-sequences 1024 \
  --workers 8
```

### 3. Launch Training

```bash
python scripts/launch_pretrain.py \
  --config train/pretrain_phase1_52l_v5.jsonl \
  --checkpoint-dir ./checkpoints \
  --keep-last 3
```

## Model Configuration

Example configuration for a 52-layer model:

```python
from model_core.config import ModelConfig

cfg = ModelConfig(
    vocab_size=32000,
    d_model=2048,
    n_layers=52,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=8192,
    ffn_kind="swiglu",
    
    # Attention boundaries
    gqa_layers=4,
    lightning_end_layer=16,
    mla_latent_dim=512,
    
    # Cognitive block
    cognitive_loops=5,
    cognitive_num_experts=8,
    cognitive_num_groups=2,
    cognitive_top_k=2,
    cognitive_groups_top_k=1,
    cognitive_gate_type="sigmoid",
    cognitive_aux_alpha=0.01,
    cognitive_entropy_alpha=0.001,
)
```

## Training Details

### Optimizer
- Muon optimizer for all trainable parameters
- Orthogonalization via Newton-Schulz iteration

### Loss Components
- Cross-entropy language modeling loss
- MoE load balancing loss (cognitive block)
- Entropy regularization (cognitive block)

### Data Pipeline
- Packed sequences (no padding)
- Fixed sequence length (8192 tokens)
- Sharded storage with manifest indexing
- Optional factual content filtering

## Inference

```python
import torch
from model_core.model import PCAModel
from model_core.config import ModelConfig

# Load model
cfg = ModelConfig()
model = PCAModel(cfg)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

# Generate
input_ids = torch.tensor([[1, 2, 3, 4]])  # Your token IDs
with torch.no_grad():
    output = model(input_ids)
    logits = output["logits"]
```

## Testing

Run component tests:

```bash
python scripts/smoke_test_task3.py  # Tokenizer
python scripts/smoke_test_task4.py  # Attention modules
python scripts/smoke_test_task5.py  # Model forward pass
python scripts/smoke_test_task7.py  # Cognitive router
python scripts/smoke_test_task8.py  # Training step
python scripts/smoke_test_task9.py  # Data pipeline
```

## Architecture Details

### Attention Layer Selection
```python
def get_attention_kind(cfg, layer_idx):
    if layer_idx < cfg.gqa_layers:
        return "gqa"
    elif layer_idx < cfg.lightning_end_layer:
        return "lightning"
    else:
        return "mla"
```

### Cognitive Block Flow
1. Input normalization (RMSNorm)
2. Hierarchical routing (macro → micro)
3. Expert application with weighted mixing
4. Residual connection
5. Repeat for N cognitive loops

### MoE Routing
- **Load Balance Loss**: Encourages uniform expert usage
- **Entropy Regularization**: Prevents routing collapse
- **Co-activation Tracking**: Monitors expert interaction patterns

## Performance Considerations

- **Memory**: MLA reduces KV cache size via latent compression
- **Speed**: Lightning attention provides linear complexity for middle layers
- **Efficiency**: Packed sequences eliminate padding overhead
- **Scalability**: Hierarchical MoE routing scales to many experts

## Research Features

- Multiple attention mechanisms in a single model
- Recursive cognitive processing with MoE
- Hierarchical expert routing
- Comprehensive routing statistics and telemetry
- Flexible layer-wise configuration

## License

This is a research project. Check with the repository owner for licensing details.

## Citation

If you use this code in your research, please cite appropriately.

## Contributing

This is a research codebase. Contributions should maintain the experimental nature while ensuring reproducibility.
