# Progress Log: Three-Phase PCA Research

## Goal
Build and validate a **three-phase language model architecture** that separates:
1. **Comprehension (Encoder)**
2. **Imagination / latent workspace refinement**
3. **Rendering (Decoder)**

instead of relying on a single fused autoregressive path where generation and reasoning are entangled.

---

## What We Are Trying To Achieve
- Prove the architecture is executable on local hardware (RTX 3050 4GB).
- Improve latent workspace behavior from unstable/drifting to contractive and controlled.
- Evaluate honestly using disjoint train/eval manifests (avoid train-eval leakage).
- Find a practical stability-vs-capability balance via reproducible ablations.

---

## Major Work Completed

### 1) Environment and Runtime Bring-up
- Resolved CPU-only PyTorch issue and confirmed CUDA-enabled torch in local env.
- Added safer launcher/training fallback logic for device/precision mismatch.

### 2) Data Pipeline Reliability
- Fixed manifest path portability issues in dataset loader.
- Added `data_pipeline/create_manifest_split.py` to create disjoint train/eval splits from existing packed manifests.
- Added crop-based short-window sampling in `data_pipeline/dataset_loader.py`:
  - Train: random crop
  - Eval: deterministic center crop
  - Enables local training from long packed shards (e.g., seq_len 8192 manifests) at short context (e.g., 128).

### 3) Model Architecture Evolution
- Kept legacy autoregressive PCA path intact.
- Added a new model family: `three_phase`:
  - Encoder blocks
  - Latent imagination workspace with iterative updates
  - Decoder with causal self-attn + cross-attn to latent workspace
- Added model factory (`build_model`) and architecture switch through configs/presets.

### 4) Latent Workspace Stabilization
- Added latent auxiliary losses:
  - stability loss
  - consistency loss
  - norm loss
  - contraction loss
- Added learned update gates:
  - `self_attn_gate`
  - `ffn_gate`
- Added step-dependent damping:
  - `imagination_step_decay`
- Added full telemetry:
  - step delta norms
  - convergence ratio
  - monotonic decrease flag
  - gate values
  - generalized train/eval gap metrics

### 5) Launcher and Resume Robustness
- Fixed launcher edge case when child run emits only final summary payload.
- Added `--fresh-start` mode to bypass accidental auto-resume from `last.pt`.

### 6) Preset and Experiment Infrastructure
- Added multiple local presets for RTX 3050 experimentation.
- Added real-data local split preset:
  - `train/presets/local_rtx3050_4gb_three_phase_v5_split.json`
- Added ablation presets:
  - `ablation_three_phase_v5_base.json`
  - `ablation_three_phase_v5_low_aux.json`
  - `ablation_three_phase_v5_low_norm.json`
  - `ablation_three_phase_v5_higher_update.json`
- Added ablation runner:
  - `scripts/run_three_phase_ablation.py`

---

## What We Observed So Far

1. On tiny smoke data, model can memorize and produce misleadingly good metrics.
2. After introducing real disjoint splits, metrics became honest:
   - train/eval gaps exposed
   - latent stability issues became visible
3. Stabilization changes significantly reduced latent norm explosion.
4. Stronger contraction controls improved latent smoothness but can hurt language performance if too restrictive.
5. Current bottleneck: balancing regularization strength vs model capability.

---

## Current Hypothesis
- We are no longer blocked by infra.
- The main research question now is:
  **How much latent stabilization is enough to keep the workspace well-conditioned without suppressing useful representational power?**

---

## Recommended Next Steps

1. Run the four-preset ablation grid with fresh starts.
2. Compare using the same seed and split:
   - `eval_loss`
   - `generalization_gap` / `generalization_gap_ratio`
   - `final_latent_norm`
   - `convergence_ratio`
   - `delta_monotonic_decrease`
3. Pick best-balanced setting and train longer on the same split.
4. Only after balance is found, consider deeper model complexity changes.

---

## How To Reproduce Quickly

### A) Build train/eval split from phase1_v5
```powershell
python .\data_pipeline\create_manifest_split.py `
  --source-manifest .\data_pipeline\artifacts\phase1_v5\manifest.json `
  --train-output-dir .\data_pipeline\artifacts\phase1_v5_local_train `
  --eval-output-dir .\data_pipeline\artifacts\phase1_v5_local_eval `
  --eval-ratio 0.1 `
  --seed 42 `
  --train-shard-sequences 256 `
  --eval-shard-sequences 64
```

### B) Run the main real-data local preset
```powershell
python .\scripts\launch_pretrain.py --config .\train\presets\local_rtx3050_4gb_three_phase_v5_split.json --fresh-start
```

### C) Run full ablation grid
```powershell
python .\scripts\run_three_phase_ablation.py --fresh-start
```

---

## Important Context For A New LLM Chat
- This project is not at idea stage anymore.
- There is a working three-phase model and working local GPU training path.
- Key challenge is **objective balancing**, not basic implementation.
- Avoid redoing environment/bootstrap fixes unless something regresses.
- Prefer controlled ablations over one-off manual hyperparameter changes.
