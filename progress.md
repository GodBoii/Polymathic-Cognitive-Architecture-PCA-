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

---

## Latest Update (2026-04-11)

### New Experimental Signal
- `ablation_three_phase_v5_base` completed with:
  - high train/eval loss (`main_loss ~18.67`, `eval_loss ~20.83`)
  - very high perplexity cap in logs
  - healthy contraction trend (`step_delta_norms` strictly decreasing)
  - latent norm still elevated (`final_latent_norm ~74`)
- Interpretation: latent dynamics are now controlled better than before, but language modeling quality remains poor; we are likely over-regularizing relative to model capacity/data regime.

### New Runtime/Training Fix
- Fixed AMP scheduler-order warning in `train/train_step.py`:
  - root cause: in fp16, `GradScaler` can skip `optimizer.step()` on overflow, while scheduler was still stepping every iteration.
  - fix: step schedulers only when optimizer step actually happened (scale did not decrease after `scaler.update()`).
  - this prevents LR schedule drift and removes misleading warning noise.
- Added explicit eval telemetry for perplexity clipping:
  - `eval_perplexity_capped` is now logged when `eval_loss > 20` (because perplexity uses `exp(min(eval_loss, 20))`).
  - this avoids misreading repeated `4.851e8` values as a stable metric.

### Immediate Next Optimization Sequence
1. Re-run ablations after scheduler-step fix (fresh starts) to avoid stale comparisons.
2. Select best preset by combined criteria:
   - lower `eval_loss`
   - smaller `generalization_gap_ratio`
   - bounded `final_latent_norm`
   - monotonic or near-monotonic `step_delta_norms`
3. Train the best preset longer (same split/seed), then adjust one knob at a time:
   - first `imagination_aux_alpha`
   - then `imagination_norm_alpha`
   - then `imagination_update_scale`
4. Keep architecture fixed during this phase; focus on objective balancing.


**Analysis**

1. Your strongest idea is real, but it is narrower than it first appears. The “meaning index” and “relation index” are not the breakthrough: those are already approximated by embeddings and attention, exactly as your notes eventually conclude in [my_thoughts.md](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/my_thoughts.md):35 and [my_thoughts.md](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/my_thoughts.md):85. The genuinely novel part is the claim that comprehension, reasoning, and rendering should be separate substrates, with a latent workspace that refines before any token is emitted.

2. That claim is meaningful. Current LLMs mostly make “generation be the reasoning.” Your diagrams capture that well: the left side is one fused pipeline, the right side is a three-phase system with a non-token latent loop. Conceptually, this is the best part of the whole proposal. It points toward a world-model or latent-deliberation architecture rather than “just a better transformer.”

3. The current repo does not implement that architecture yet. It implements a standard autoregressive model with hybrid attention plus an appended recursive MoE block. The backbone is still token-space transformer layers, followed by final norm, then a cognitive block, then the LM head in [model.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/model_core/model.py):19, [model.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/model_core/model.py):20, [model.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/model_core/model.py):48, and [model.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/model_core/model.py):74. So architecturally it is still “LLM first, cognitive refinement second,” not “encoder -> latent imagination -> decoder.”

4. The recursive cognitive block is the closest thing here to your imagination idea, but it is still not a true imagination substrate. It repeatedly routes normalized token states through experts and adds the result back, as seen in [recursive_cognitive_block.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/cognitive_router/recursive_cognitive_block.py):209 and [recursive_cognitive_block.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/cognitive_router/recursive_cognitive_block.py):219. That means:
- it still operates on per-token hidden states, not on a separate global latent scene/model,
- it has no explicit halt criterion or convergence test,
- it has no decoder conditioned on a settled latent state,
- it mostly behaves like routed residual computation, not simulation/search.

5. That distinction matters. A real “imagination loop” needs a state that can be modified, tested, and stabilized independently of output tokens. Your current loop cannot really branch, simulate alternatives, verify hypotheses, or maintain an abstract answer-state. It can only transform the existing token representation harder.

6. The repo’s hybrid attention is interesting, but orthogonal to the core thesis. Layer-wise switching between GQA, Lightning, and MLA is configured in [config.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/model_core/config.py):39 and [config.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/model_core/config.py):49, and realized in the stack. That may help efficiency or specialization, but it does not solve the “separate reasoning substrate” problem. Also, the Lightning module explicitly calls itself a skeleton and uses a Python time-step loop in [lightning_attention.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/attention_modules/lightning/lightning_attention.py):10 and [lightning_attention.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/attention_modules/lightning/lightning_attention.py):94, so this codebase is still a research scaffold, not a validated systems result.

7. Training also confirms this. The model is optimized with ordinary next-token cross-entropy plus MoE auxiliary losses in [train_step.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/train/train_step.py):538 and [train_step.py](c:/Users/prajw/Downloads/Polymathic-Cognitive-Architecture-PCA-/train/train_step.py):540. That means the “cognitive” stage has no direct supervision for planning, simulation, latent consistency, or convergence. So the safest interpretation is: today it is a deeper, more specialized autoregressive model, not yet a model that thinks in a separate latent space.

8. The biggest conceptual risk in the original idea is hand-built meaning structure. If you explicitly encode “boy = human, male, young...” as a fixed semantic index, you reintroduce old symbolic-AI bottlenecks: ontology design, context dependence, ambiguity, maintenance, and grounding. Distributed learned representations beat hand-authored semantic tables precisely because meaning is contextual. So if you pursue Phase 1, it should be learned graph/slot construction, not manually curated attribute dictionaries.

9. The most promising next step is not “improve the current cognitive block a little.” It is to make the architecture genuinely three-part:
- Encoder: map prompt into a compact latent array or graph.
- Reasoner: recurrent latent update with cross-slot interaction, anchor to `z0`, and optional halt head.
- Decoder: autoregressive rendering conditioned on final latent state.
This would make your hypothesis falsifiable.

10. My bottom-line judgment: your notes are directionally right about the missing piece, and the repo is directionally pointed toward it, but the implementation has not crossed the conceptual boundary yet. Right now PCA is a hybrid-attention autoregressive transformer with a recursive routed post-processor. It is a good prototype for “deliberation on top of an LLM,” but not yet the architecture your diagrams argue for.

**Best research move**
Build the smallest possible true latent-workspace model next, then compare it against this repo at equal parameter/FLOP budget. If that model cannot beat the current baseline on tasks requiring hidden search, planning, or multi-step reasoning, then the imagination thesis is probably aesthetically compelling but not practically stronger. If it does win, that becomes the real foundation.