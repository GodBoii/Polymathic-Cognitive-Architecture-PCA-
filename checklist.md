# PCA Roadmap Checklist (From This Moment Forward)

This checklist is the execution plan from the current state of the repo. It is written to keep us focused, measurable, and aligned with your direction:
- Muon-only optimization
- true three-phase reasoning architecture
- multimodal training (text + images + audio + video)
- evaluation for factuality and calibration, not just fluency

## Phase 0: Lock Foundations (Immediate)

- [ ] Confirm Muon-only is enforced everywhere.
Context: We removed `adamw` as a selectable optimizer in the train loop and sweeper, and switched local/ablation presets to `muon`. This avoids split behavior and keeps experiments comparable.

- [ ] Freeze a single baseline preset for controlled comparisons.
Context: Pick one preset as the canonical baseline (`local_rtx3050_4gb_three_phase_v5_split.json`) and avoid changing many knobs at once.

- [ ] Re-run short smoke + ablation sanity after Muon switch.
Context: Optimizer changes can shift loss scale and stability. We need fresh metrics before making architecture conclusions.

## Phase 1: Stabilize Three-Phase Latent Reasoning (Short-Term)

- [ ] Run the 4-way three-phase ablation with `--fresh-start`.
Context: This tells us whether latent losses are helping or over-regularizing under Muon.

- [ ] Compare only key metrics for selection:
Context: `eval_loss`, `generalization_gap_ratio`, `final_latent_norm`, `convergence_ratio`, `delta_monotonic_decrease`.

- [ ] Select one best-balanced preset and run longer.
Context: First choose stability+capability balance; only then increase training duration.

- [ ] Keep architecture fixed while tuning objectives.
Context: During this phase, only tune `imagination_aux_alpha`, then `imagination_norm_alpha`, then `imagination_update_scale`.

## Phase 2: Add Multimodal Data Plane (Core Expansion)

- [ ] Create multimodal corpus layout and manifests.
Context: We need one unified sample format that can hold `text`, `image`, `audio`, and `video` references with alignment metadata.

- [ ] Build modality preprocessors:
Context: Tokenize text, encode images into visual tokens/features, audio into frame tokens/features, and video into temporal visual features.

- [ ] Build a multimodal batch builder.
Context: Each batch should include mixed-modality examples and masks, not text-only sequences.

- [ ] Add curriculum for modality ratios.
Context: Start text-heavy for stability, then increase image/audio/video mixture over time.

## Phase 3: Extend Architecture for Multimodal Reasoning

- [ ] Add modality-specific encoders.
Context: Text encoder exists; we need image/audio/video front-ends producing shared latent-space inputs.

- [ ] Add a shared cross-modal latent workspace.
Context: The imagination loop should refine a unified latent state that combines all modalities before rendering.

- [ ] Add modality-aware decoder heads.
Context: Keep language generation head, and optionally add retrieval/classification heads for visual/audio factual probes.

## Phase 4: Objectives and Alignment

- [ ] Keep LM loss, add multimodal grounding losses.
Context: Text next-token loss alone is insufficient for grounded world knowledge.

- [ ] Add cross-modal consistency losses.
Context: Enforce that text answers agree with image/audio/video evidence representations.

- [ ] Keep latent stability controls (RMS target + contraction) and track drift.
Context: Prevent unstable latent growth when modalities become noisier.

## Phase 5: Evaluation and Honest Readiness Gates

- [ ] Add multimodal benchmark evaluation track.
Context: Include WorldVQA-style visual factual checks and calibration metrics (ECE/slope style reporting) for confidence honesty.
Reference: https://www.kimi.com/blog/worldvqa

- [ ] Define release gates before “model ready” claim.
Context: Model is ready only if it passes both capability and reliability thresholds.

- [ ] Release candidate process.
Context: fixed config, fixed data split, fixed seed set, reproducible command log, checkpoint snapshot, and report.

## Ready Criteria (Proposed)

- [ ] Reproducible runs: same config/seed produce tightly similar eval trends.
- [ ] Stable latent behavior: bounded `final_latent_norm`, no collapse in convergence metrics.
- [ ] Generalization: eval loss trend consistently improves on disjoint splits.
- [ ] Multimodal factuality: meaningful lift over text-only baseline on visual/audio/video factual tasks.
- [ ] Calibration: confidence quality improves, not only raw accuracy.

## What We Will Not Do

- [ ] No architecture thrashing.
Context: Do not simultaneously change optimizer, architecture depth, data distribution, and losses in one run.

- [ ] No readiness claims from tiny smoke metrics.
Context: Smoke runs are for breakage detection, not intelligence claims.
