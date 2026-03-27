# Task 6: Training Harness

Run a short smoke train:

```powershell
& .\lrm-matrix\Scripts\python.exe .\train\train_step.py `
  --steps 5 `
  --optimizer muon `
  --precision bf16 `
  --micro-batch-size 2 `
  --seq-len 128 `
  --grad-accum-steps 2 `
  --cognitive-num-experts 8 `
  --cognitive-top-k 4 `
  --cognitive-gate-type sigmoid `
  --log-router-stats-every 1 `
  --collapse-cv-threshold 0.5 `
  --collapse-entropy-threshold 0.25 `
  --adaptive-aux-alpha `
  --aux-alpha-max 0.2 `
  --aux-alpha-up-mult 1.5 `
  --aux-alpha-decay 0.95
```

Resume from checkpoint:

```powershell
& .\lrm-matrix\Scripts\python.exe .\train\train_step.py `
  --resume .\train\checkpoints\last.pt `
  --steps 20
```

Phase-2 reasoning specialization (freeze bottom 20 layers, train top 32):

```powershell
& .\lrm-matrix\Scripts\python.exe .\train\train_step.py `
  --resume .\train\checkpoints_phase1_52l_bulk\last.pt `
  --resume-model-only `
  --n-layers 52 `
  --d-model 3072 `
  --n-heads 24 `
  --n-kv-heads 8 `
  --ffn-dim 2048 `
  --reasoning-start-layer 20 `
  --freeze-layers-below 20 `
  --gqa-layers 12 `
  --lightning-end-layer 36 `
  --mla-latent-dim 768 `
  --steps 15000
```

Launcher preset flow:

```powershell
& .\lrm-matrix\Scripts\python.exe .\scripts\launch_pretrain.py --config .\train\presets\phase1_52l_bulk.json
& .\lrm-matrix\Scripts\python.exe .\scripts\launch_pretrain.py --config .\train\presets\phase2_52l_reasoning.json
```

V5 MoE-reasoning preset flow:

```powershell
& .\lrm-matrix\Scripts\python.exe .\scripts\launch_pretrain.py --config .\train\presets\phase1_52l_v5.json
& .\lrm-matrix\Scripts\python.exe .\scripts\launch_pretrain.py --config .\train\presets\phase2_52l_v5.json
```

Key metrics emitted each step as JSON:
- `loss`
- `main_loss`
- `moe_aux_loss`
- `moe_load_balance_loss`
- `moe_entropy_reg_loss`
- `grad_norm`
- `tokens_per_sec`
- `step_time_sec`
- `gpu_mem_gb`
- `total_tokens`
- `router_entropy` (when telemetry is enabled)
- `expert_usage` (when telemetry is enabled)
- `co_activation` (when telemetry is enabled)
- `expert_usage_cv` (when telemetry is enabled)
- `warning` (when collapse thresholds are crossed)
- `current_aux_alpha`
- `aux_alpha_action` / `prev_aux_alpha` (when adaptive control is enabled)

Analyze MoE telemetry from a log file:

```powershell
& .\lrm-matrix\Scripts\python.exe .\train\moe_telemetry_dashboard.py `
  --log-file .\train\run.jsonl
```
