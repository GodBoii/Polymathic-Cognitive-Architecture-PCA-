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
