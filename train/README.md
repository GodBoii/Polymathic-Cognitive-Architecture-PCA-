# Task 6: Training Harness

Run a short smoke train:

```powershell
& .\lrm-matrix\Scripts\python.exe .\train\train_step.py `
  --steps 5 `
  --optimizer muon `
  --precision bf16 `
  --micro-batch-size 2 `
  --seq-len 128 `
  --grad-accum-steps 2
```

Resume from checkpoint:

```powershell
& .\lrm-matrix\Scripts\python.exe .\train\train_step.py `
  --resume .\train\checkpoints\last.pt `
  --steps 20
```

Key metrics emitted each step as JSON:
- `loss`
- `grad_norm`
- `tokens_per_sec`
- `step_time_sec`
- `gpu_mem_gb`
- `total_tokens`
