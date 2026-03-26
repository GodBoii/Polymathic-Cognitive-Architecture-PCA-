# Task 9: Phase-1 Data Pipeline

## 1) Build packed shards (fixed-length sequences)

```powershell
& .\lrm-matrix\Scripts\python.exe .\data_pipeline\build_phase1_dataset.py `
  --input-dir .\tokenizer\data\raw `
  --tokenizer-model .\tokenizer\artifacts\pca_tokenizer.model `
  --output-dir .\data_pipeline\artifacts\phase1 `
  --seq-len 8192 `
  --shard-sequences 1024 `
  --factual-filter
```

Output:
- `data_pipeline/artifacts/phase1/phase1_shard_*.npy`
- `data_pipeline/artifacts/phase1/manifest.json`

## 2) Load packed dataset in training

Use `PackedSequenceDataset` or `create_packed_dataloader` from:
- `data_pipeline/dataset_loader.py`

Each sample returns:
- `input_ids`: `[seq_len-1]`
- `labels`: `[seq_len-1]` (next-token shifted)
- `attention_mask`: ones `[seq_len-1]`
