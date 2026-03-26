# Task 2: English-First Tokenizer Pipeline

This folder contains the initial tokenizer implementation for PCA.

## 1) Verify environment

```powershell
& .\lrm-matrix\Scripts\python.exe .\scripts\check_torch_cuda.py
```

## 2) Prepare corpus location

Place raw text files under a directory (example: `tokenizer\data\raw`).

Supported extensions by default:
- `.txt`
- `.md`
- `.json`
- `.csv`

## 3) Train SentencePiece BPE (32k)

```powershell
& .\lrm-matrix\Scripts\python.exe .\tokenizer\train_sentencepiece.py `
  --input-dir .\tokenizer\data\raw `
  --output-dir .\tokenizer\artifacts `
  --model-prefix pca_tokenizer `
  --vocab-size 32000 `
  --min-english-ratio 0.85 `
  --ascii-only `
  --allow-math-unicode
```

To keep extra symbols beyond the curated math set, add:

```powershell
--extra-unicode-chars "∝∴⊕⊗"
```

Outputs:
- `tokenizer/artifacts/pca_tokenizer.model`
- `tokenizer/artifacts/pca_tokenizer.vocab`
- `tokenizer/artifacts/pca_tokenizer.corpus.txt`
- `tokenizer/artifacts/pca_tokenizer.metadata.json`

## 4) Validate tokenizer

```powershell
& .\lrm-matrix\Scripts\python.exe .\tokenizer\validate_tokenizer.py `
  --model .\tokenizer\artifacts\pca_tokenizer.model
```
