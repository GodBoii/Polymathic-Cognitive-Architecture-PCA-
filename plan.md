# PCA Native Multimodal Build Plan

Current date: 2026-05-01

This is the operator guide from the current repo state to a complete native multimodal PCA model. It covers what data to collect, where to save it, how to train the tokenizer, how to build manifests, how to run smoke tests, how to overfit tiny data, how to scale to cloud GPUs, and what must be proven before spending serious H100 money.

## 0. Truthful Status

The current repo is a runnable research prototype, not yet a trained foundation model.

What works now:

- Text-only autoregressive PCA path.
- Three-phase text path: encoder -> latent imagination workspace -> decoder.
- Multimodal PCA scaffold: text, image tensors, audio tensors, video tensors -> shared latent workspace -> text logits plus image/audio/video patch heads.
- Packed text dataset builder and loader.
- JSONL multimodal manifest builder.
- JSONL multimodal dataset loader.
- Tiny multimodal training harness.
- Muon optimizer implementation.
- CUDA smoke tests.

What does not work yet:

- No serious aligned multimodal training data has been added.
- Image/audio/video outputs are patch smoke heads, not real generation heads.
- No image/audio/video codec-token training yet.
- No multimodal eval suite yet.
- No distributed FSDP/ZeRO H100 training script yet.
- The latent imagination loop is not proven to reason. It is a trainable workspace that still needs grounded objectives and evals.

The target is native multimodal training, not bolt-on multimodal. That means the main model must learn with mixed modality data during core training, not learn text first and only later receive a vision/audio adapter.

## 1. Correct Optimizer Plan: Muon Stack

Use Muon for hidden-layer 2D matrices. Use AdamW only for parameters Muon should not optimize.

Correct optimizer split:

```text
Muon:
  transformer linear weights
  attention projection weights
  FFN weights
  latent workspace 2D weights
  image/audio/video projection matrices when 2D

AdamW:
  token embeddings
  modality embeddings
  positional embeddings
  output/lm heads if untied
  RMSNorm weights
  biases
  scalar gates
  1D parameters
```

Current repo note:

- `train/muon.py` currently applies Muon-style momentum to all parameters and Newton-Schulz only to 2D tensors.
- This is okay for smoke tests.
- Before serious H100 training, update the optimizer builder to create separate Muon and AdamW parameter groups.

Memory planning with Muon:

```text
Muon hidden weights, no fp32 master:
  bf16 parameter  2 bytes
  bf16 gradient   2 bytes
  fp32 momentum   4 bytes
  total           ~8 bytes / hidden parameter

Muon hidden weights, with fp32 master:
  bf16 parameter  2 bytes
  bf16 gradient   2 bytes
  fp32 momentum   4 bytes
  fp32 master     4 bytes
  total           ~12 bytes / hidden parameter
```

Practical H100 80GB planning:

```text
150M-700M proof: 1x H100 80GB
1B native multimodal: 1x H100 possible, 2x H100 better
2B native multimodal: 2-4x H100
3B native multimodal: 4x H100 minimum, 8x H100 comfortable
```

## 2. Repo Paths And Data Layout

Keep all real training data out of `tmp/`. Use this layout:

```text
raw_data_corpus/
  code_and_systems/
  language_structure/
  reasoning/
  math/
  safety/

raw_data_multimodal/
  images/
    train/
      image_000001.jpg
      image_000001.txt
    eval/
      image_eval_000001.jpg
      image_eval_000001.txt
  audio/
    train/
      audio_000001.wav
      audio_000001.txt
    eval/
      audio_eval_000001.wav
      audio_eval_000001.txt
  video/
    train/
      video_000001.npy
      video_000001.txt
    eval/
      video_eval_000001.npy
      video_eval_000001.txt
  mixed_jsonl/
    train.jsonl
    eval.jsonl

tokenizer/artifacts/
  pca_tokenizer.model
  pca_tokenizer.vocab
  pca_tokenizer.metadata.json

data_pipeline/artifacts/
  phase1_text_train/
  phase1_text_eval/
  multimodal_train/
    manifest.jsonl
  multimodal_eval/
    manifest.jsonl
  multimodal_overfit16/
    manifest.jsonl

train/checkpoints/
  text_baseline/
  multimodal_overfit16/
  multimodal_150m/
  multimodal_700m/
  multimodal_1b/
  multimodal_3b/

train/logs/
  text_baseline.jsonl
  multimodal_overfit16.jsonl
  multimodal_150m.jsonl
```

Important rules:

- Save raw text under `raw_data_corpus/`.
- Save raw image/audio/video data under `raw_data_multimodal/`.
- Save generated manifests and tensor shards under `data_pipeline/artifacts/`.
- Save checkpoints under `train/checkpoints/`.
- Save temporary smoke artifacts under `tmp/`.
- Never mix eval examples into train folders.

## 3. Data Needed

### 3.1 Text Data

Minimum local text data:

```text
raw_data_corpus/code_and_systems/
raw_data_corpus/language_structure/
raw_data_corpus/reasoning/
raw_data_corpus/math/
raw_data_corpus/safety/
```

What to include:

- Clean English prose.
- Code documentation.
- Python, JavaScript, web development examples.
- Math reasoning examples.
- Instruction/answer pairs.
- Safety and refusal examples.
- Tool-use transcripts if you want agent behavior later.

Avoid:

- Copyrighted books copied wholesale without rights.
- Private personal data.
- Duplicates.
- Corrupt OCR.
- Low-quality scraped noise.

### 3.2 Image Data

For each image, store a sidecar caption/instruction file with the same stem:

```text
raw_data_multimodal/images/train/image_000001.jpg
raw_data_multimodal/images/train/image_000001.txt
```

Sidecar text examples:

```text
Describe the image in detail.
The image shows a red bicycle leaning against a brick wall.
```

Recommended early tasks:

- Image captioning.
- VQA.
- OCR-heavy questions.
- UI screenshot to HTML/CSS description.
- Chart and diagram explanation.
- Spatial relation questions.

### 3.3 Audio Data

Use WAV first because the current loader supports `.wav`.

```text
raw_data_multimodal/audio/train/audio_000001.wav
raw_data_multimodal/audio/train/audio_000001.txt
```

Sidecar text examples:

```text
Transcribe the audio.
The speaker says: welcome to the project demo.
```

Recommended early tasks:

- ASR.
- Audio captioning.
- Speaker emotion labels.
- Sound event explanation.
- Intent from voice tone.

### 3.4 Video Data

The current loader supports `.npy` video tensors first. Save video arrays as:

```text
[frames, channels, height, width]
```

or:

```text
[frames, height, width, channels]
```

Example path:

```text
raw_data_multimodal/video/train/video_000001.npy
raw_data_multimodal/video/train/video_000001.txt
```

Sidecar text examples:

```text
Describe what happens in the video.
A person opens a code editor, edits a button color, and refreshes the browser.
```

Recommended early tasks:

- Video captioning.
- Temporal order questions.
- Screen recording to frontend reconstruction.
- Action recognition.
- Before/after causal questions.

### 3.5 Mixed Instruction JSONL

Use JSONL when one example has multiple modalities or a more precise target.

Path:

```text
raw_data_multimodal/mixed_jsonl/train.jsonl
raw_data_multimodal/mixed_jsonl/eval.jsonl
```

Current supported JSONL fields:

```json
{
  "id": "sample_000001",
  "task": "vqa",
  "text": "What is the person holding?",
  "image_path": "../images/train/image_000001.jpg",
  "audio_path": null,
  "video_path": null,
  "target_text": "The person is holding a red umbrella."
}
```

Supported text fields:

```text
text
prompt
input_text
target_text
answer
caption
```

Supported media fields:

```text
image_path
audio_path
video_path
```

## 4. Local Windows Setup

Run from repo root:

```powershell
cd C:\Users\prajw\Downloads\Polymathic-Cognitive-Architecture-PCA-
.\pca\Scripts\python.exe .\scripts\check_torch_cuda.py
.\pca\Scripts\python.exe -m compileall data_pipeline train model_core scripts
```

If dependencies are missing:

```powershell
.\pca\Scripts\python.exe -m pip install -r .\requirements.txt
.\pca\Scripts\python.exe -m pip install pillow
```

## 5. Cloud H100 Setup

On a Linux H100 VM:

```bash
git clone <your-repo-url> pca
cd pca
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pillow
python scripts/check_torch_cuda.py
```

If PyTorch is missing or CPU-only, install the CUDA build matching the cloud image. Example:

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
python scripts/check_torch_cuda.py
```

Copy data to the cloud:

```bash
rsync -av --progress raw_data_corpus/ user@H100_HOST:/workspace/pca/raw_data_corpus/
rsync -av --progress raw_data_multimodal/ user@H100_HOST:/workspace/pca/raw_data_multimodal/
```

Copy results back:

```bash
rsync -av --progress user@H100_HOST:/workspace/pca/train/checkpoints/ ./train/checkpoints/
rsync -av --progress user@H100_HOST:/workspace/pca/train/logs/ ./train/logs/
```

## 6. Train The Tokenizer

Train tokenizer from text data:

```powershell
.\pca\Scripts\python.exe .\tokenizer\train_sentencepiece.py `
  --input-dir .\raw_data_corpus `
  --output-dir .\tokenizer\artifacts `
  --model-prefix pca_tokenizer `
  --vocab-size 32000 `
  --min-english-ratio 0.85 `
  --ascii-only `
  --input-sentence-size 2000000
```

Linux:

```bash
python tokenizer/train_sentencepiece.py \
  --input-dir raw_data_corpus \
  --output-dir tokenizer/artifacts \
  --model-prefix pca_tokenizer \
  --vocab-size 32000 \
  --min-english-ratio 0.85 \
  --ascii-only \
  --input-sentence-size 2000000
```

Validate tokenizer:

```powershell
.\pca\Scripts\python.exe .\tokenizer\validate_tokenizer.py `
  --tokenizer-model .\tokenizer\artifacts\pca_tokenizer.model `
  --sample-text "The model should reason over text, images, audio, and video."
```

If `validate_tokenizer.py` has different CLI flags, run:

```powershell
.\pca\Scripts\python.exe .\tokenizer\validate_tokenizer.py --help
```

## 7. Build Text Dataset

Build packed text shards:

```powershell
.\pca\Scripts\python.exe .\data_pipeline\build_phase1_dataset.py `
  --input-dir .\raw_data_corpus `
  --tokenizer-model .\tokenizer\artifacts\pca_tokenizer.model `
  --output-dir .\data_pipeline\artifacts\phase1_text_all `
  --seq-len 2048 `
  --shard-sequences 1024 `
  --workers 8
```

Create disjoint train/eval split:

```powershell
.\pca\Scripts\python.exe .\data_pipeline\create_manifest_split.py `
  --source-manifest .\data_pipeline\artifacts\phase1_text_all\manifest.json `
  --train-output-dir .\data_pipeline\artifacts\phase1_text_train `
  --eval-output-dir .\data_pipeline\artifacts\phase1_text_eval `
  --eval-ratio 0.05 `
  --seed 42 `
  --train-shard-sequences 1024 `
  --eval-shard-sequences 256
```

## 8. Build Multimodal Manifests

Build a manifest from sidecar files:

```powershell
.\pca\Scripts\python.exe .\data_pipeline\build_multimodal_manifest.py `
  --input-root .\raw_data_multimodal `
  --output .\data_pipeline\artifacts\multimodal_train\manifest.jsonl `
  --include-text-only
```

For eval, keep a separate eval source folder:

```powershell
.\pca\Scripts\python.exe .\data_pipeline\build_multimodal_manifest.py `
  --input-root .\raw_data_multimodal_eval `
  --output .\data_pipeline\artifacts\multimodal_eval\manifest.jsonl `
  --include-text-only
```

If using hand-written JSONL, place files here:

```text
data_pipeline/artifacts/multimodal_train/manifest.jsonl
data_pipeline/artifacts/multimodal_eval/manifest.jsonl
```

Check JSONL validity:

```powershell
@'
import json
from pathlib import Path
for p in [
    Path("data_pipeline/artifacts/multimodal_train/manifest.jsonl"),
    Path("data_pipeline/artifacts/multimodal_eval/manifest.jsonl"),
]:
    if not p.exists():
        print(f"MISSING {p}")
        continue
    n = 0
    for i, line in enumerate(p.open("r", encoding="utf-8"), start=1):
        if not line.strip():
            continue
        json.loads(line)
        n += 1
    print(f"{p}: {n} rows")
'@ | .\pca\Scripts\python.exe -
```

## 9. Required Smoke Tests

Run before every serious training session:

```powershell
.\pca\Scripts\python.exe .\scripts\check_torch_cuda.py
.\pca\Scripts\python.exe -m compileall data_pipeline train model_core scripts
.\pca\Scripts\python.exe .\scripts\smoke_test_task4.py
.\pca\Scripts\python.exe .\scripts\smoke_test_task5.py
.\pca\Scripts\python.exe .\scripts\smoke_test_task7.py
.\pca\Scripts\python.exe .\scripts\smoke_test_task8.py
.\pca\Scripts\python.exe .\scripts\smoke_test_task9.py
.\pca\Scripts\python.exe .\scripts\smoke_test_task10_three_phase_causality.py
.\pca\Scripts\python.exe .\scripts\smoke_test_multimodal_pca.py
```

Run one manifest-backed multimodal training step:

```powershell
.\pca\Scripts\python.exe .\train\train_multimodal_step.py `
  --manifest .\data_pipeline\artifacts\multimodal_train\manifest.jsonl `
  --tokenizer-model .\tokenizer\artifacts\pca_tokenizer.model `
  --steps 1 `
  --batch-size 1 `
  --seq-len 32 `
  --vocab-size 32000 `
  --d-model 64 `
  --n-layers 2 `
  --encoder-layers 1 `
  --decoder-layers 1 `
  --n-heads 4 `
  --n-kv-heads 2 `
  --max-seq-len 64 `
  --ffn-dim 128 `
  --imagination-num-latents 4 `
  --imagination-steps 1 `
  --imagination-heads 4 `
  --imagination-ffn-dim 128 `
  --checkpoint-dir .\tmp\multimodal_train_smoke
```

## 10. Overfit 16 Examples

Create a tiny overfit manifest:

```text
data_pipeline/artifacts/multimodal_overfit16/manifest.jsonl
```

It should contain exactly 16 rows, mixing examples like:

```json
{"id":"overfit_text_001","task":"text","text":"What is 2+2?","target_text":"2+2 equals 4."}
{"id":"overfit_image_001","task":"caption","text":"Describe this image.","image_path":"../../../raw_data_multimodal/images/train/image_000001.jpg","target_text":"A red bicycle is leaning against a brick wall."}
{"id":"overfit_audio_001","task":"asr","text":"Transcribe the audio.","audio_path":"../../../raw_data_multimodal/audio/train/audio_000001.wav","target_text":"Welcome to the project demo."}
{"id":"overfit_video_001","task":"video_caption","text":"Describe the video.","video_path":"../../../raw_data_multimodal/video/train/video_000001.npy","target_text":"A person edits a button color and refreshes the browser."}
```

Run overfit:

```powershell
.\pca\Scripts\python.exe .\train\train_multimodal_step.py `
  --manifest .\data_pipeline\artifacts\multimodal_overfit16\manifest.jsonl `
  --tokenizer-model .\tokenizer\artifacts\pca_tokenizer.model `
  --steps 500 `
  --batch-size 2 `
  --seq-len 128 `
  --vocab-size 32000 `
  --d-model 256 `
  --n-layers 6 `
  --encoder-layers 3 `
  --decoder-layers 3 `
  --n-heads 8 `
  --n-kv-heads 2 `
  --max-seq-len 256 `
  --ffn-dim 1024 `
  --imagination-num-latents 8 `
  --imagination-steps 2 `
  --imagination-heads 8 `
  --imagination-ffn-dim 1024 `
  --image-size 128 `
  --audio-samples 16000 `
  --video-frames 4 `
  --video-frame-size 128 `
  --checkpoint-dir .\train\checkpoints\multimodal_overfit16 `
  2>&1 | Tee-Object -FilePath .\train\logs\multimodal_overfit16.log
```

Success criteria:

```text
loss should drop clearly
grad_norm finite
latent_rms stable
avg_delta_norm not exploding
checkpoint saved
```

If this fails, do not scale.

## 11. Text Baseline Training

Run local text baseline:

```powershell
.\pca\Scripts\python.exe .\scripts\launch_pretrain.py `
  --config .\train\presets\local_rtx3050_4gb_three_phase_v5_split.json `
  --fresh-start `
  2>&1 | Tee-Object -FilePath .\train\logs\text_three_phase_local.log
```

Run H100 text baseline by creating or editing a preset with:

```text
device: cuda
precision: bf16
train_manifest: data_pipeline/artifacts/phase1_text_train/manifest.json
eval_manifest: data_pipeline/artifacts/phase1_text_eval/manifest.json
d_model: 1024-2048 for first cloud run
n_layers: 12-24 for first cloud run
seq_len: 2048 or 4096
gradient_checkpointing: true
```

Then:

```bash
python scripts/launch_pretrain.py --config train/presets/phase1_h100_stable.json --fresh-start \
  2>&1 | tee train/logs/text_h100_baseline.log
```

## 12. Native Multimodal Training Ladder

### 12.1 Tiny Debug Run

```powershell
.\pca\Scripts\python.exe .\train\train_multimodal_step.py `
  --manifest .\data_pipeline\artifacts\multimodal_train\manifest.jsonl `
  --tokenizer-model .\tokenizer\artifacts\pca_tokenizer.model `
  --steps 20 `
  --batch-size 1 `
  --seq-len 64 `
  --vocab-size 32000 `
  --d-model 128 `
  --n-layers 4 `
  --encoder-layers 2 `
  --decoder-layers 2 `
  --n-heads 4 `
  --n-kv-heads 2 `
  --max-seq-len 128 `
  --ffn-dim 512 `
  --imagination-num-latents 4 `
  --imagination-steps 2 `
  --imagination-heads 4 `
  --imagination-ffn-dim 512 `
  --checkpoint-dir .\tmp\multimodal_debug
```

### 12.2 150M-350M H100 Proof

Linux H100 example:

```bash
python train/train_multimodal_step.py \
  --manifest data_pipeline/artifacts/multimodal_train/manifest.jsonl \
  --tokenizer-model tokenizer/artifacts/pca_tokenizer.model \
  --steps 5000 \
  --batch-size 4 \
  --seq-len 512 \
  --vocab-size 32000 \
  --d-model 768 \
  --n-layers 12 \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --n-heads 12 \
  --n-kv-heads 4 \
  --max-seq-len 1024 \
  --ffn-dim 3072 \
  --imagination-num-latents 16 \
  --imagination-steps 3 \
  --imagination-heads 12 \
  --imagination-ffn-dim 3072 \
  --image-size 224 \
  --audio-samples 32000 \
  --video-frames 4 \
  --video-frame-size 224 \
  --precision bf16 \
  --checkpoint-dir train/checkpoints/multimodal_150m \
  2>&1 | tee train/logs/multimodal_150m.log
```

### 12.3 700M-1B H100 Capability Run

Use only after overfit and 150M proof work.

```bash
python train/train_multimodal_step.py \
  --manifest data_pipeline/artifacts/multimodal_train/manifest.jsonl \
  --tokenizer-model tokenizer/artifacts/pca_tokenizer.model \
  --steps 20000 \
  --batch-size 2 \
  --seq-len 1024 \
  --vocab-size 32000 \
  --d-model 1536 \
  --n-layers 24 \
  --encoder-layers 12 \
  --decoder-layers 12 \
  --n-heads 16 \
  --n-kv-heads 4 \
  --max-seq-len 2048 \
  --ffn-dim 4096 \
  --imagination-num-latents 32 \
  --imagination-steps 3 \
  --imagination-heads 16 \
  --imagination-ffn-dim 4096 \
  --image-size 224 \
  --audio-samples 32000 \
  --video-frames 4 \
  --video-frame-size 224 \
  --precision bf16 \
  --checkpoint-dir train/checkpoints/multimodal_700m_1b \
  2>&1 | tee train/logs/multimodal_700m_1b.log
```

### 12.4 1-3B Distributed Run

Do not use the current single-process `train_multimodal_step.py` for a full 3B run. First add FSDP/ZeRO and correct Muon/AdamW parameter grouping.

Target distributed command shape:

```bash
torchrun --nproc_per_node=8 train/train_multimodal_fsdp.py \
  --manifest data_pipeline/artifacts/multimodal_train/manifest.jsonl \
  --eval-manifest data_pipeline/artifacts/multimodal_eval/manifest.jsonl \
  --tokenizer-model tokenizer/artifacts/pca_tokenizer.model \
  --steps 100000 \
  --global-batch-tokens 1048576 \
  --seq-len 2048 \
  --vocab-size 32000 \
  --d-model 2560 \
  --n-layers 36 \
  --encoder-layers 18 \
  --decoder-layers 18 \
  --n-heads 20 \
  --n-kv-heads 4 \
  --ffn-dim 8192 \
  --imagination-num-latents 64 \
  --imagination-steps 4 \
  --precision bf16 \
  --optimizer muon_adamw_split \
  --checkpoint-dir train/checkpoints/multimodal_2b_3b \
  2>&1 | tee train/logs/multimodal_2b_3b.log
```

This command is a target after adding `train_multimodal_fsdp.py`.

## 13. Codec Plan For Real Multimodal Output

The current model can accept multimodal inputs and produce text. For real image/audio/video output, add codecs.

Recommended storage:

```text
data_pipeline/artifacts/codecs/
  image_tokens/
  audio_tokens/
  video_tokens/
```

Codec targets:

```text
image: VQGAN/VQ-VAE tokens or diffusion bridge
audio: EnCodec/SoundStream-style tokens
video: frame image tokens plus temporal compression
```

Training should become:

```text
input media/text -> shared latent imagination -> predict text tokens and/or codec tokens
```

Do not train raw pixel/audio/video generation first. Predict codec tokens first.

## 14. Evaluation Procedure

Create eval files:

```text
data_pipeline/artifacts/evals/text_prompts.jsonl
data_pipeline/artifacts/evals/image_vqa.jsonl
data_pipeline/artifacts/evals/audio_asr.jsonl
data_pipeline/artifacts/evals/video_qa.jsonl
data_pipeline/artifacts/evals/imagination_counterfactual.jsonl
```

Track every run:

```text
train loss
eval loss
generalization gap
tokens/sec
GPU memory
latent_rms
avg_delta_norm
convergence_ratio
sample outputs
failure examples
```

Minimum pass condition before scaling:

```text
16-example overfit passes
train/eval split has no leakage
eval loss decreases
samples improve
latent stats stable
no NaN gradients
checkpoints resume correctly
```

## 15. What To Send Back After Each Run

Send these files or pasted summaries:

```text
train/logs/<run_name>.log
train/checkpoints/<run_name>/last.pt exists or not
final JSON training payload
5 sample prompts
5 model outputs
GPU type and count
peak GPU memory
dataset row counts
```

Extract final metrics:

```powershell
@'
import json
from pathlib import Path
for p in Path("train/logs").glob("*.log"):
    last = None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if "step" in obj or obj.get("status") == "done":
            last = obj
    print(f"\n=== {p} ===")
    print(json.dumps(last, indent=2) if last else "NO_JSON_PAYLOAD")
'@ | .\pca\Scripts\python.exe -
```

## 16. Definition Of Done

First complete version:

- Accepts text, image, audio, and video inputs.
- Produces reliable text output.
- Trains on disjoint multimodal train/eval manifests.
- Can overfit 16 multimodal examples.
- Shows decreasing eval loss on held-out multimodal data.
- Has stable latent imagination telemetry.
- Can save, resume, and sample from checkpoints.
- Has baseline comparisons against text-only PCA and three-phase PCA.

Later complete version:

- Predicts image/audio/video codec tokens.
- Decodes generated media through pretrained codecs.
- Runs SFT.
- Runs preference training.
- Runs multimodal RL.
- Scales to 1-3B with distributed Muon/AdamW split.
