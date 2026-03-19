# SAMA

<div align="center">
  <a href="https://arxiv.org/" target="_blank"><img src="https://img.shields.io/badge/Paper-b31b1b.svg?logo=arxiv&logoColor=white" height="22px"></a>
  <a href="https://cynthiazxy123.github.io/SAMA/" target="_blank"><img src="https://img.shields.io/badge/Webpage-4f46e5.svg?logo=googlechrome&logoColor=white" height="22px"></a>
  <a href="https://huggingface.co/syxbb/SAMA-14B" target="_blank"><img src="https://img.shields.io/badge/Model-f59e0b.svg?logo=huggingface&logoColor=white" height="22px"></a>
  <a href="https://huggingface.co/datasets/syxbb/SAMA-edit-filtered-1M" target="_blank"><img src="https://img.shields.io/badge/Dataset-2563eb.svg?logo=huggingface&logoColor=white" height="22px"></a>
  <a href="https://github.com/Cynthiazxy123/SAMA" target="_blank"><img src="https://img.shields.io/badge/Code-111111.svg?logo=github&logoColor=white" height="22px"></a>
</div>

Official inference code for **SAMA: Factorized Semantic Anchoring and Motion Alignment for Instruction-Guided Video Editing**.

SAMA factorizes instruction-guided video editing into semantic anchoring and motion alignment, improving edit precision while preserving temporal dynamics from the source video.

---

## 🚀 Quick Start

### 🛠️ Installation

Recommended environment:

- Linux
- NVIDIA GPU
- CUDA 12.1 or a compatible environment
- Python 3.10

```bash
git clone https://github.com/Cynthiazxy123/SAMA
cd SAMA

conda create -n sama python=3.10 -y
conda activate sama

pip install --upgrade pip
pip install -r requirements.txt
```

### ▶️ Inference

Prepare:

1. The base `Wan2.1-T2V-14B` model directory.
2. A SAMA checkpoint from [Hugging Face](https://huggingface.co/syxbb/SAMA-14B).
3. A source video and an edit instruction.

The inference script is:

`infer_sh/run_sama.sh`

Edit the variables at the top of that script before running:

- `MODEL_ROOT`
- `STATE_DICT`
- `SRC_VIDEO`
- `PROMPT`
- `OUTPUT_DIR`

Then run:

```bash
bash infer_sh/run_sama.sh
```

The generated result will be saved to:

```text
outputs/seed_1/<input_video_filename>
```

A recommended local model layout is:

```text
models/
├── Wan2.1-T2V-14B/
│   ├── diffusion_pytorch_model-00001-of-00006.safetensors
│   ├── diffusion_pytorch_model-00002-of-00006.safetensors
│   ├── diffusion_pytorch_model-00003-of-00006.safetensors
│   ├── diffusion_pytorch_model-00004-of-00006.safetensors
│   ├── diffusion_pytorch_model-00005-of-00006.safetensors
│   ├── diffusion_pytorch_model-00006-of-00006.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.1_VAE.pth
│   └── google/
└── SAMA-14B/
    └── <downloaded_checkpoint>.safetensors
```

If you have `huggingface_hub` installed, you can download the released checkpoint with:

```bash
huggingface-cli download syxbb/SAMA-14B --local-dir ./models/SAMA-14B
```

### 📝 Notes

- Input frames are automatically padded to satisfy the `4k+1` frame requirement used by Wan video inference.
- The output video uses the source video FPS when available; otherwise it falls back to `--fps`.
- If `--model-root` is incomplete, the script will stop and report the missing files or directories.

---

## 🤗 Available Models

| Model | Status | Link |
| --- | --- | --- |
| SAMA-5B | Coming soon | Coming soon |
| SAMA-14B | Available | [syxbb/SAMA-14B](https://huggingface.co/syxbb/SAMA-14B) |
