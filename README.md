# SAMA: Factorized Semantic Anchoring and Motion Alignment for Instruction-Guided Video Editing

<div align="center">
  <a href="https://arxiv.org/abs/2603.19228" target="_blank"><img src="https://img.shields.io/badge/Paper-b31b1b.svg?logo=arxiv&logoColor=white" height="22px"></a>
  <a href="https://cynthiazxy123.github.io/SAMA/" target="_blank"><img src="https://img.shields.io/badge/Webpage-4f46e5.svg?logo=googlechrome&logoColor=white" height="22px"></a>
  <a href="https://huggingface.co/syxbb/SAMA-14B" target="_blank"><img src="https://img.shields.io/badge/Model-f59e0b.svg?logo=huggingface&logoColor=white" height="22px"></a>
  <a href="https://huggingface.co/datasets/syxbb/SAMA-edit-filtered-1M" target="_blank"><img src="https://img.shields.io/badge/Dataset-2563eb.svg?logo=huggingface&logoColor=white" height="22px"></a>
  <a href="https://github.com/Cynthiazxy123/SAMA" target="_blank"><img src="https://img.shields.io/badge/Code-111111.svg?logo=github&logoColor=white" height="22px"></a>
</div>

<div align="center">
  <a href="https://scholar.google.com/citations?hl=zh-TW&user=3nRkR1wAAAAJ">Xinyao Zhang</a><sup>1,2,*</sup>,
  <a href="https://openreview.net/profile?id=~Wenkai_Dong1">Wenkai Dong</a><sup>1,*</sup>,
  <a href="https://scholar.google.com/citations?hl=zh-TW&user=1uL_9HAAAAAJ">Yuxin Song</a><sup>1,*,&dagger;</sup>,
  <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=sphnU4UAAAAJ">Bo Fang</a><sup>1,3</sup>,
  <a href="https://openreview.net/profile?id=~Qi_Zhang40">Qi Zhang</a><sup>1</sup>,
  <a href="https://openreview.net/profile?id=~Jing_Wang68">Jing Wang</a><sup>1,2</sup>,
  <a href="https://openreview.net/profile?id=~Fan_Chen14">Fan Chen</a><sup>1</sup>,
  <a href="https://openreview.net/profile?id=~Hui_Zhang28">Hui Zhang</a><sup>1</sup>,
  <a href="https://scholar.google.com.hk/citations?user=pnuQ5UsAAAAJ&hl=zh-CN&oi=ao">Haocheng Feng</a><sup>1</sup>,
  <a href="https://yulu.net.cn/">Yu Lu</a><sup>4,&Dagger;</sup>,
  <a href="https://hangz-nju-cuhk.github.io/">Hang Zhou</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=fYdxi2sAAAAJ&hl=zh-TW">Chun Yuan</a><sup>2</sup>,
  <a href="https://jingdongwang2017.github.io/">Jingdong Wang</a><sup>1</sup>
</div>

<div align="center">
  <sup>1</sup> Baidu Inc &nbsp;&nbsp;
  <sup>2</sup> Tsinghua University &nbsp;&nbsp;
  <sup>3</sup> City University of Hong Kong &nbsp;&nbsp;
  <sup>4</sup> Zhejiang University
</div>

<div align="center">
  * Equal Contribution &nbsp;&nbsp; &dagger; Project leader &nbsp;&nbsp; &Dagger; Corresponding Author
</div>

Official inference code for **SAMA: Factorized Semantic Anchoring and Motion Alignment for Instruction-Guided Video Editing**.

SAMA factorizes instruction-guided video editing into semantic anchoring and motion alignment, improving edit precision while preserving temporal dynamics from the source video.

## 🧾 Abstract

Current instruction-guided video editing models struggle to simultaneously balance precise semantic modifications with faithful motion preservation. While existing approaches rely on injecting explicit external priors (e.g., VLM features or structural conditions) to mitigate these issues, this reliance severely bottlenecks model robustness and generalization. To overcome this limitation, we present **SAMA** (factorized **S**emantic **A**nchoring and **M**otion **A**lignment), a framework that factorizes video editing into semantic anchoring and motion modeling. First, we introduce **Semantic Anchoring**, which establishes a reliable visual anchor by jointly predicting semantic tokens and video latents at sparse anchor frames, enabling purely instruction-aware structural planning. Second, **Motion Alignment** pre-trains the same backbone on motion-centric video restoration pretext tasks (cube inpainting, speed perturbation, and tube shuffle), enabling the model to internalize temporal dynamics directly from raw videos. SAMA is optimized with a two-stage pipeline: a factorized pre-training stage that learns inherent semantic-motion representations without paired video-instruction editing data, followed by supervised fine-tuning on paired editing data. Remarkably, the factorized pre-training alone already yields strong zero-shot video editing ability, validating the proposed factorization. SAMA achieves state-of-the-art performance among open-source models and is competitive with leading commercial systems (e.g. Kling-Omni). Code, models, and datasets will be released.

## 🖼️ Overview

![SAMA teaser overview](./assets/images/teaser-overview.png)

## 📰 News

- 🔥 2026.03.20 Release paper.

## 📊 Benchmark Highlight

![VIE-Bench results](./assets/images/table2-vie-bench.png)

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

## 🤗 Available Models

| Model | Status | Link |
| --- | --- | --- |
| SAMA-5B | Coming soon | Coming soon |
| SAMA-14B | Available | [syxbb/SAMA-14B](https://huggingface.co/syxbb/SAMA-14B) |

## 🙏 Acknowledgement

- **Wan**: We build SAMA on top of the Wan video generation backbone and follow its model ecosystem for video synthesis and editing.
- **DiffSynth**: We use DiffSynth as the underlying implementation framework for model components, inference utilities, and training-related infrastructure.

## 📚 Citation

```bibtex
@misc{zhang2026samafactorizedsemanticanchoring,
      title={SAMA: Factorized Semantic Anchoring and Motion Alignment for Instruction-Guided Video Editing},
      author={Xinyao Zhang and Wenkai Dong and Yuxin Song and Bo Fang and Qi Zhang and Jing Wang and Fan Chen and Hui Zhang and Haocheng Feng and Yu Lu and Hang Zhou and Chun Yuan and Jingdong Wang},
      year={2026},
      eprint={2603.19228},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.19228},
}
```
