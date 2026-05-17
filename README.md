# RadAssist AI

AI-powered lung segmentation from chest X-rays using a U-Net convolutional neural network.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.9-orange)

## Overview

RadAssist AI assists radiologists by automatically highlighting lung regions in chest X-rays. Upload a scan and the model returns a pixel-accurate segmentation mask, lung area percentage, and confidence score in under a second.

**Educational/research project. Not for clinical use.**

## How It Works

### Task: Semantic Segmentation

This isn't classification ("is there a tumor: yes/no") or detection ("tumor is somewhere in this box"). It's **segmentation**: classifying every single pixel as *lung* or *not-lung*. The output is a binary mask the exact size of the input image.

### Model: U-Net (a CNN)

The model is a **U-Net**, a **Convolutional Neural Network (CNN)** designed for biomedical image segmentation (Ronneberger et al., 2015).

**Why a CNN and not other architectures?**

| Architecture | Good for | Why not here |
|---|---|---|
| **CNN (U-Net)** ✅ | Spatial data: images, scans | Best fit — captures local pixel patterns + spatial hierarchy |
| RNN / LSTM | Sequential data: text, time series | Images aren't sequences |
| Transformer (ViT) | Large-scale image tasks | Needs huge datasets (millions of images); overkill for 704 X-rays |
| Plain MLP | Tabular data | Loses spatial information when flattening images |

CNNs are the right tool because lung shape is a **spatial pattern**: edges, curves, contrast between air-filled lungs and ribcage. Convolutional filters slide across the image detecting these patterns at multiple scales.

### Architecture Details

U-Net has two halves shaped like a "U":

```
Input (256×256)
      │
   ┌──▼──┐  Encoder (downsampling)
   │ 16  │  ─────────┐  ← captures "what" is in the image
   └──┬──┘           │
   ┌──▼──┐           │
   │ 32  │  ───────┐ │
   └──┬──┘         │ │
   ┌──▼──┐         │ │   skip connections
   │ 64  │  ─────┐ │ │   preserve spatial detail
   └──┬──┘       │ │ │
   ┌──▼──┐       │ │ │
   │ 128 │  ───┐ │ │ │
   └──┬──┘     │ │ │ │
   ┌──▼──┐     │ │ │ │
   │ 256 │ ← bottleneck (most abstract features)
   └──┬──┘     │ │ │ │
   ┌──▼──┐ ◀───┘ │ │ │
   │ 128 │       │ │ │   Decoder (upsampling)
   └──┬──┘       │ │ │   ← reconstructs "where" things are
   ┌──▼──┐ ◀─────┘ │ │
   │ 64  │         │ │
   └──┬──┘         │ │
   ┌──▼──┐ ◀───────┘ │
   │ 32  │           │
   └──┬──┘           │
   ┌──▼──┐ ◀─────────┘
   │ 16  │
   └──┬──┘
      ▼
  Mask (256×256)
```

- **Encoder**: 5 downsampling blocks. Each halves the image size and doubles the feature channels (16→32→64→128→256).
- **Decoder**: 5 upsampling blocks that rebuild the image at full resolution.
- **Skip connections**: copy detailed spatial info from encoder to decoder. Without them, fine edges get lost during downsampling.
- **Output**: single-channel probability map. Threshold > 0.5 → final binary mask.

**Parameters**: 1,624,844 (lightweight — runs on CPU in <1s).

### Training

- **Framework**: PyTorch + [MONAI](https://monai.io/) (medical imaging toolkit)
- **Hardware**: Apple Silicon GPU (Metal Performance Shaders / MPS)
- **Loss function**: **Dice Loss** — directly optimizes overlap between predicted and ground-truth mask. Better than cross-entropy for segmentation because it handles class imbalance (most pixels are background).
- **Optimizer**: Adam, learning rate 1e-3
- **Epochs**: 10
- **Batch size**: 8
- **Train/val split**: 80/20
- **Preprocessing**: resize to 256×256, normalize to [0,1]

### Results

| Metric | Value |
|---|---|
| Final train loss | 0.077 |
| Final val loss | 0.073 |
| Approximate Dice score | ~0.93 (93% overlap with ground truth) |
| Inference time | <1s on CPU |

Train and val loss decrease together with a small gap → no overfitting.

## System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  React UI   │ ───▶ │ FastAPI      │ ───▶ │  U-Net      │
│  (Vite)     │ ◀─── │ (Python)     │ ◀─── │  (PyTorch)  │
└─────────────┘      └──────────────┘      └─────────────┘
```

## Stack

- **Model**: U-Net (MONAI), 1.6M parameters
- **Backend**: FastAPI, PyTorch
- **Frontend**: React + Vite
- **Training**: Apple Silicon GPU (MPS)
- **Container**: Docker
- **Deployment**: Kubernetes (k3s) on DigitalOcean

## Quick Start

### Backend
```bash
pip install -r requirements.txt
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd web
npm install
npm run dev
```

### Docker
```bash
docker build -t radassist-ai:v1 .
docker run -p 8000:8000 radassist-ai:v1
```

## Project Structure

```
radassist-ai/
├── src/              # Backend (FastAPI + model)
│   ├── api.py        # REST endpoints
│   ├── model.py      # U-Net definition
│   ├── train.py      # Training script
│   └── dataset.py    # Data loader
├── web/              # React frontend
├── models/           # Trained weights
├── Dockerfile        # Container build
└── requirements.txt
```

## Dataset

- [Montgomery County + Shenzhen Hospital X-ray Sets](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels) — 704 paired X-rays and lung masks.

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
- [MONAI Framework](https://monai.io/) — PyTorch-based medical imaging library.

## License

MIT
