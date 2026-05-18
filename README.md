# RadAssist AI

AI-powered lung segmentation from chest X-rays using a U-Net convolutional neural network. Production-deployed on Kubernetes with HTTPS, custom domain, and automated CI/CD.

![Status](https://img.shields.io/badge/status-live-success)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.9-orange)
![k8s](https://img.shields.io/badge/kubernetes-k3s-326CE5)

🌐 **Live Demo**: [radassist.durak.dev](https://radassist.durak.dev)

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

- **Encoder**: 5 downsampling blocks (16→32→64→128→256 channels)
- **Decoder**: 5 upsampling blocks that rebuild the image
- **Skip connections**: copy spatial detail from encoder to decoder
- **Output**: probability map, threshold > 0.5 → binary mask

**Parameters**: 1,624,844 (lightweight — runs on CPU in <1s).

### Training

- **Framework**: PyTorch + [MONAI](https://monai.io/)
- **Hardware**: Apple Silicon GPU (MPS)
- **Loss**: Dice Loss (handles class imbalance better than cross-entropy)
- **Optimizer**: Adam, lr 1e-3
- **Epochs**: 10 · **Batch size**: 8 · **Split**: 80/20

### Results

| Metric | Value |
|---|---|
| Train loss | 0.077 |
| Val loss | 0.073 |
| Dice score | ~0.93 |
| Inference | <1s on CPU |

## System Architecture

```
                       ┌─────────────────────┐
                       │ radassist.durak.dev │  ← Custom domain + HTTPS
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │  Traefik Ingress    │  ← cert-manager + Let's Encrypt
                       └──────────┬──────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
   ┌────────▼────────┐                         ┌────────▼────────┐
   │  Frontend Pod   │ ──── /predict ────▶    │  Backend Pod    │
   │ (nginx + React) │   (with API key)        │ (FastAPI + ML) │
   └─────────────────┘                         └─────────────────┘
                                                        │
                                                ┌───────▼────────┐
                                                │  U-Net Model   │
                                                │ (1.6M params)  │
                                                └────────────────┘
```

## Stack

**ML** — PyTorch · MONAI · U-Net
**Backend** — FastAPI · API key auth
**Frontend** — React · Vite · drag-drop · overlay · stats
**Infrastructure** — Docker · k3s · Traefik · cert-manager · Let's Encrypt · DigitalOcean
**DevOps** — GHCR · GitHub Actions CI/CD · rolling updates

## CI/CD

Every push to `main` triggers:
1. Build & push API image to GHCR
2. Build & push Web image to GHCR
3. SSH to droplet, update k8s deployments
4. Zero-downtime rolling update

## Quick Start

### Backend
```bash
pip install -r requirements.txt
export API_KEY=your-secret-key
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
docker run -p 8000:8000 -e API_KEY=your-secret-key radassist-ai:v1
```

## API Usage

```bash
curl -X POST https://radassist.durak.dev/predict \
  -H "x-api-key: YOUR_KEY" \
  -F "file=@chest_xray.png"
```

Returns:
```json
{
  "mask": "data:image/png;base64,...",
  "stats": {
    "lung_area_pct": 31.08,
    "confidence": 99.37,
    "resolution": "256x256"
  }
}
```

## Project Structure

```
radassist-ai/
├── src/                  # Backend (FastAPI + model)
├── web/                  # React frontend + nginx
├── k8s/                  # Kubernetes manifests
├── .github/workflows/    # CI/CD pipeline
├── models/               # Trained weights
├── Dockerfile
└── requirements.txt
```

## Dataset

[Montgomery + Shenzhen Hospital X-rays](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels) — 704 paired X-rays and masks.

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
- [MONAI Framework](https://monai.io/)

## License

MIT
