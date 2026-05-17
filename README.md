# RadAssist AI

AI-powered lung segmentation from chest X-rays using a U-Net convolutional neural network.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.9-orange)

## Overview

RadAssist AI assists radiologists by automatically highlighting lung regions in chest X-rays. Upload a scan and the model returns a pixel-accurate segmentation mask, lung area percentage, and confidence score in under a second.

**Educational/research project. Not for clinical use.**

## Demo

| Original X-ray | Segmentation Overlay |
|---|---|
| Input chest X-ray | Predicted lung region overlay |

## Architecture

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

## Performance

- **Validation Dice Loss**: 0.073 (~93% accuracy)
- **Inference**: <1s per X-ray on CPU
- **Dataset**: 704 chest X-rays (Montgomery + Shenzhen)

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

- [Montgomery County X-ray Set](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
- [Shenzhen Hospital X-ray Set](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)

## License

MIT
