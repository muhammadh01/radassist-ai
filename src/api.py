from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
from torchvision import transforms
import io, base64
import numpy as np
from src.model import get_model

app = FastAPI(title="RadAssist AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = get_model().to(device)
model.load_state_dict(torch.load("models/unet.pth", map_location=device))
model.eval()

tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.get("/")
def root():
    return {"status": "ok", "service": "radassist-ai"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out).cpu().squeeze().numpy()
        mask = (prob > 0.5).astype(np.uint8)

    lung_pixels = int(mask.sum())
    total_pixels = mask.size
    area_pct = round(lung_pixels / total_pixels * 100, 2)
    confidence = round(float(prob[mask == 1].mean()) * 100, 2) if lung_pixels > 0 else 0

    mask_img = Image.fromarray(mask * 255)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "mask": f"data:image/png;base64,{mask_b64}",
        "stats": {
            "lung_area_pct": area_pct,
            "confidence": confidence,
            "resolution": "256x256",
        }
    }
