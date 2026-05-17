from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import torch
from PIL import Image
from torchvision import transforms
import io
import numpy as np
from src.model import get_model

app = FastAPI(title="RadAssist AI")

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
        pred = model(tensor)
        pred = torch.sigmoid(pred).cpu().squeeze().numpy()
        pred = (pred > 0.5).astype(np.uint8) * 255

    mask_img = Image.fromarray(pred)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
