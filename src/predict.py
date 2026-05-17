import torch
import matplotlib.pyplot as plt
from dataset import LungDataset
from model import get_model

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = get_model().to(device)
model.load_state_dict(torch.load("models/unet.pth", map_location=device))
model.eval()

ds = LungDataset("data/Lung Segmentation")
img, mask = ds[5]

with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device))
    pred = torch.sigmoid(pred).cpu().squeeze().numpy()
    pred = (pred > 0.5).astype(float)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img.squeeze(), cmap="gray"); ax[0].set_title("X-ray")
ax[1].imshow(mask.squeeze(), cmap="gray"); ax[1].set_title("Ground Truth")
ax[2].imshow(pred, cmap="gray"); ax[2].set_title("Model Prediction")
plt.show()
