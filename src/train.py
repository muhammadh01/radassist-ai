import torch
from torch.utils.data import DataLoader, random_split
from monai.losses import DiceLoss
from dataset import LungDataset
from model import get_model

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

ds = LungDataset("data/Lung Segmentation")
train_size = int(0.8 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = random_split(ds, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=8)

model = get_model().to(device)
loss_fn = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for img, mask in train_dl:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = loss_fn(out, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, mask in val_dl:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            val_loss += loss_fn(out, mask).item()

    print(f"Epoch {epoch+1}/{EPOCHS} | train: {train_loss/len(train_dl):.4f} | val: {val_loss/len(val_dl):.4f}")

torch.save(model.state_dict(), "models/unet.pth")
print("✅ Saved model to models/unet.pth")
