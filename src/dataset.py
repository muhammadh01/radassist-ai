import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LungDataset(Dataset):
    def __init__(self, base_dir, size=256):
        self.base = base_dir
        self.masks = sorted(os.listdir(f"{base_dir}/masks"))
        self.size = size
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask_name = self.masks[idx]
        img_name = mask_name.replace("_mask", "")
        img = Image.open(f"{self.base}/CXR_png/{img_name}").convert("L")
        mask = Image.open(f"{self.base}/masks/{mask_name}").convert("L")
        img = self.tf(img)
        mask = (self.tf(mask) > 0.5).float()
        return img, mask

if __name__ == "__main__":
    ds = LungDataset("data/Lung Segmentation")
    print(f"Total samples: {len(ds)}")
    img, mask = ds[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
