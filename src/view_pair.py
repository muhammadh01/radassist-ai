import matplotlib.pyplot as plt
from PIL import Image
import os

base = "data/Lung Segmentation"
masks = os.listdir(f"{base}/masks")
mask_name = masks[0]
img_name = mask_name.replace("_mask", "")

img = Image.open(f"{base}/CXR_png/{img_name}")
mask = Image.open(f"{base}/masks/{mask_name}")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("X-ray")
ax[1].imshow(mask, cmap="gray"); ax[1].set_title("Mask")
plt.show()
