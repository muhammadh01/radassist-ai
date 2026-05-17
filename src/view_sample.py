import matplotlib.pyplot as plt
from PIL import Image
import os

base = "data/Lung Segmentation"
img_name = os.listdir(f"{base}/CXR_png")[0]

img = Image.open(f"{base}/CXR_png/{img_name}")
plt.imshow(img, cmap="gray")
plt.title(img_name)
plt.show()
