import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
from PIL import Image

class SpaceNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, limit=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".tif")
        ])

        if limit:
            self.images = self.images[:limit]

        
        self.empty_mask_flags = []

        for name in self.images:
            mask_name = name.replace(".tif", ".png")
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask: {mask_path}")

            mask = np.array(Image.open(mask_path).convert("L"))
            self.empty_mask_flags.append(mask.sum() == 0)

        print(f"[INFO] Total images: {len(self.images)}")
        print(f"[INFO] Empty masks: {sum(self.empty_mask_flags)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".tif", ".png")

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        with rasterio.open(img_path) as img:
            image = img.read([1, 2, 3]).astype(np.float32)
        image = image / (image.max() + 1e-6)

        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(self.empty_mask_flags[idx], dtype=torch.bool)
        )
dataset = SpaceNetDataset(
    image_dir="/content/drive/MyDrive/SpaceNet2/train/AOI_4_Shanghai/images",
    mask_dir="/content/drive/MyDrive/SpaceNet2/train/AOI_4_Shanghai/masks",
    limit=5
)

img, mask, empty = dataset[0]
print(img.shape, mask.shape, empty)

