### define augmentations of photos
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),])

### define label mapping
id2label = {
    0: "Gravel",
    1: "MC_Sand",
    2: "FV_Sand",
    3: "Quadrat_or_Paper",
    4: "Mud",
    5: "Barnacles",
    6: "Algae",
    7: "Shell"}

label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

### Build a PyTorch dataset
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class BeachSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".jpg", ".png").replace(".tif", ".png")

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))  # must be single-channel class IDs

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # convert to tensor
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask).long()
        
        return image, mask

