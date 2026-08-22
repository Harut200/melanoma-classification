import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class MelanomaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row['image_name']
        img_path = os.path.join(self.img_dir, f"{image_name}.jpg")

        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        target = torch.tensor(row['target'], dtype=torch.float32) if 'target' in row else torch.tensor(0.0)

        return image, target