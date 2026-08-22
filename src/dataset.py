import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# timm's pretrained weights were trained on images normalised with these
# numbers. If we skip this and only divide by 255, the pretrained weights are
# being fed a distribution they have never seen and most of the benefit of
# using them is thrown away.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MelanomaDataset(Dataset):
    """
    Reads the preprocessed jpg photos and hands back (image_tensor, target).

    `transform` is for AUGMENTATION ONLY (flips, rotations, colour jitter).
    Do not put Normalize or ToTensorV2 in it. Normalising and converting to a
    tensor happens here, every time, so it can never be forgotten in one place
    and applied twice in another.
    """

    def __init__(self, df, img_dir, image_size=224, transform=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.image_size = image_size
        self.transform = transform
        self.has_target = has_target and 'target' in self.df.columns

        if not os.path.isdir(img_dir):
            raise NotADirectoryError(
                f"Image directory does not exist: {img_dir}\n"
                "The preprocessed photos live in data/processed/train_512."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row['image_name']
        img_path = os.path.join(self.img_dir, f"{image_name}.jpg")

        # A missing photo is a bug in the setup, not something to paper over.
        # The previous version returned a black square here, which meant a wrong
        # image directory trained the model on 58,000 identical blank images and
        # reported metrics as if everything were fine.
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Check that img_dir points at the resized photos."
            )

        image = cv2.imread(img_path)
        if image is None:
            raise OSError(f"Could not decode image (file may be corrupt): {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Every image must be the same size or the batch cannot be stacked.
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            interpolation = (cv2.INTER_AREA if image.shape[0] > self.image_size
                             else cv2.INTER_LINEAR)
            image = cv2.resize(image, (self.image_size, self.image_size),
                               interpolation=interpolation)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.has_target:
            target = torch.tensor(float(row['target']), dtype=torch.float32)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)

        return image, target


def check_images_exist(df, img_dir, sample_size=200):
    """
    Look for missing photos BEFORE training starts, instead of discovering it
    an epoch in. Checks a random sample, then reports how many are missing.
    """
    if not os.path.isdir(img_dir):
        raise NotADirectoryError(f"Image directory does not exist: {img_dir}")

    sample = df if len(df) <= sample_size else df.sample(sample_size, random_state=0)

    missing = []
    for image_name in sample['image_name']:
        if not os.path.exists(os.path.join(img_dir, f"{image_name}.jpg")):
            missing.append(image_name)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(sample)} sampled photos are missing from {img_dir}\n"
            f"First few: {missing[:5]}\n"
            "Wrong img_dir, or step 3 of the preprocessing has not been run."
        )

    print(f"  image check: {len(sample)} sampled photos all present in {img_dir}")
