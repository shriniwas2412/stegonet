"""Dataset of (cover, secret) pairs drawn from a folder of images."""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder) -> list:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


class PairDataset(Dataset):
    """Each item is a random (cover, secret) pair of distinct images.

    In training mode pairs are re-drawn every epoch and images are randomly
    cropped and flipped. In eval mode pairs are fixed (image i hides image i+n/2)
    so evaluation numbers are reproducible.
    """

    def __init__(self, folder, size: int = 128, train: bool = True, limit: int | None = None):
        self.files = list_images(folder)
        if limit:
            self.files = self.files[:limit]
        if len(self.files) < 2:
            raise ValueError(f"Need at least two images in {folder}")
        self.train = train
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size, antialias=True),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.files) // 2 if not self.train else len(self.files)

    def _load(self, path) -> torch.Tensor:
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        if self.train:
            j = random.randrange(len(self.files) - 1)
            j = j + 1 if j >= idx else j
            return self._load(self.files[idx]), self._load(self.files[j])
        half = len(self.files) // 2
        return self._load(self.files[idx]), self._load(self.files[idx + half])
