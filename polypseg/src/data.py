"""Datasets and augmentation for polyp segmentation.


Handles the in-domain Kvasir-SEG splits (from splits.csv) and out-of-domain
test sets used for the cross-dataset generalization study.
"""
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size: int):
    """Augmentations reflecting real colonoscopy variability: viewpoint,
    scale, illumination and mild optical degradation."""
    return A.Compose([
        A.Resize(size, size),
        A.RandomResizedCrop(size=(size, size), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-90, 90), scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transform(size: int):
    """Deterministic resize used for val/test. Plain (non aspect-preserving)
    resize matches the polyp-segmentation literature and inverts trivially,
    so predictions can be mapped back to native resolution for scoring."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class PolypDataset(Dataset):
    """Generic image/mask folder dataset.

    `records` is a list of (name, image_path, mask_path) so that per-image
    metrics can be joined back to the stratum metadata by name.
    """

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        name, img_path, mask_path = self.records[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        out = self.transform(image=image, mask=mask)
        return {
            "image": out["image"],
            "mask": out["mask"].unsqueeze(0).float(),
            "name": name,
        }


def kvasir_records(root: Path, splits_csv: Path, split: str):
    df = pd.read_csv(splits_csv)
    df = df[df.split == split]
    return [
        (r["name"], root / "images" / f"{r['name']}.jpg", root / "masks" / f"{r['name']}.jpg")
        for _, r in df.iterrows()
    ]


def folder_records(image_dir: Path, mask_dir: Path):
    """Records for an external dataset laid out as images/ + masks/ with
    matching stems (extension may differ between the two folders)."""
    masks = {p.stem: p for p in sorted(mask_dir.iterdir()) if p.suffix.lower() in {".jpg", ".png", ".tif", ".tiff", ".bmp"}}
    records = []
    for img in sorted(image_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".png", ".tif", ".tiff", ".bmp"}:
            continue
        if img.stem in masks:
            records.append((img.stem, img, masks[img.stem]))
    return records


def seed_worker(worker_id):
    """Seed every RNG a DataLoader worker may use.

    Seeding only NumPy is a common and consequential omission: Albumentations
    samples transform parameters from Python's `random` module, so leaving it
    unseeded makes the augmentation stream differ between otherwise identical
    runs. That is a source of run-to-run variance which CUDA determinism
    settings (torch.use_deterministic_algorithms, cudnn.deterministic) cannot
    address, because it originates in the data pipeline rather than in kernels.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
