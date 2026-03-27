"""
tools/augmentation.py — Image augmentation pipelines for vision research.

These are building blocks the agent can compose in train.py.
All functions accept and return PIL Images or torchvision transforms.

Utilities:
  get_train_transforms()  — Standard training augmentation pipeline (configurable)
  get_val_transforms()    — Deterministic validation pipeline
  RandAugment             — Random augmentation policy (can be added to train pipeline)
  MixupCutmix             — Batch-level augmentation (apply after dataloader)

Usage in train.py:
    from tools.augmentation import get_train_transforms, get_val_transforms
    train_loader = make_dataloader(TASK_NAME, BATCH_SIZE, "train",
                                   train_transform=get_train_transforms(img_size=224, strong=True))
"""

import random
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Standard pipelines
# ---------------------------------------------------------------------------


def get_train_transforms(
    img_size: int = 224,
    strong: bool = False,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Training augmentation pipeline.

    Args:
        img_size: output image size (square)
        strong:   if True, add stronger augmentations (RandAugment, random erasing)
        mean/std: normalization params (ImageNet defaults work well even from scratch)
    """
    base = [
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        T.RandomRotation(degrees=10),
    ]

    if strong:
        base += [
            T.RandAugment(num_ops=2, magnitude=9),
            T.RandomGrayscale(p=0.05),
        ]

    base += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    if strong:
        base.append(T.RandomErasing(p=0.2, scale=(0.02, 0.1)))

    return T.Compose(base)


def get_val_transforms(
    img_size: int = 224,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Deterministic validation pipeline."""
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


# ---------------------------------------------------------------------------
# Mixup / CutMix (batch-level)
# ---------------------------------------------------------------------------


class MixupCutmix(nn.Module):
    """
    Apply Mixup and/or CutMix to a batch of images and one-hot labels.

    Usage in train.py:
        augmenter = MixupCutmix(num_classes=NUM_CLASSES, mixup_alpha=0.2, cutmix_alpha=1.0)
        images, labels = augmenter(images, labels)
        # labels are now soft/mixed

    Note: use `F.cross_entropy(logits, labels)` — it supports soft labels in PyTorch 2.x.
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W) float
            labels: (B,) long — will be converted to one-hot soft labels

        Returns:
            images: (B, C, H, W) float, augmented
            labels: (B, num_classes) float, soft labels
        """
        B = images.size(0)
        # One-hot encode
        soft_labels = torch.zeros(
            B, self.num_classes, device=images.device, dtype=images.dtype
        )
        soft_labels.scatter_(1, labels.view(-1, 1), 1.0)

        r = random.random()
        if r < self.mixup_prob and self.mixup_alpha > 0:
            images, soft_labels = self._mixup(images, soft_labels)
        elif r < self.mixup_prob + self.cutmix_prob and self.cutmix_alpha > 0:
            images, soft_labels = self._cutmix(images, soft_labels)

        return images, soft_labels

    def _mixup(self, images, soft_labels):
        import numpy as np

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        idx = torch.randperm(images.size(0), device=images.device)
        images = lam * images + (1 - lam) * images[idx]
        soft_labels = lam * soft_labels + (1 - lam) * soft_labels[idx]
        return images, soft_labels

    def _cutmix(self, images, soft_labels):
        import numpy as np

        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        B, C, H, W = images.shape
        idx = torch.randperm(B, device=images.device)

        cut_ratio = math.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
        y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)

        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        # Adjust lambda based on actual cut area
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        soft_labels = lam * soft_labels + (1 - lam) * soft_labels[idx]
        return images, soft_labels
