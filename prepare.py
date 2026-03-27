"""
Data preparation and runtime utilities for vision autoresearch experiments.

One-time setup:
    python prepare.py                   # generates/validates dataset for the current scope
    python prepare.py --task overlay    # explicitly name the task (reads from scope.md)

Datasets are stored in ~/.cache/autoresearch/datasets/<task_name>/ in ImageFolder format:
    train/
        class_a/img1.jpg ...
        class_b/img2.jpg ...
    val/
        class_a/ ...
        class_b/ ...

This file is the fixed contract. DO NOT modify the evaluation harness (evaluate()).
The agent (train.py) may use make_dataloader() and evaluate() but must not change them.
"""

import os
import sys
import json
import time
import math

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

IMG_SIZE = 224  # input image resolution
TIME_BUDGET = 1200  # training time budget in seconds (20 minutes)
EVAL_BATCHES = 50  # number of val batches for evaluation
NUM_WORKERS = 4  # dataloader workers

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATASETS_DIR = os.path.join(CACHE_DIR, "datasets")
SCOPE_FILE = os.path.join(os.path.dirname(__file__), "scope.md")

# ---------------------------------------------------------------------------
# Scope loading
# ---------------------------------------------------------------------------


def load_scope() -> dict:
    """
    Load task scope from scope.md. Returns a dict with at minimum:
      task_name  : str  -- short slug used as directory name
      classes    : list[str]
      metric     : str  -- 'accuracy' | 'f1_macro' | 'f1_weighted'
    """
    if not os.path.exists(SCOPE_FILE):
        print(f"ERROR: scope.md not found at {SCOPE_FILE}")
        print("Create scope.md with the task definition before running prepare.py")
        sys.exit(1)
    with open(SCOPE_FILE) as f:
        content = f.read()
    # Extract JSON block fenced as ```json ... ```
    import re

    m = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not m:
        print(
            "ERROR: scope.md must contain a ```json { ... } ``` block with task metadata"
        )
        sys.exit(1)
    scope = json.loads(m.group(1))
    required = {"task_name", "classes", "metric"}
    missing = required - set(scope.keys())
    if missing:
        print(f"ERROR: scope.md JSON block is missing keys: {missing}")
        sys.exit(1)
    return scope


def get_dataset_dir(task_name: str) -> str:
    return os.path.join(DATASETS_DIR, task_name)


def get_num_classes(task_name: str) -> int:
    """Read number of classes from the dataset directory structure."""
    train_dir = os.path.join(get_dataset_dir(task_name), "train")
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Dataset not found at {train_dir}. Run prepare.py first.")
    classes = sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )
    return len(classes)


def get_class_names(task_name: str) -> list:
    train_dir = os.path.join(get_dataset_dir(task_name), "train")
    return sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# Training augmentation — intentionally moderate. Agent may tune in train.py.
TRAIN_TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

VAL_TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------


def make_dataloader(
    task_name: str,
    batch_size: int,
    split: str,
    train_transform=None,
    val_transform=None,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Returns a DataLoader for the given task and split ('train' or 'val').
    Images must be stored as ImageFolder format under get_dataset_dir(task_name)/<split>/.

    The returned loader yields (images, labels) where:
      images: FloatTensor (B, 3, IMG_SIZE, IMG_SIZE) on CPU (moved to device in train.py)
      labels: LongTensor  (B,)

    pin_memory: set True only when using CUDA; False for MPS/CPU.
    """
    assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"
    split_dir = os.path.join(get_dataset_dir(task_name), split)
    if not os.path.isdir(split_dir):
        raise RuntimeError(
            f"Dataset split '{split}' not found at {split_dir}. Run prepare.py first."
        )

    if split == "train":
        transform = train_transform if train_transform is not None else TRAIN_TRANSFORM
    else:
        transform = val_transform if val_transform is not None else VAL_TRANSFORM

    dataset = datasets.ImageFolder(split_dir, transform=transform)
    shuffle = split == "train"
    # persistent_workers can deadlock on macOS with some Python versions
    use_persistent = (NUM_WORKERS > 0) and (
        os.name != "posix" or os.uname().sysname != "Darwin"
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        persistent_workers=use_persistent,
    )
    return loader


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric contract)
# ---------------------------------------------------------------------------


def _get_device() -> str:
    """Resolve the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(
    model: nn.Module,
    task_name: str,
    batch_size: int,
    metric: str = "f1_macro",
    device: str = "auto",
) -> dict:
    """
    Fixed evaluation harness. Returns a dict with:
      primary_metric : float  -- the value of `metric` (higher is better)
      accuracy       : float
      f1_macro       : float
      f1_weighted    : float
      num_samples    : int

    metric must be one of: 'accuracy', 'f1_macro', 'f1_weighted'

    The model is called as: logits = model(images)  where images is (B, 3, H, W) on `device`.
    logits shape: (B, num_classes).
    """
    assert metric in ("accuracy", "f1_macro", "f1_weighted"), (
        f"metric must be accuracy/f1_macro/f1_weighted, got '{metric}'"
    )

    if device == "auto":
        device = _get_device()

    # bf16 on CUDA/MPS, fp32 on CPU
    _dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    model.eval()
    val_loader = make_dataloader(
        task_name, batch_size, "val", pin_memory=(device == "cuda")
    )

    all_preds = []
    all_labels = []

    for i, (images, labels) in enumerate(val_loader):
        if i >= EVAL_BATCHES:
            break
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device, dtype=_dtype):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    primary = {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}[
        metric
    ]

    return {
        "primary_metric": primary,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "num_samples": len(all_labels),
    }


# ---------------------------------------------------------------------------
# Dataset validation helper
# ---------------------------------------------------------------------------


def validate_dataset(task_name: str) -> bool:
    """
    Check that dataset exists and has reasonable structure.
    Returns True if valid, prints issues and returns False otherwise.
    """
    base = get_dataset_dir(task_name)
    ok = True
    for split in ("train", "val"):
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            print(f"  MISSING: {split_dir}")
            ok = False
            continue
        classes = sorted(
            d
            for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )
        if len(classes) == 0:
            print(f"  NO CLASS DIRS in {split_dir}")
            ok = False
            continue
        total = 0
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            imgs = [
                f
                for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            total += len(imgs)
            print(f"  {split}/{cls}: {len(imgs)} images")
        print(f"  {split} total: {total} images across {len(classes)} classes")
        if total == 0:
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Main (one-time prep invoked by human before starting the agent)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare dataset for autoresearch vision task"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name override (default: read from scope.md)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing dataset, do not generate",
    )
    args = parser.parse_args()

    scope = load_scope()
    task_name = args.task or scope["task_name"]
    classes = scope["classes"]
    metric = scope["metric"]

    print(f"Task:    {task_name}")
    print(f"Classes: {classes}")
    print(f"Metric:  {metric}")
    print(f"Dataset: {get_dataset_dir(task_name)}")
    print()

    if args.validate_only:
        valid = validate_dataset(task_name)
        sys.exit(0 if valid else 1)

    # Check if dataset already exists
    if os.path.isdir(get_dataset_dir(task_name)):
        print("Dataset directory exists. Validating...")
        valid = validate_dataset(task_name)
        if valid:
            print("\nDataset looks good. Ready to train.")
            print(f"  Run: uv run train.py")
        else:
            print("\nDataset has issues. You may need to regenerate it.")
            print("  See tools/ for dataset generation utilities.")
        sys.exit(0 if valid else 1)

    print("No dataset found.")
    print()
    print("Before running the agent, you need to generate a dataset.")
    print("Options:")
    print("  1. Use or write a script in tools/ to generate/download the dataset")
    print(f"  2. Manually place images in: {get_dataset_dir(task_name)}/train/<class>/")
    print(f"                           and: {get_dataset_dir(task_name)}/val/<class>/")
    print()
    print(
        "The agent (OpenCode) will also attempt to build the dataset as part of Phase 1."
    )
    print("See program.md for the full protocol.")
    sys.exit(1)
