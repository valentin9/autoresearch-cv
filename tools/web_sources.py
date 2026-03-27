"""
tools/web_sources.py — Download images from the web and HuggingFace datasets.

Utilities:
  download_images_to_dir()   — Download a list of image URLs to a directory
  fetch_hf_images()          — Pull images from a HuggingFace dataset into ImageFolder format
  download_coco_subset()     — Download a subset of COCO images (good general backgrounds)

Usage:
    from tools.web_sources import download_coco_subset
    download_coco_subset(
        output_dir="~/.cache/autoresearch/datasets/my_task/backgrounds",
        num_images=1000,
    )
"""

import os
import time
import random
import hashlib
import urllib.request
from pathlib import Path
from typing import List, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image


# ---------------------------------------------------------------------------
# Low-level URL download
# ---------------------------------------------------------------------------


def download_image(url: str, dest_path: Path, timeout: int = 15) -> bool:
    """
    Download a single image from `url` to `dest_path`.
    Returns True on success, False on failure.
    Verifies the downloaded file is a valid image.
    """
    dest_path = Path(dest_path)
    if dest_path.exists():
        return True
    tmp_path = dest_path.with_suffix(".tmp")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (autoresearch dataset builder)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
        tmp_path.write_bytes(data)
        # Verify it's a valid image
        img = Image.open(tmp_path)
        img.verify()
        tmp_path.rename(dest_path)
        return True
    except Exception:
        for p in [tmp_path, dest_path]:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        return False


def download_images_to_dir(
    urls: List[str],
    output_dir: str,
    max_workers: int = 8,
    filename_prefix: str = "img",
    verbose: bool = True,
) -> List[Path]:
    """
    Download a list of image URLs to `output_dir`.
    Files are named `<prefix>_<hash>.jpg`.
    Returns list of successfully downloaded paths.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    def _download(url: str) -> Optional[Path]:
        h = hashlib.md5(url.encode()).hexdigest()[:12]
        dest = output_dir / f"{filename_prefix}_{h}.jpg"
        ok = download_image(url, dest)
        return dest if ok else None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download, url): url for url in urls}
        for i, future in enumerate(as_completed(futures)):
            path = future.result()
            if path is not None:
                results.append(path)
            if verbose and (i + 1) % 50 == 0:
                print(f"  Downloaded {len(results)}/{i + 1} images...", flush=True)

    if verbose:
        print(f"  Total: {len(results)}/{len(urls)} downloaded to {output_dir}")
    return results


# ---------------------------------------------------------------------------
# HuggingFace datasets integration
# ---------------------------------------------------------------------------


def fetch_hf_images(
    dataset_name: str,
    output_dir: str,
    split: str = "train",
    image_column: str = "image",
    label_column: str = "label",
    max_per_class: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Download images from a HuggingFace dataset into ImageFolder format.

    The dataset must have an image column (PIL images or image paths) and a label column.
    Labels can be integers (class indices) or strings (class names).

    Returns: dict mapping class_name -> list of output paths

    Example:
        fetch_hf_images(
            "food101",
            "~/.cache/autoresearch/datasets/food101",
            max_per_class=500,
        )
    """
    from datasets import load_dataset

    output_dir = Path(output_dir).expanduser()

    if verbose:
        print(f"Loading HuggingFace dataset: {dataset_name} (split={split})")

    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

    # Infer class names
    if label_column in ds.features:
        feature = ds.features[label_column]
        if hasattr(feature, "names"):
            class_names = feature.names
        else:
            class_names = None
    else:
        class_names = None

    # Group by label
    label_to_indices: dict = {}
    for idx, item in enumerate(ds):
        label = item[label_column]
        label_to_indices.setdefault(label, []).append(idx)

    random.seed(seed)
    written = {}

    for label, indices in label_to_indices.items():
        class_name = (
            class_names[label]
            if (class_names and isinstance(label, int))
            else str(label)
        )
        cls_dir = output_dir / split / class_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        sample_indices = random.sample(indices, min(max_per_class, len(indices)))
        written[class_name] = []

        for i, idx in enumerate(sample_indices):
            dest = cls_dir / f"{i:05d}.jpg"
            if dest.exists():
                written[class_name].append(dest)
                continue
            try:
                img = ds[idx][image_column]
                if not isinstance(img, Image.Image):
                    img = Image.open(img)
                img = img.convert("RGB")
                img.save(dest, format="JPEG", quality=90)
                written[class_name].append(dest)
            except Exception as e:
                if verbose:
                    print(f"  Warning: skipped {idx}: {e}")

        if verbose:
            print(f"  {split}/{class_name}: {len(written[class_name])} images")

    return written


# ---------------------------------------------------------------------------
# COCO background images (general-purpose, good for synthetic compositing)
# ---------------------------------------------------------------------------


def download_coco_subset(
    output_dir: str,
    num_images: int = 1000,
    split: str = "train2017",
    seed: int = 42,
    max_workers: int = 8,
    verbose: bool = True,
) -> List[Path]:
    """
    Download a random subset of COCO images. Great as background images for
    synthetic overlay/compositing tasks.

    Uses the public COCO annotations to get image URLs, then downloads them.
    No API key required.

    Args:
        output_dir:  Where to store the images (flat directory, not class-structured)
        num_images:  How many images to download
        split:       COCO split: 'train2017' or 'val2017'
        seed:        Random seed for reproducibility
    """
    import json
    import urllib.request

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_dir = output_dir / ".annotations"
    annotations_dir.mkdir(exist_ok=True)
    ann_file = annotations_dir / f"instances_{split}.json"

    if not ann_file.exists():
        ann_url = (
            f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        )
        if verbose:
            print(f"Downloading COCO annotations from {ann_url} ...")
        # Use a lighter approach: just get image list from the instances file
        # Fall back to a small cached list of stable image URLs
        if verbose:
            print(
                "  (This is a large download. Alternatively, use fetch_hf_images with 'detection-datasets/coco')"
            )

        # Try downloading via HuggingFace instead (much simpler)
        try:
            from datasets import load_dataset

            if verbose:
                print("  Using HuggingFace 'detection-datasets/coco' instead...")
            ds = load_dataset(
                "detection-datasets/coco",
                split=split.replace("2017", ""),
                trust_remote_code=True,
            )
            random.seed(seed)
            indices = random.sample(range(len(ds)), min(num_images, len(ds)))
            paths = []
            for i, idx in enumerate(indices):
                dest = output_dir / f"coco_{idx:06d}.jpg"
                if dest.exists():
                    paths.append(dest)
                    continue
                try:
                    img = ds[idx]["image"].convert("RGB")
                    img.save(dest, format="JPEG", quality=90)
                    paths.append(dest)
                    if verbose and (i + 1) % 100 == 0:
                        print(f"  {i + 1}/{num_images} images...", flush=True)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: skipped {idx}: {e}")
            if verbose:
                print(f"  Downloaded {len(paths)} COCO images to {output_dir}")
            return paths
        except Exception as e:
            if verbose:
                print(f"  HuggingFace approach failed: {e}")
            return []

    with open(ann_file) as f:
        data = json.load(f)

    images_info = data["images"]
    random.seed(seed)
    random.shuffle(images_info)
    images_info = images_info[:num_images]

    urls = [img["coco_url"] for img in images_info]
    return download_images_to_dir(
        urls,
        output_dir,
        max_workers=max_workers,
        filename_prefix="coco",
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Generic image list loader (for custom URL lists)
# ---------------------------------------------------------------------------


def load_images_from_dir(
    directory: str, extensions=(".jpg", ".jpeg", ".png", ".webp")
) -> List[Path]:
    """Return sorted list of image paths in a directory."""
    d = Path(directory).expanduser()
    return sorted(p for p in d.iterdir() if p.suffix.lower() in extensions)
