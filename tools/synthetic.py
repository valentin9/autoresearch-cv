"""
tools/synthetic.py — Base classes and helpers for synthetic image dataset generation.

Usage example:
    from tools.synthetic import SyntheticDatasetGenerator, DatasetSplit

    class MyGenerator(SyntheticDatasetGenerator):
        def generate_sample(self, class_name: str, index: int) -> Image.Image:
            # ... return a PIL Image
            pass

    gen = MyGenerator(
        output_dir="~/.cache/autoresearch/datasets/my_task",
        classes=["class_a", "class_b"],
        train_per_class=2000,
        val_per_class=400,
    )
    gen.generate()
"""

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from PIL import Image
import numpy as np


class SyntheticDatasetGenerator(ABC):
    """
    Base class for synthetic image dataset generators.

    Subclass this, implement `generate_sample()`, then call `generate()`.
    The output is written to `output_dir` in ImageFolder format:

        output_dir/
          train/
            class_a/  0000.jpg  0001.jpg ...
            class_b/  ...
          val/
            class_a/  ...
            class_b/  ...
    """

    def __init__(
        self,
        output_dir: str,
        classes: List[str],
        train_per_class: int = 2000,
        val_per_class: int = 400,
        seed: int = 42,
        image_format: str = "JPEG",
        jpeg_quality: int = 90,
    ):
        self.output_dir = Path(output_dir).expanduser()
        self.classes = classes
        self.train_per_class = train_per_class
        self.val_per_class = val_per_class
        self.seed = seed
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality

    @abstractmethod
    def generate_sample(self, class_name: str, split: str, index: int) -> Image.Image:
        """
        Generate a single sample image for the given class and split.

        Args:
            class_name: one of self.classes
            split:      'train' or 'val'
            index:      sample index within this class/split

        Returns:
            A PIL.Image.Image in RGB mode.
        """
        raise NotImplementedError

    def generate(self, overwrite: bool = False, verbose: bool = True) -> None:
        """
        Generate the full dataset. Skips already-existing images unless overwrite=True.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        splits = [("train", self.train_per_class), ("val", self.val_per_class)]

        for split, count in splits:
            for cls in self.classes:
                cls_dir = self.output_dir / split / cls
                cls_dir.mkdir(parents=True, exist_ok=True)

                ext = (
                    "jpg"
                    if self.image_format.upper() in ("JPEG", "JPG")
                    else self.image_format.lower()
                )
                existing = len(list(cls_dir.glob(f"*.{ext}")))

                if existing >= count and not overwrite:
                    if verbose:
                        print(
                            f"  {split}/{cls}: {existing} images already exist, skipping"
                        )
                    continue

                start_idx = 0 if overwrite else existing
                to_generate = count - start_idx

                if verbose:
                    print(
                        f"  {split}/{cls}: generating {to_generate} images...",
                        flush=True,
                    )

                for i in range(start_idx, count):
                    img = self.generate_sample(cls, split, i)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    out_path = cls_dir / f"{i:05d}.{ext}"
                    save_kwargs = {}
                    if self.image_format.upper() in ("JPEG", "JPG"):
                        save_kwargs["quality"] = self.jpeg_quality
                    img.save(out_path, format=self.image_format, **save_kwargs)

                if verbose:
                    print(f"  {split}/{cls}: done ({count} total)")

        if verbose:
            print(f"\nDataset written to: {self.output_dir}")
            self._print_summary()

    def _print_summary(self) -> None:
        for split in ("train", "val"):
            total = 0
            for cls in self.classes:
                cls_dir = self.output_dir / split / cls
                n = len(list(cls_dir.glob("*.*"))) if cls_dir.exists() else 0
                total += n
                print(f"  {split}/{cls}: {n}")
            print(f"  {split} total: {total}")

    # ------------------------------------------------------------------
    # Convenience helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def random_color(alpha: bool = False) -> tuple:
        """Return a random RGBA or RGB tuple."""
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        if alpha:
            return r, g, b, random.randint(64, 255)
        return r, g, b

    @staticmethod
    def contrasting_color(bg_color: tuple) -> tuple:
        """Return black or white depending on the luminance of bg_color."""
        r, g, b = bg_color[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return (0, 0, 0) if luminance > 128 else (255, 255, 255)

    @staticmethod
    def blend(
        base: Image.Image, overlay: Image.Image, alpha: float = 0.8
    ) -> Image.Image:
        """Alpha-blend overlay onto base at the given global alpha."""
        base_rgba = base.convert("RGBA")
        overlay_rgba = overlay.convert("RGBA")
        # Scale overlay alpha channel by global alpha
        r, g, b, a = overlay_rgba.split()
        a = a.point(lambda x: int(x * alpha))
        overlay_rgba = Image.merge("RGBA", (r, g, b, a))
        result = Image.alpha_composite(base_rgba, overlay_rgba)
        return result.convert("RGB")
