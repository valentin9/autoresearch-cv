"""
tools/overlay.py — Synthetic image overlay dataset generator.

Generates a dataset for classifying whether an image has:
  - no overlay (clean)
  - a text overlay (subtitle, caption, watermark, meme text, etc.)
  - a graphic overlay (logo, banner, channel bug, frame, etc.)

Requires background images (any diverse set of photos works well).
Use tools/web_sources.download_coco_subset() to get backgrounds.

Usage:
    python tools/overlay.py \
        --backgrounds ~/.cache/autoresearch/datasets/coco_backgrounds \
        --output      ~/.cache/autoresearch/datasets/image_overlay \
        --train       2000 \
        --val         400

Or from Python:
    from tools.overlay import OverlayDatasetGenerator
    gen = OverlayDatasetGenerator(
        backgrounds_dir="~/.cache/autoresearch/datasets/coco_backgrounds",
        output_dir="~/.cache/autoresearch/datasets/image_overlay",
        train_per_class=2000,
        val_per_class=400,
    )
    gen.generate()
"""

import os
import random
import math
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from tools.synthetic import SyntheticDatasetGenerator
from tools.web_sources import load_images_from_dir


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------


def _get_system_fonts() -> List[str]:
    """Return a list of available truetype font paths on the system."""
    import subprocess, sys

    paths = []

    # Common font directories
    font_dirs = [
        "/usr/share/fonts",  # Linux
        "/usr/local/share/fonts",  # Linux local
        "/System/Library/Fonts",  # macOS system
        "/Library/Fonts",  # macOS user
        os.path.expanduser("~/Library/Fonts"),  # macOS personal
        "C:/Windows/Fonts",  # Windows
    ]

    for d in font_dirs:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        paths.append(os.path.join(root, f))

    return paths if paths else []


_SYSTEM_FONTS: Optional[List[str]] = None


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a PIL font at the given size. Falls back to default if no system fonts."""
    global _SYSTEM_FONTS
    if _SYSTEM_FONTS is None:
        _SYSTEM_FONTS = _get_system_fonts()

    if _SYSTEM_FONTS:
        font_path = random.choice(_SYSTEM_FONTS)
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass

    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Sample text corpora
# ---------------------------------------------------------------------------

LOREM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute "
    "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia "
    "deserunt mollit anim id est laborum"
).split()

NEWS_PHRASES = [
    "Breaking News",
    "Live",
    "EXCLUSIVE",
    "DEVELOPING STORY",
    "Watch Live",
    "Today at 9PM",
    "Subscribe Now",
    "Learn More",
    "Click Here",
    "SALE 50% OFF",
    "Limited Time Offer",
    "New Episode",
    "Season Finale",
    "Official Channel",
    "Verified",
    "AD",
    "Sponsored",
    "Follow Us",
    "Like and Subscribe",
    "Powered by",
]


def random_text(max_words: int = 6) -> str:
    if random.random() < 0.3:
        return random.choice(NEWS_PHRASES)
    n = random.randint(2, max_words)
    return " ".join(random.choices(LOREM_WORDS, k=n)).title()


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def add_text_overlay(img: Image.Image) -> Image.Image:
    """
    Add a random text overlay to a PIL image (in-place on a copy).
    Simulates: subtitles, captions, watermarks, meme text, channel IDs.
    """
    img = img.copy().convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # Random position: bottom-third subtitle, corner watermark, center meme
    position_type = random.choice(["subtitle", "watermark", "meme", "corner_id"])

    text = random_text(max_words=7)
    font_size = random.randint(
        max(12, H // 30),
        max(24, H // 10),
    )
    font = get_font(font_size)

    # Measure text
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * font_size // 2, font_size

    # Text color
    if random.random() < 0.6:
        text_color = (255, 255, 255, random.randint(200, 255))  # white
    elif random.random() < 0.5:
        text_color = (0, 0, 0, random.randint(200, 255))  # black
    else:
        r, g, b = (
            random.randint(180, 255),
            random.randint(180, 255),
            random.randint(180, 255),
        )
        text_color = (r, g, b, random.randint(180, 255))

    if position_type == "subtitle":
        x = random.randint(max(0, W // 2 - tw // 2 - 20), max(1, W // 2 - tw // 2 + 20))
        y = random.randint(int(H * 0.72), int(H * 0.88)) - th
        # Semi-transparent background bar
        if random.random() < 0.7:
            pad = 8
            bar_color = (0, 0, 0, random.randint(120, 200))
            draw.rectangle(
                [x - pad, y - pad, x + tw + pad, y + th + pad], fill=bar_color
            )

    elif position_type == "watermark":
        corner = random.choice(["tl", "tr", "bl", "br"])
        margin = random.randint(10, 40)
        if corner == "tl":
            x, y = margin, margin
        elif corner == "tr":
            x, y = W - tw - margin, margin
        elif corner == "bl":
            x, y = margin, H - th - margin
        else:
            x, y = W - tw - margin, H - th - margin
        # Low opacity for watermarks
        a = random.randint(80, 160)
        text_color = text_color[:3] + (a,)

    elif position_type == "meme":
        # Big text, top or bottom
        font_size = random.randint(H // 12, H // 6)
        font = get_font(font_size)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(text) * font_size // 2, font_size
        x = max(0, (W - tw) // 2)
        y = random.choice([H // 20, H - th - H // 20])
        # Outline effect
        draw.text((x - 2, y - 2), text, font=font, fill=(0, 0, 0, 255))
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 255))

    else:  # corner_id (channel bug)
        corner = random.choice(["tl", "tr"])
        margin = random.randint(5, 20)
        if corner == "tl":
            x, y = margin, margin
        else:
            x, y = W - tw - margin, margin

    draw.text((x, y), text, font=font, fill=text_color)
    return img.convert("RGB")


def add_graphic_overlay(img: Image.Image) -> Image.Image:
    """
    Add a random graphic overlay to a PIL image.
    Simulates: channel logos (colored rectangles/shapes), banners, frames, bugs.
    """
    img = img.copy().convert("RGBA")
    W, H = img.size
    overlay_type = random.choice(["logo_bug", "banner", "frame", "badge"])

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Random solid color for the graphic element
    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    alpha = random.randint(180, 240)
    color = (r, g, b, alpha)

    if overlay_type == "logo_bug":
        # Small colored rectangle in a corner (like a TV channel bug)
        size = random.randint(W // 12, W // 6)
        corner = random.choice(["tl", "tr", "bl", "br"])
        margin = random.randint(8, 24)
        if corner == "tl":
            x1, y1 = margin, margin
        elif corner == "tr":
            x1, y1 = W - size - margin, margin
        elif corner == "bl":
            x1, y1 = margin, H - size - margin
        else:
            x1, y1 = W - size - margin, H - size - margin
        x2, y2 = x1 + size, y1 + size

        # Rounded or irregular shape
        shape = random.choice(["rect", "ellipse", "diamond"])
        if shape == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:  # diamond
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = size // 2
            draw.polygon([(cx, y1), (x2, cy), (cx, y2), (x1, cy)], fill=color)

        # Optionally add a small letter/symbol
        if random.random() < 0.5:
            letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            font = get_font(size // 2)
            lw = size // 3
            draw.text(
                (x1 + size // 4, y1 + size // 6),
                letter,
                font=font,
                fill=(255, 255, 255, 220),
            )

    elif overlay_type == "banner":
        # Horizontal bar at top or bottom
        bar_h = random.randint(H // 12, H // 5)
        position = random.choice(["top", "bottom"])
        if position == "top":
            draw.rectangle([0, 0, W, bar_h], fill=color)
        else:
            draw.rectangle([0, H - bar_h, W, H], fill=color)

    elif overlay_type == "frame":
        # Border frame around the image
        border = random.randint(W // 30, W // 12)
        draw.rectangle([0, 0, W, border], fill=color)  # top
        draw.rectangle([0, H - border, W, H], fill=color)  # bottom
        draw.rectangle([0, 0, border, H], fill=color)  # left
        draw.rectangle([W - border, 0, W, H], fill=color)  # right

    elif overlay_type == "badge":
        # A pill/badge shape somewhere prominent
        bw = random.randint(W // 6, W // 3)
        bh = random.randint(H // 16, H // 8)
        x1 = random.randint(0, W - bw)
        y1 = random.randint(0, H // 3)
        x2, y2 = x1 + bw, y1 + bh
        radius = bh // 2
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=color)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------


class OverlayDatasetGenerator(SyntheticDatasetGenerator):
    """
    Generates synthetic image overlay classification dataset.

    Classes:
      clean          — unmodified background image
      text_overlay   — background + text rendered on top
      graphic_overlay — background + colored graphic shape/banner on top

    Requires a directory of background images (any diverse photos).
    """

    VALID_CLASSES = {"clean", "text_overlay", "graphic_overlay"}

    def __init__(
        self,
        backgrounds_dir: str,
        output_dir: str,
        classes: Optional[List[str]] = None,
        train_per_class: int = 2000,
        val_per_class: int = 400,
        seed: int = 42,
    ):
        classes = classes or ["clean", "text_overlay", "graphic_overlay"]
        invalid = set(classes) - self.VALID_CLASSES
        if invalid:
            raise ValueError(f"Invalid classes: {invalid}. Valid: {self.VALID_CLASSES}")

        super().__init__(
            output_dir=output_dir,
            classes=classes,
            train_per_class=train_per_class,
            val_per_class=val_per_class,
            seed=seed,
        )
        self.backgrounds_dir = Path(backgrounds_dir).expanduser()
        self._backgrounds: Optional[List[Path]] = None

    def _load_backgrounds(self) -> List[Path]:
        if self._backgrounds is None:
            self._backgrounds = load_images_from_dir(self.backgrounds_dir)
            if not self._backgrounds:
                raise RuntimeError(
                    f"No background images found in {self.backgrounds_dir}. "
                    "Run tools/web_sources.download_coco_subset() first."
                )
        return self._backgrounds

    def generate_sample(self, class_name: str, split: str, index: int) -> Image.Image:
        backgrounds = self._load_backgrounds()
        bg_path = backgrounds[index % len(backgrounds)]
        img = Image.open(bg_path).convert("RGB")

        # Resize to a reasonable working size
        img = img.resize((512, 512), Image.BILINEAR)

        if class_name == "clean":
            return img
        elif class_name == "text_overlay":
            return add_text_overlay(img)
        elif class_name == "graphic_overlay":
            return add_graphic_overlay(img)
        else:
            raise ValueError(f"Unknown class: {class_name}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate overlay detection dataset")
    parser.add_argument(
        "--backgrounds", required=True, help="Directory containing background images"
    )
    parser.add_argument(
        "--output",
        default="~/.cache/autoresearch/datasets/image_overlay",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--train", type=int, default=2000, help="Number of train images per class"
    )
    parser.add_argument(
        "--val", type=int, default=400, help="Number of val images per class"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["clean", "text_overlay", "graphic_overlay"],
        help="Classes to generate",
    )
    args = parser.parse_args()

    gen = OverlayDatasetGenerator(
        backgrounds_dir=args.backgrounds,
        output_dir=args.output,
        classes=args.classes,
        train_per_class=args.train,
        val_per_class=args.val,
    )
    gen.generate()
