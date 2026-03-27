"""
tools/font_generator.py — Synthetic font classification dataset generator.

Renders text strings in target fonts onto varied backgrounds, producing a
labeled ImageFolder dataset for font classification research.

Usage:
    uv run tools/font_generator.py

Or from Python:
    from tools.font_generator import FontDatasetGenerator
    gen = FontDatasetGenerator(
        output_dir="~/.cache/autoresearch/datasets/font_detection",
        classes=[...],
        train_per_class=2000,
        val_per_class=400,
    )
    gen.generate()
"""

import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from tools.synthetic import SyntheticDatasetGenerator

# ---------------------------------------------------------------------------
# Text content pools
# ---------------------------------------------------------------------------

PANGRAMS = [
    "The quick brown fox jumps over the lazy dog",
    "Pack my box with five dozen liquor jugs",
    "How vexingly quick daft zebras jump",
    "The five boxing wizards jump quickly",
    "Sphinx of black quartz judge my vow",
    "Brick quiz whangs jumpy veldt fox",
    "Blowzy night-frumps vex'd jack q",
    "Cwm fjord bank glyphs vext quiz",
]

WORDS = [
    "Typography",
    "Alphabet",
    "Design",
    "Script",
    "Letter",
    "Glyph",
    "Serif",
    "Weight",
    "Kerning",
    "Ligature",
    "Baseline",
    "Ascender",
    "Descender",
    "Contrast",
    "Rhythm",
    "Balance",
    "Form",
    "Shape",
    "Structure",
    "Style",
    "Hello",
    "World",
    "Python",
    "Neural",
    "Network",
    "Vision",
    "Image",
    "Research",
    "Model",
    "Training",
    "Dataset",
    "Class",
    "Label",
    "Font",
    "OpenAI",
    "Google",
    "Meta",
    "Apple",
    "Linux",
    "Ubuntu",
    "Windows",
    "abcdefgh",
    "ijklmnop",
    "qrstuvwx",
    "ABCDEFGH",
    "IJKLMNOP",
    "QRSTUVWX",
    "0123456789",
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
]

SENTENCES = [
    "Typography is the art of arranging type",
    "Good design is invisible",
    "Form follows function",
    "Less is more",
    "Keep it simple",
    "Hello World",
    "The lazy dog",
    "Quick brown fox",
    "Vision research",
    "Font detection",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789 !@#$%",
    "To be or not to be",
    "All that glitters is not gold",
]

ALL_TEXTS = PANGRAMS + WORDS + SENTENCES

# ---------------------------------------------------------------------------
# Google Fonts download helper
# ---------------------------------------------------------------------------

# Direct download URLs for each font (Google Fonts zip archives)
# Direct GitHub raw URLs for Google Fonts (reliable, no redirect issues)
_GH = "https://raw.githubusercontent.com/google/fonts/main"
FONT_DIRECT_URLS = {
    "Roboto": f"{_GH}/ofl/roboto/Roboto[wdth,wght].ttf",
    "Open Sans": f"{_GH}/ofl/opensans/OpenSans[wdth,wght].ttf",
    "Lato": f"{_GH}/ofl/lato/Lato-Regular.ttf",
    "Montserrat": f"{_GH}/ofl/montserrat/Montserrat[wght].ttf",
    "Oswald": f"{_GH}/ofl/oswald/Oswald[wght].ttf",
    "Raleway": f"{_GH}/ofl/raleway/Raleway[wght].ttf",
    "Merriweather": f"{_GH}/ofl/merriweather/Merriweather[opsz,wdth,wght].ttf",
    "Playfair Display": f"{_GH}/ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
    "Source Sans Pro": f"{_GH}/ofl/sourcesans3/SourceSans3[wght].ttf",
    "PT Sans": f"{_GH}/ofl/ptsans/PT_Sans-Web-Regular.ttf",
    "Nunito": f"{_GH}/ofl/nunito/Nunito[wght].ttf",
    "Ubuntu": f"{_GH}/ufl/ubuntu/Ubuntu-Regular.ttf",
}
FONT_URLS = {k: None for k in ["Arial", "Times New Roman", "Georgia"]}

# Variant preference order for each font (we want Regular/400 weight)
VARIANT_PREFERENCE = [
    "Regular",
    "400",
    "-Regular",
    "_Regular",
    "Bold",
    "700",
    "-Bold",
    "_Bold",
    "Medium",
    "500",
    "Light",
    "300",
]

FONT_CACHE_DIR = Path.home() / ".cache" / "autoresearch" / "fonts"


def _find_system_font(font_name: str) -> Optional[Path]:
    """Try to find a system font by name."""
    name_lower = font_name.lower().replace(" ", "")
    search_dirs = [
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path.home() / "Library" / "Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("C:/Windows/Fonts"),
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.rglob("*.ttf"):
            stem = f.stem.lower().replace(" ", "").replace("-", "").replace("_", "")
            if name_lower.replace(" ", "") in stem:
                return f
        for f in d.rglob("*.otf"):
            stem = f.stem.lower().replace(" ", "").replace("-", "").replace("_", "")
            if name_lower.replace(" ", "") in stem:
                return f
    return None


def _pick_best_variant(ttf_files: List[Path]) -> Optional[Path]:
    """Pick the best variant from a list of .ttf/.otf files."""
    for pref in VARIANT_PREFERENCE:
        for f in ttf_files:
            if pref.lower() in f.stem.lower():
                return f
    # fallback: pick alphabetically first
    return sorted(ttf_files)[0] if ttf_files else None


def download_font(font_name: str, cache_dir: Path = FONT_CACHE_DIR) -> Optional[Path]:
    """
    Download a font from Google Fonts or find it on the system.
    Returns path to a .ttf/.otf file, or None if not found.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    font_dir = cache_dir / font_name.replace(" ", "_")
    font_dir.mkdir(exist_ok=True)

    # Check if already cached
    existing = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
    if existing:
        return _pick_best_variant(existing)

    # If there's a direct URL, prefer downloading over system font lookup
    # (avoids picking up wrong variants like Roboto Mono for Roboto)
    has_direct_url = font_name in FONT_DIRECT_URLS

    # Try system fonts only for fonts without a direct URL
    if not has_direct_url:
        sys_font = _find_system_font(font_name)
        if sys_font:
            print(f"  Found system font: {font_name} -> {sys_font}")
            return sys_font

    # Try direct GitHub raw URL
    url = FONT_DIRECT_URLS.get(font_name)
    if url:
        ttf_path = font_dir / Path(url).name
        try:
            print(f"  Downloading {font_name} from GitHub...", end=" ", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            ttf_path.write_bytes(data)
            print(f"OK ({ttf_path.name})")
            return ttf_path
        except Exception as e:
            print(f"FAILED ({e})")
            return None

    return None


# ---------------------------------------------------------------------------
# Background generation
# ---------------------------------------------------------------------------


def _solid_bg(w: int, h: int) -> Image.Image:
    color = (
        random.randint(180, 255),
        random.randint(180, 255),
        random.randint(180, 255),
    )
    return Image.new("RGB", (w, h), color)


def _dark_bg(w: int, h: int) -> Image.Image:
    color = (
        random.randint(0, 80),
        random.randint(0, 80),
        random.randint(0, 80),
    )
    return Image.new("RGB", (w, h), color)


def _gradient_bg(w: int, h: int) -> Image.Image:
    c1 = np.array([random.randint(150, 255) for _ in range(3)], dtype=np.float32)
    c2 = np.array([random.randint(150, 255) for _ in range(3)], dtype=np.float32)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        t = i / max(h - 1, 1)
        arr[i] = ((1 - t) * c1 + t * c2).astype(np.uint8)
    return Image.fromarray(arr)


def _noise_bg(w: int, h: int) -> Image.Image:
    base_color = np.array(
        [random.randint(200, 255) for _ in range(3)], dtype=np.float32
    )
    noise = np.random.normal(0, 12, (h, w, 3))
    arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _make_background(w: int, h: int) -> Image.Image:
    dark = random.random() < 0.3
    r = random.random()
    if dark:
        return _dark_bg(w, h)
    if r < 0.4:
        return _solid_bg(w, h)
    elif r < 0.7:
        return _gradient_bg(w, h)
    else:
        return _noise_bg(w, h)


# ---------------------------------------------------------------------------
# Text color
# ---------------------------------------------------------------------------


def _text_color(bg: Image.Image, x: int, y: int) -> tuple:
    """Pick a text color that contrasts with the local background region."""
    region = bg.crop((x, y, min(x + 50, bg.width), min(y + 30, bg.height)))
    arr = np.array(region, dtype=np.float32)
    luminance = (
        0.299 * arr[:, :, 0].mean()
        + 0.587 * arr[:, :, 1].mean()
        + 0.114 * arr[:, :, 2].mean()
    )
    if luminance > 128:
        # dark text on light bg
        v = random.randint(0, 80)
        return (v, v, v)
    else:
        # light text on dark bg
        v = random.randint(175, 255)
        return (v, v, v)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


class FontDatasetGenerator(SyntheticDatasetGenerator):
    """
    Generates a font classification dataset by rendering text in each font
    onto varied backgrounds with size/color/positioning variation.

    Each sample is a 224x224 RGB image containing rendered text in a specific font.
    """

    def __init__(
        self,
        output_dir: str,
        classes: List[str],
        train_per_class: int = 2000,
        val_per_class: int = 400,
        img_size: int = 224,
        seed: int = 42,
    ):
        super().__init__(
            output_dir=output_dir,
            classes=classes,
            train_per_class=train_per_class,
            val_per_class=val_per_class,
            seed=seed,
        )
        self.img_size = img_size
        self._font_paths: dict = {}  # font_name -> Path or None
        self._pil_font_cache: dict = {}  # (font_name, size) -> ImageFont

    def _setup_fonts(self):
        """Download/locate all fonts. Called once before generation."""
        print("Setting up fonts...")
        for font_name in self.classes:
            path = download_font(font_name)
            self._font_paths[font_name] = path
            if path is None:
                print(
                    f"  WARNING: Could not find font for '{font_name}', will use PIL default"
                )

    def _get_pil_font(self, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        key = (font_name, size)
        if key not in self._pil_font_cache:
            path = self._font_paths.get(font_name)
            if path is not None:
                try:
                    self._pil_font_cache[key] = ImageFont.truetype(str(path), size)
                    return self._pil_font_cache[key]
                except Exception:
                    pass
            # Fallback to default
            self._pil_font_cache[key] = ImageFont.load_default()
        return self._pil_font_cache[key]

    def generate(self, overwrite: bool = False, verbose: bool = True) -> None:
        self._setup_fonts()
        super().generate(overwrite=overwrite, verbose=verbose)

    def generate_sample(self, class_name: str, split: str, index: int) -> Image.Image:
        """Render a random text string in the given font on a varied background."""
        rng_state = random.getstate()
        np_state = np.random.get_state()

        # Deterministic per (class, split, index) for reproducibility
        seed = hash((class_name, split, index)) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)

        img = self._render_sample(class_name)

        random.setstate(rng_state)
        np.random.set_state(np_state)
        return img

    def _render_sample(self, font_name: str) -> Image.Image:
        """
        Render text in the target font as a tight crop.

        Strategy: render text on a large canvas, find the tight bounding box of
        the actual pixels, add small padding, then resize to img_size×img_size.
        This maximizes the fraction of the image occupied by font-discriminative
        glyph shapes rather than background.
        """
        W = H = self.img_size

        # Pick text — use strings that expose multiple glyphs across all fonts
        text = random.choice(ALL_TEXTS)
        # Use a moderate-length string (not too short, not too long)
        words = text.split()
        if len(words) > 5:
            text = " ".join(words[: random.randint(3, 5)])
        elif len(words) < 2 and len(text) > 10:
            text = text[: random.randint(8, 15)]

        # Background: mostly clean (white/light) with occasional dark
        dark_bg = random.random() < 0.3
        if dark_bg:
            bg_val = random.randint(10, 60)
            text_val = random.randint(180, 255)
        else:
            bg_val = random.randint(220, 255)
            text_val = random.randint(0, 60)
        bg_color = (bg_val, bg_val, bg_val)
        text_color = (text_val, text_val, text_val)

        # Font size: render at large size, then resize → better glyph detail
        font_size = random.randint(48, 96)
        font = self._get_pil_font(font_name, font_size)

        # Render on a large canvas
        canvas_w = 1200
        canvas_h = 256
        canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
        draw = ImageDraw.Draw(canvas)

        # Measure text
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = font_size * len(text) // 2, font_size

        # If too wide, truncate
        while tw > canvas_w - 20 and len(text) > 3:
            text = text[:-2].rstrip()
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                break

        # Center on canvas
        x = max(5, (canvas_w - tw) // 2)
        y = max(5, (canvas_h - th) // 2)

        # Slight color randomization for variation
        r_jitter = random.randint(-15, 15)
        g_jitter = random.randint(-15, 15)
        b_jitter = random.randint(-15, 15)
        actual_text_color = (
            max(0, min(255, text_val + r_jitter)),
            max(0, min(255, text_val + g_jitter)),
            max(0, min(255, text_val + b_jitter)),
        )

        draw.text((x, y), text, font=font, fill=actual_text_color)

        # Find tight bounding box of non-background pixels
        arr = np.array(canvas)
        if dark_bg:
            # Text is light: find bright pixels
            mask = arr.max(axis=2) > (bg_val + 20)
        else:
            # Text is dark: find dark pixels
            mask = arr.min(axis=2) < (bg_val - 20)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Add padding (10% of text size)
            pad = max(8, int((rmax - rmin) * 0.12))
            rmin = max(0, rmin - pad)
            rmax = min(canvas_h - 1, rmax + pad)
            cmin = max(0, cmin - pad)
            cmax = min(canvas_w - 1, cmax + pad)
            crop = canvas.crop((cmin, rmin, cmax + 1, rmax + 1))
        else:
            # Fallback if no text rendered
            crop = canvas.crop((x, y, x + tw + 10, y + th + 10))

        # Resize to target size
        img = crop.resize((W, H), Image.LANCZOS)

        # Light augmentation: mild blur and noise
        if random.random() < 0.1:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
        if random.random() < 0.1:
            arr2 = np.array(img, dtype=np.float32)
            arr2 += np.random.normal(0, random.uniform(2, 8), arr2.shape)
            img = Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8))

        return img


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import re

    scope_path = Path(__file__).parent.parent / "scope.md"
    content = scope_path.read_text()
    m = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    scope = json.loads(m.group(1))

    output_dir = (
        Path.home() / ".cache" / "autoresearch" / "datasets" / scope["task_name"]
    )

    gen = FontDatasetGenerator(
        output_dir=str(output_dir),
        classes=scope["classes"],
        train_per_class=2000,
        val_per_class=400,
    )
    gen.generate()

    print("\nValidating...")
    from prepare import validate_dataset

    validate_dataset(scope["task_name"])
