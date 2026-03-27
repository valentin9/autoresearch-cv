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
# Exact system font paths for fonts not on Google Fonts (Microsoft/system fonts)
# These are pinned to avoid fuzzy matching picking wrong fonts (e.g. SFGeorgian ≠ Georgia)
SYSTEM_FONT_EXACT = {
    "Arial": [
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    ],
    "Times New Roman": [
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    ],
    "Georgia": [
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/Library/Fonts/Georgia.ttf",
        "C:/Windows/Fonts/georgia.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Georgia.ttf",
    ],
}

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

    # For system fonts with exact known paths, use those directly (avoids fuzzy match)
    if font_name in SYSTEM_FONT_EXACT:
        for candidate in SYSTEM_FONT_EXACT[font_name]:
            p = Path(candidate)
            if p.exists():
                print(f"  Found system font: {font_name} -> {p}")
                return p
        print(f"  WARNING: {font_name} not found at known system paths")
        return None

    # If there's a direct URL, prefer downloading over system font lookup
    # (avoids picking up wrong variants like Roboto Mono for Roboto)
    if font_name in FONT_DIRECT_URLS:
        url = FONT_DIRECT_URLS[font_name]
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

        Strategy: build a multi-line paragraph (2-4 lines, ≥40 chars total),
        render on a generously-sized canvas, tight-crop to the actual pixel
        bounding box, then letterbox-resize to img_size×img_size (≤20% distortion).

        Multi-line gives richer glyph coverage (ascenders, descenders, spacing,
        leading) and naturally produces a squarer crop that wastes less padding.
        """
        W = H = self.img_size
        MIN_CHARS = 50  # minimum total characters across all lines

        # --- Build a multi-line paragraph ---
        # Combine two pangrams/sentences to guarantee ≥50 chars and glyph diversity.
        base = random.choice(PANGRAMS) + " " + random.choice(SENTENCES)
        # Shuffle in a random word occasionally for variation
        if random.random() < 0.4:
            base = base + " " + random.choice(WORDS)

        # Font size: moderate — large enough to see glyph detail, small enough
        # to fit 2-3 lines on a reasonable canvas.
        font_size = random.randint(36, 72)
        font = self._get_pil_font(font_name, font_size)

        # Wrap width: target ~20-25 chars per line so we get 2-3 lines.
        # Measure a reference character width to estimate pixels per char.
        canvas_w = 900
        canvas_h = 600  # tall enough for several lines

        draw_tmp = ImageDraw.Draw(Image.new("RGB", (canvas_w, canvas_h)))

        def measure(t):
            try:
                b = draw_tmp.textbbox((0, 0), t, font=font)
                return b[2] - b[0], b[3] - b[1]
            except Exception:
                return font_size * len(t) // 2, font_size

        # Estimate pixels per character and set wrap width for ~12 chars/line
        # → 4-5 lines → blocky aspect ratio that fills the 224×224 image well
        char_w = measure("abcdefghijklmnopqrstuvwxyz")[0] / 26
        wrap_px = max(int(char_w * 12), 60)  # ~12 chars per line

        # Word-wrap into lines using the narrow wrap width
        words = base.split()
        lines = []
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            w, _ = measure(candidate)
            if w > wrap_px and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)

        # Add a second source string as another line if we have fewer than 2 lines
        # or if total chars is still low
        total_chars = sum(len(l) for l in lines)
        if len(lines) < 2 or total_chars < MIN_CHARS:
            extra = random.choice(PANGRAMS + SENTENCES)
            extra_words = extra.split()
            current2 = ""
            for word in extra_words:
                candidate = (current2 + " " + word).strip()
                w, _ = measure(candidate)
                if w > wrap_px and current2:
                    lines.append(current2)
                    current2 = word
                    break
                else:
                    current2 = candidate
            if current2:
                lines.append(current2)

        # Keep 2-6 lines max
        lines = lines[:6]

        # --- Measure total block size ---
        line_h = measure("Ag")[
            1
        ]  # representative height including ascenders/descenders
        line_spacing = int(line_h * 1.3)
        block_h = line_spacing * (len(lines) - 1) + line_h
        block_w = max(measure(l)[0] for l in lines)

        # --- Background and text colours (greyscale) ---
        dark_bg = random.random() < 0.3
        if dark_bg:
            bg_val = random.randint(10, 60)
            text_val = random.randint(180, 255)
        else:
            bg_val = random.randint(220, 255)
            text_val = random.randint(0, 60)
        bg_color = (bg_val, bg_val, bg_val)
        text_color = (text_val, text_val, text_val)

        # --- Render ---
        # Make canvas just big enough for the text block + padding
        pad = max(12, int(block_h * 0.15))
        c_w = block_w + 2 * pad
        c_h = block_h + 2 * pad
        canvas = Image.new("RGB", (c_w, c_h), bg_color)
        draw = ImageDraw.Draw(canvas)

        y_cursor = pad
        for line in lines:
            draw.text((pad, y_cursor), line, font=font, fill=text_color)
            y_cursor += line_spacing

        # --- Tight crop on actual pixels ---
        arr = np.array(canvas)
        if dark_bg:
            mask = arr.max(axis=2) > (bg_val + 20)
        else:
            mask = arr.min(axis=2) < (bg_val - 20)

        if mask.any():
            rows_idx = np.where(np.any(mask, axis=1))[0]
            cols_idx = np.where(np.any(mask, axis=0))[0]
            rmin, rmax = rows_idx[0], rows_idx[-1]
            cmin, cmax = cols_idx[0], cols_idx[-1]
            crop_pad = max(8, int((rmax - rmin) * 0.10))
            rmin = max(0, rmin - crop_pad)
            rmax = min(c_h - 1, rmax + crop_pad)
            cmin = max(0, cmin - crop_pad)
            cmax = min(c_w - 1, cmax + crop_pad)
            crop = canvas.crop((cmin, rmin, cmax + 1, rmax + 1))
        else:
            crop = canvas  # fallback

        # --- Letterbox resize to W×H (preserves aspect ratio, ≤20% distortion) ---
        cw, ch = crop.size
        scale = min(W / cw, H / ch)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        resized = crop.resize((new_w, new_h), Image.LANCZOS)
        img = Image.new("RGB", (W, H), bg_color)
        img.paste(resized, ((W - new_w) // 2, (H - new_h) // 2))

        # Light augmentation
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
