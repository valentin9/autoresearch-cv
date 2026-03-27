# Task Scope

## Description

<!-- Write a natural language description of the task here. Be specific:
     - What is the input image?
     - What are the classes?
     - What does each class represent?
     - Any known challenges (class imbalance, fine-grained differences, etc.)?
     - Dataset hints: can it be generated synthetically? Any known public sources?
-->

**Example task**: Train a model to detect and classify image overlays on photos.
Given an input image, the model should classify what type of overlay (if any) is present.
Overlays include: text overlays (subtitles, captions, watermarks), graphic overlays
(logos, banners, frames), and clean images with no overlay.

Classes:
- `clean` — no overlay
- `text_overlay` — text rendered onto the image (subtitles, captions, watermarks, memes)
- `graphic_overlay` — non-text graphic elements (logos, banners, frames, channel bugs)

This task can be addressed with synthetic data: take any background image, optionally
composite a text or graphic overlay onto it, and label accordingly. See `tools/` for
synthetic generation utilities.

---

## Task Metadata

```json
{
  "task_name": "image_overlay",
  "classes": ["clean", "graphic_overlay", "text_overlay"],
  "metric": "f1_macro"
}
```

---

## Dataset Generation Notes

<!-- Expand on how the dataset should be built. The agent will read this. -->

**Recommended approach**: Synthetic generation.

1. **Background images**: Download a diverse set of background photos. Good sources:
   - COCO images (unlicensed for research): `datasets` library, `"detection-datasets/coco"`
   - OpenImages: `datasets` library
   - Or download ~1000 misc web images via `tools/web_sources.py`

2. **Clean samples**: background images with no modification.

3. **Text overlay samples**: render text onto background images using PIL:
   - Random fonts from system fonts or Google Fonts
   - Random text content (lorem ipsum, news headlines, random strings)
   - Random position (corners, bottom third, center)
   - Random font size (12–72px), color (white, black, semi-transparent)
   - Optional: drop shadow, stroke, background box

4. **Graphic overlay samples**: composite a small PNG (logo, banner) onto background:
   - Random logos from a small set of placeholder/generic logos
   - Random position (corner bugs, top banners)
   - Random opacity (0.5–1.0)

5. **Class balance**: aim for ~equal numbers per class.

Target: 2000+ train images per class, 400+ val images per class.

See `tools/synthetic.py` and `tools/overlay.py` (the agent will create these if needed).
