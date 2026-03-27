# Task Scope

## Description

Train a model to detect and classify the font used in a text region of an image.
Given an input image containing rendered text, the model should identify which font
(by name) was used to render that text.

The input is a cropped image patch containing text rendered in a single font.
The model outputs a font class label corresponding to the font name.

Classes: one class per font name (e.g. `Arial`, `Times New Roman`, `Roboto`, `Georgia`, etc.).
The exact class list is defined in the metadata below.

**Research goal**: Find a novel approach that achieves strong font classification with limited
training data. Sample efficiency is a core research objective — the approach should not rely
on scale to succeed, and demonstrating good performance at 2k samples/class is part of what
makes it novel.

Known challenges:
- Fine-grained visual similarity between fonts (e.g. serif vs. serif variants)
- Variation in font size, color, background, and rendering quality
- Some fonts differ only subtly (e.g. `Helvetica` vs. `Arial`)

This task is well-suited for synthetic data: render text strings using known fonts onto
varied backgrounds, then label by font name. See `tools/` for synthetic generation utilities.

---

## Task Metadata

```json
{
  "task_name": "font_detection",
  "classes": [
    "Arial",
    "Times New Roman",
    "Georgia",
    "Roboto",
    "Open Sans",
    "Lato",
    "Montserrat",
    "Oswald",
    "Raleway",
    "Merriweather",
    "Playfair Display",
    "Source Sans Pro",
    "PT Sans",
    "Nunito",
    "Ubuntu"
  ],
  "metric": "f1_macro"
}
```

---

## Dataset Generation Notes

**Recommended approach**: Fully synthetic generation.

1. **Font sources**: Download target fonts from Google Fonts or use system-installed fonts.
   - Google Fonts API or direct download: https://fonts.google.com
   - Ensure all fonts in the class list are available as `.ttf` or `.otf` files

2. **Text content**: Render varied text strings to expose different character shapes:
   - Pangrams (e.g. "The quick brown fox jumps over the lazy dog")
   - Random words and short sentences
   - Single words and individual characters
   - Mix of upper/lowercase

3. **Rendering variation** (use PIL/Pillow):
   - Font size: 16–96px
   - Text color: random (dark on light, light on dark)
   - Background: solid colors, gradients, or cropped natural image patches
   - Optional: slight rotation (±3°), JPEG compression artifacts, blur

4. **Image format**: crop tightly around the text region, with small random padding.
   Output size: normalize to a fixed height (e.g. 64px) preserving aspect ratio, or
   use a fixed square crop (e.g. 224×224).

5. **Class balance**: generate equal numbers of samples per font class.

Target: 2000+ train images per class, 400+ val images per class.

See `tools/synthetic.py` (the agent will create this if needed).
