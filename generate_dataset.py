"""
Generate (or regenerate) the font detection dataset from scope.md.

Usage:
    uv run generate_dataset.py
    uv run generate_dataset.py --validate-only
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools.font_generator import FontDatasetGenerator
from prepare import validate_dataset


def load_scope():
    content = (Path(__file__).parent / "scope.md").read_text()
    m = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not m:
        raise ValueError("Could not find JSON block in scope.md")
    return json.loads(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate, don't generate"
    )
    parser.add_argument(
        "--no-overwrite", action="store_true", help="Skip classes that already exist"
    )
    args = parser.parse_args()

    scope = load_scope()
    task_name = scope["task_name"]
    classes = scope["classes"]
    output_dir = Path.home() / ".cache" / "autoresearch" / "datasets" / task_name

    print(f"Task:       {task_name}")
    print(f"Classes:    {len(classes)}")
    print(f"Output dir: {output_dir}")
    print()

    if args.validate_only:
        validate_dataset(task_name)
        return

    gen = FontDatasetGenerator(
        output_dir=str(output_dir),
        classes=classes,
        train_per_class=2000,
        val_per_class=400,
    )
    gen.generate(overwrite=not args.no_overwrite)

    print("\nValidating...")
    validate_dataset(task_name)


if __name__ == "__main__":
    main()
