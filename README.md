# autoresearch — Vision Edition

*A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for autonomous vision research on a single RTX 4090.*

The idea is the same: give an AI agent a real training setup and let it experiment overnight. You define the task, it builds the dataset, then continuously searches for novel and efficient image classification architectures — modifying code, training for 20 minutes, checking if results improved, keeping or discarding, and repeating indefinitely. You wake up to a log of experiments.

Instead of LLMs and bits-per-byte, this version trains **CNNs and Vision Transformers from scratch** for image classification tasks like overlay detection, font recognition, or anything you define in `scope.md`.

---

## How it works

Four files that matter:

- **`scope.md`** — you fill this in. Describes the task, classes, metric, and dataset hints. **Edited by the human.**
- **`prepare.py`** — fixed evaluation harness and dataloader. Read-only. Contains `evaluate()` (the ground truth metric) and `make_dataloader()`.
- **`train.py`** — the agent's playground. Model architecture, optimizer, training loop, hyperparameters. **Edited by the agent.**
- **`program.md`** — research protocol for the AI agent. Defines the two-phase loop (dataset construction → model experiments). **Read by the agent, occasionally edited by the human.**

The `tools/` directory is a **growing library of reusable dataset generation utilities** the agent builds over time:

```
tools/
  synthetic.py    — base class for synthetic dataset generators
  web_sources.py  — download images from the web and HuggingFace
  augmentation.py — reusable augmentation pipelines
  overlay.py      — synthetic overlay dataset generator (example)
```

### The experiment loop

Each training run has a **fixed 20-minute time budget**. The agent:

1. Checks or builds the dataset (Phase 1)
2. Edits `train.py` with an architectural hypothesis
3. `git commit` → `uv run train.py > run.log 2>&1`
4. Parses results, logs to `results.tsv`
5. Keeps the commit if the metric improved, reverts if not
6. Repeats forever (~3 experiments/hour, ~50 overnight)

The primary research direction is **novel architectures from scratch** — not fine-tuning pretrained models. The fixed time constraint becomes a forcing function for discovering efficient designs.

---

## Requirements

- **NVIDIA RTX 4090** (24 GB VRAM, CUDA 12.x) — or any NVIDIA GPU with CUDA 12.x
- **Apple Silicon M-series** (MPS, macOS 13+) — M3 Pro 32 GB works well
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

Device is auto-detected at startup: CUDA → MPS → CPU. No code changes needed when switching machines.

---

## Quick start

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
#    On macOS: installs the standard PyPI torch with MPS support built in
#    On Linux/Windows with NVIDIA GPU: installs the CUDA 12.8 wheel
uv sync
```

---

## Running your first task

### Step 1 — Define the task in `scope.md`

Edit `scope.md` to describe your task. The file must contain a `task_name`, `classes`, and `metric` in a JSON block. An overlay detection example is pre-filled:

```json
{
  "task_name": "image_overlay",
  "classes": ["clean", "graphic_overlay", "text_overlay"],
  "metric": "f1_macro"
}
```

Change this to whatever you want to classify. Examples:
- Font recognition: `"task_name": "font_recognition"`, classes are font names
- Watermark detection: `"task_name": "watermark"`, classes `["clean", "watermark"]`

### Step 2 — Build the dataset

Datasets live at `~/.cache/autoresearch/datasets/<task_name>/` in ImageFolder format:
```
train/
  class_a/  img001.jpg  img002.jpg ...
  class_b/  ...
val/
  class_a/  ...
  class_b/  ...
```

**Option A — Use the pre-built overlay generator** (for the example task):

```bash
# Download ~1000 background images from COCO via HuggingFace
python -c "
from tools.web_sources import download_coco_subset
download_coco_subset('~/.cache/autoresearch/datasets/coco_backgrounds', num_images=1000)
"

# Generate the overlay dataset (synthetic compositing)
python tools/overlay.py \
    --backgrounds ~/.cache/autoresearch/datasets/coco_backgrounds \
    --output ~/.cache/autoresearch/datasets/image_overlay \
    --train 2000 \
    --val 400
```

**Option B — Use a HuggingFace dataset** (for any classification task with existing data):

```python
from tools.web_sources import fetch_hf_images
fetch_hf_images(
    "food101",
    "~/.cache/autoresearch/datasets/food101",
    split="train",
    max_per_class=500,
)
```

**Option C — Let the agent build it.** If you skip this step, the agent will attempt to build the dataset itself during Phase 1 of the experiment loop, writing reusable code to `tools/` as it goes.

### Step 3 — Validate the dataset

```bash
uv run prepare.py --validate-only
```

You should see image counts per class for both `train` and `val` splits.

### Step 4 — Test a single training run

```bash
uv run train.py
```

This trains for 20 minutes and prints a summary:

```
---
val_f1_macro:        0.847231
val_accuracy:        0.863000
val_f1_weighted:     0.859412
val_samples:         3200
training_seconds:    1200.1
total_seconds:       1284.3
peak_vram_mb:        8431.2
total_images_k:      412.3
num_steps:           1604
num_params_m:        12.34
```

### Step 5 — Start the agent

Open OpenCode (or Claude Code, Codex, etc.) in this repo and say:

```
Read program.md and let's kick off a new experiment — do the setup first.
```

The agent will:
1. Propose a branch name and create it
2. Read the in-scope files
3. Check or build the dataset
4. Establish a baseline run
5. Loop forever, searching for better architectures

---

## Project structure

```
scope.md        — task definition (you write this)
prepare.py      — fixed eval harness and dataloader (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent research protocol
pyproject.toml  — dependencies
tools/
  synthetic.py  — base class for synthetic dataset generators
  web_sources.py — web/HuggingFace image downloading
  augmentation.py — reusable augmentation pipelines
  overlay.py    — overlay detection dataset generator (example)
analysis.ipynb  — experiment result charts
results.tsv     — experiment log (untracked, agent writes this)
```

---

## Platform performance

| Hardware | Device | Batch size | Approx imgs/sec | Notes |
|---|---|---|---|---|
| RTX 4090 24 GB | `cuda` | 64 | ~3000 | Full speed, `torch.compile` active |
| M3 Pro 32 GB | `mps` | 32–64 | ~300–600 | No `torch.compile` (MPS limitation), ~5-10× slower |
| M1/M2 16 GB | `mps` | 16–32 | ~150–300 | Reduce `DEVICE_BATCH_SIZE` if OOM |
| CPU | `cpu` | 8–16 | ~30–80 | Last resort only |

On an M3 Pro the 20-minute budget still trains a reasonable model — you just get fewer gradient steps. If you want more experiments per hour on Mac, reduce `TIME_BUDGET` in `prepare.py` to e.g. 600 (10 min) and `DEVICE_BATCH_SIZE` in `train.py` to 32.

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Diffs are reviewable, git history is the experiment log.
- **Fixed 20-minute budget.** All experiments are directly comparable regardless of model size, batch size, or architecture. The agent finds the best model for your hardware in that time.
- **Novel architectures from scratch.** The research focus is discovering efficient designs, not fine-tuning. Pretrained models are allowed as a comparison baseline but not the goal.
- **Growing tools library.** Any dataset generation code the agent writes goes into `tools/` as a reusable module. The repo accumulates useful building blocks over time.
- **Task-defined metric.** `scope.md` specifies whether to optimize `accuracy`, `f1_macro`, or `f1_weighted`. The agent always optimizes the metric you care about.

---

## Defining a new task

1. Edit `scope.md` — fill in `task_name`, `classes`, `metric`, and a description of how to build the dataset.
2. Either generate the dataset yourself (see Step 2 above), or let the agent do it.
3. Start the agent on a new branch.

The agent will write any new dataset generation tools to `tools/<task_name>.py` and commit them so they're reusable in future tasks.

---

## License

MIT
