# autoresearch — Vision Edition

Autonomous vision research on a single RTX 4090. The agent (you) trains CNNs and Vision
Transformers from scratch for image classification tasks, continuously searching for novel
and efficient architectures.

---

## Task Scope

The task is defined in `scope.md`. Read it before doing anything else. It contains:
- A natural language description of the problem
- A JSON block with `task_name`, `classes`, and `metric`
- Dataset hints (how to generate or source the data)

---

## Repository Structure

```
autoresearch/
├── scope.md          # Task definition — YOU read this, human writes it
├── prepare.py        # Fixed: evaluation harness, dataloader, constants. DO NOT MODIFY.
├── train.py          # Your file: model, training loop, hyperparams. Edit freely.
├── tools/            # Growing library of reusable dataset generation utilities
│   ├── synthetic.py  # Base classes for synthetic dataset generation
│   ├── web_sources.py
│   └── augmentation.py
├── datasets/         # Generated datasets land here (not committed to git)
├── results.tsv       # Experiment log (not committed to git)
└── analysis.ipynb    # Charts
```

Cache is at `~/.cache/autoresearch/datasets/<task_name>/`.

---

## Setup (do this once per task)

Work with the user to complete these steps:

1. **Read scope.md** — understand the task fully before doing anything.

2. **Agree on a run tag**: propose a tag like `mar27-overlay`. Branch `autoresearch/<tag>`
   must not already exist.

3. **Create the branch**: `git checkout -b autoresearch/<tag>`

4. **Read the in-scope files** for full context:
   - `scope.md` — the task
   - `prepare.py` — fixed eval harness and dataloader API (do not modify)
   - `train.py` — your starting model and training loop
   - `tools/` — the reusable utility library

5. **Check or build the dataset**:
   - Run `uv run prepare.py --validate-only` to check if dataset exists.
   - If the dataset does NOT exist, enter **Phase 1: Dataset Construction** (see below).
   - If the dataset DOES exist, skip to **Phase 2: Model Experimentation**.

6. **Initialize results.tsv** with just the header row:
   ```
   commit	primary_metric	memory_gb	status	description
   ```

7. **Confirm with user and kick off the loop.**

---

## Phase 1: Dataset Construction

Your goal is to produce a dataset at `~/.cache/autoresearch/datasets/<task_name>/` in
ImageFolder format:

```
train/
  class_a/  img1.jpg  img2.jpg ...
  class_b/  ...
val/
  class_a/  ...
  class_b/  ...
```

**Minimum viable dataset**: 500+ images per class for train, 100+ for val. Aim for
balanced classes. More is better, but don't spend more than 30 minutes on data prep.

**Strategy** (in order of preference):

1. **Synthetic generation** — if the task allows it (fonts, overlays, etc.), generate
   images programmatically. This is the best approach: unlimited data, full control.
   Write generation code in `tools/` as a reusable module.

2. **Existing dataset** — search HuggingFace datasets (`datasets` library) for a
   relevant dataset. Example: `load_dataset("imagenet-1k", ...)`. Use `tools/web_sources.py`.

3. **Web scraping** — download images via search APIs (Google, Bing, DuckDuckGo) or
   known repositories. Always prefer licenses that allow research use.

**Reusability rule**: Any dataset generation code you write MUST be placed in `tools/`
as a reusable module, not as a one-off script. The module should be usable by future
tasks with minimal configuration. Commit new tools to git with a clear message.

Example: if you write an overlay generator, put it in `tools/overlay.py` with a clean
API. Future tasks involving overlays can import and extend it.

**When you are done building the dataset**:
- Run `uv run prepare.py --validate-only` and confirm output looks healthy.
- Check class balance — if a class has <100 train images, flag it but continue.
- Commit any new tools you created.

---

## Phase 2: Model Experimentation

Each experiment runs for a **fixed time budget of 20 minutes** (1200s wall-clock
training time, excluding startup/compilation). Launch with:

```
uv run train.py > run.log 2>&1
```

### What you CAN do

Edit `train.py` freely:
- Model architecture (the most important dimension — see guidance below)
- Optimizer, learning rate, schedule, augmentation
- Batch size, gradient accumulation, precision
- Loss function, regularization
- Anything that runs in Python + the installed packages

### What you CANNOT do

- Modify `prepare.py` — it is read-only. It contains the fixed eval, dataloader, and
  constants. The `evaluate()` function is the ground truth metric.
- Change the interface contract: `model(images)` must accept `(B, 3, H, W)` float
  tensors and return `(B, num_classes)` logits.
- Install new packages. You can only use what's in `pyproject.toml`:
  torch, torchvision, timm, scikit-learn, Pillow, requests, numpy, pandas, matplotlib,
  datasets.

### Architecture research — the main goal

The primary research direction is **novel, efficient architectures from scratch**. This
means: do not start from pretrained weights. Discover what architectural choices lead to
the best accuracy per parameter, or best accuracy per training minute on a 4090.

Interesting directions to explore:
- Hybrid CNN/attention designs (local conv + global attention)
- Extremely deep vs. extremely wide models under the 24GB VRAM budget
- Non-standard activation functions, normalization strategies
- Depthwise/grouped convolutions, inverted bottlenecks
- Token mixing alternatives: Fourier, MLP-Mixer-style, state-space models
- Unusual skip connection patterns (dense connections, cross-stage shortcuts)
- Adaptive computation: different depths per image region
- Lightweight attention (linear attention, sliding window, axial)

Pretrained weights (`timm` models) are allowed as a comparison baseline, but the
primary focus is from-scratch novel designs. If you use pretrained weights, note it
clearly in results.tsv and treat it as a separate branch of experiments.

**Simplicity criterion**: All else equal, simpler is better. A 0.001 improvement that
adds 50 lines of hacky code is not worth it. Equal performance with less code is a win.

### The VRAM budget

RTX 4090 has 24 GB. Be mindful:
- If a run OOMs, reduce `DEVICE_BATCH_SIZE` first, then reduce model size.
- `peak_vram_mb` is reported — watch it.
- You cannot change the eval in `prepare.py` but you can reduce `DEVICE_BATCH_SIZE`
  used during training.

---

## Experiment Loop

LOOP FOREVER (once setup is complete):

1. **Review state**: check current branch, last results in `results.tsv`.
2. **Formulate a hypothesis**: what architectural or training change might help?
   Base this on previous results, research intuition, or first principles.
3. **Edit `train.py`** with the experimental change.
4. **Commit**: `git add train.py && git commit -m "experiment: <brief description>"`
5. **Run**: `uv run train.py > run.log 2>&1`
   (This takes ~20 minutes. While waiting you can think about the next experiment.)
6. **Parse results**:
   ```
   grep "^val_\|^peak_vram" run.log
   ```
   If empty → crash. Run `tail -n 60 run.log` to read the traceback.
7. **Log to results.tsv** (tab-separated, do not commit this file):
   ```
   <commit>  <primary_metric>  <memory_gb>  <status>  <description>
   ```
   - `primary_metric`: the value of `val_<METRIC>` from the log (e.g. `val_f1_macro`)
   - `memory_gb`: `peak_vram_mb / 1024`, rounded to 1 decimal
   - `status`: `keep`, `discard`, or `crash`
8. **Keep or revert**:
   - If metric improved → keep the commit, advance.
   - If metric equal or worse → `git reset --hard HEAD~1`, move on.

**Crashes**: if it's a trivial bug (typo, missing import), fix and re-run. If the idea
is fundamentally broken, log `crash`, revert, and try something else.

**Timeout**: if a run exceeds 30 minutes, kill it and treat as a crash.

**Going back to Phase 1**: if you believe the dataset quality is the main bottleneck
(e.g., noisy labels, too few samples, class imbalance), you may spend time improving
the dataset. Add more samples, fix labels, balance classes. Commit new tools.

**NEVER STOP**: once the loop has started, do NOT pause to ask the user if you should
continue. Do NOT ask "should I keep going?". The user expects you to run indefinitely
until manually interrupted. If you run out of obvious ideas, think harder: re-read the
results, combine near-misses, try more radical changes, explore entirely different
architecture families. The loop runs until the human stops it.

---

## Output Format

The training script prints a summary at the end. Example for an overlay detection task:

```
---
val_f1_macro:        0.847231
val_accuracy:        0.863000
val_f1_macro:        0.847231
val_f1_weighted:     0.859412
val_samples:         3200
training_seconds:    1200.1
total_seconds:       1284.3
peak_vram_mb:        8431.2
total_images_k:      412.3
num_steps:           1604
num_params_m:        12.34
```

Key metric extraction:
```
grep "^val_f1_macro:\|^peak_vram_mb:" run.log
```
(Replace `f1_macro` with whatever METRIC is set to in scope.md.)

---

## results.tsv Format

Header + tab-separated rows. **Do not use commas** in descriptions.

```
commit	primary_metric	memory_gb	status	description
a1b2c3d	0.847231	8.2	keep	baseline ConvNeXt-Micro from scratch
b2c3d4e	0.861000	9.1	keep	add stochastic depth 0.2
c3d4e5f	0.843000	8.2	discard	switch to MLP-Mixer (worse)
d4e5f6g	0.000000	0.0	crash	ViT-Tiny OOM at bs=64
```

---

## Installed Packages

```
torch==2.9.1 (CUDA 12.8)
torchvision
timm
scikit-learn
Pillow
requests
numpy
pandas
matplotlib
datasets (HuggingFace)
```
