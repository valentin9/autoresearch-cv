# autoresearch — Vision Edition

Autonomous vision research on a single RTX 4090. The agent (you) trains CNNs and Vision
Transformers from scratch for image classification tasks, continuously searching for novel
and efficient architectures.

---

## Task Scope

The task is defined in `scope.md`. Read it before doing anything else. It contains:
- A natural language description of the problem
- A JSON block with `task_name`, `classes`, and `metric`
- Dataset status, experiment plan, and known bugs/findings

---

## Repository Structure

```
autoresearch/
├── scope.md          # Task definition + dataset notes + experiment plan — READ THIS FIRST
├── prepare.py        # Fixed: evaluation harness, dataloader, constants. DO NOT MODIFY.
├── train.py          # Your file: model, training loop, hyperparams. Edit freely.
├── tools/            # Growing library of reusable dataset generation utilities
│   ├── synthetic.py  # Base classes for synthetic dataset generation
│   ├── font_generator.py  # Font classification dataset generator (current task)
│   ├── web_sources.py
│   └── augmentation.py
├── results.tsv       # Experiment log (not committed to git)
└── analysis.ipynb    # Charts
```

Cache is at `~/.cache/autoresearch/datasets/<task_name>/`.

---

## Setup (do this once per task)

1. **Read scope.md** — understand the task, dataset status, and experiment history.

2. **Check the dataset**: `uv run prepare.py --validate-only`
   - Dataset is already generated. Do NOT regenerate unless explicitly asked.

3. **Check the branch**: we are on `autoresearch/mar27-fonts`. Continue on this branch.

4. **Read `train.py`** for the current model and training loop state.

5. **Initialize results.tsv** if empty:
   ```
   commit	primary_metric	memory_gb	status	description
   ```

6. **Kick off the loop.**

---

## CRITICAL: Known MPS Training Bug (macOS Apple Silicon)

**Symptom**: model trains (loss decreases to ~1.2) but `evaluate()` always returns
`val_f1_macro=0.0278`, `val_accuracy=0.125` — i.e. random/collapsed predictions.

**Root cause**: NOT fully understood yet. Ruled out:
- Time-based LR warmup (fixed: now using step-count warmup)
- Grad accumulation (fixed: removed, using bs=64 direct)
- Dataset issues (fixed: see scope.md)
- Class label mismatch (verified: train and val use identical `class_to_idx`)
- Nested autocast (tested: doesn't affect result)
- `gc.freeze()/gc.disable()` (tested: doesn't affect result)
- BatchNorm train vs eval mode (verified: eval mode works correctly in isolation)

**What we know**:
- Direct test (import model, train 300 steps in a plain script): val_f1=0.52, val_acc=0.87 ✓
- Full `train.py` run (same model, same data, same hyperparams): val_f1=0.028 ✗
- The model's loss DOES decrease to ~1.2 during training (not a gradient/LR issue)
- At checkpoint evals, the model predicts class 0 for almost everything

**Suspected cause**: something in the `train.py` execution environment (module-level
side effects, multiprocessing state, MPS memory state) causes a subtle corruption after
many training steps. The bug does not manifest in a fresh Python process with <400 steps.

**Workaround to try in a fresh session**:
1. Use `torch.backends.mps.is_available()` check and fall back to CPU if MPS causes issues
2. Try running `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run train.py` (disables MPS caching)
3. Try moving `import timm` and model creation inside `if __name__ == '__main__':` guard
4. Try disabling bfloat16 autocast on MPS (use fp32 instead)
5. If on CUDA (RTX 4090), this bug likely does not apply — just run normally

**On RTX 4090 (the intended hardware)**: the MPS bug is irrelevant. The training loop
in `train.py` is otherwise correct — just run it.

---

## Phase 1: Dataset Construction

**DATASET ALREADY EXISTS** — skip to Phase 2. See scope.md for full details.

If for any reason you need to regenerate:
```bash
uv run python3 -c "
import sys, json, re
from pathlib import Path
sys.path.insert(0, '.')
from tools.font_generator import FontDatasetGenerator
scope = json.loads(re.search(r'\`\`\`json\s*(\{.*?\})\s*\`\`\`', Path('scope.md').read_text(), re.DOTALL).group(1))
output_dir = Path.home() / '.cache/autoresearch/datasets' / scope['task_name']
gen = FontDatasetGenerator(output_dir=str(output_dir), classes=scope['classes'], train_per_class=2000, val_per_class=400)
gen.generate(overwrite=True)
"
```

---

## Phase 2: Model Experimentation

Each experiment runs for a **fixed time budget of 20 minutes** (1200s wall-clock
training time). Launch with:

```
uv run train.py > run.log 2>&1
```

### What you CAN do

Edit `train.py` freely:
- Model architecture (the most important dimension — see guidance below)
- Optimizer, learning rate, schedule, augmentation
- Batch size, gradient accumulation, precision
- Loss function, regularization

### What you CANNOT do

- Modify `prepare.py` — it is read-only.
- Change the interface: `model(images)` must accept `(B, 3, H, W)` and return `(B, num_classes)`.
- Install new packages. Available: torch, torchvision, timm, scikit-learn, Pillow, requests,
  numpy, pandas, matplotlib, datasets.

### Training loop requirements (non-negotiable after debugging)

```python
# 1. Fix macOS multiprocessing
import prepare as _prepare, platform
if platform.system() == "Darwin":
    _prepare.NUM_WORKERS = 0

# 2. Step-count LR warmup (NOT time-based)
if step < WARMUP_STEPS:
    lr = BASE_LR * (step + 1) / WARMUP_STEPS
else:
    t = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

# 3. No grad accumulation (use bs=64 directly on MPS; bs can be larger on CUDA)
# 4. No gc.freeze() / gc.disable()
# 5. Checkpoint eval fires no earlier than step 300
# 6. Call evaluate() WITHOUT wrapping in outer autocast context
results = evaluate(model, TASK_NAME, DEVICE_BATCH_SIZE, metric=METRIC)  # correct
with autocast_ctx:                                                        # WRONG
    results = evaluate(...)
```

### Experiment plan (from scope.md)

1. **Pretrained ResNet-18** (diagnostic ceiling) — expected val_f1 > 0.65
2. **ConvNeXt-Micro from scratch** (the baseline model already in train.py)
3. **Novel architecture search** — see directions below

### Architecture research — the main goal

Primary direction: **novel, efficient architectures from scratch**.

Interesting directions:
- Hybrid CNN/attention designs (local conv + global attention)
- Depthwise/grouped convolutions, inverted bottlenecks (ConvNeXt-style)
- Token mixing alternatives: Fourier, MLP-Mixer-style
- Unusual skip connection patterns (dense connections, cross-stage shortcuts)
- Lightweight attention (linear attention, axial)
- For font detection specifically: architectures that emphasise fine local texture
  (stroke width, serif shape) — e.g. multi-scale feature extraction, high-frequency
  preserving stems

Pretrained weights are allowed as a comparison baseline but primary focus is from-scratch.

---

## Experiment Loop

LOOP FOREVER (once setup is complete):

1. **Review state**: check `results.tsv` and current `train.py`.
2. **Formulate a hypothesis**: what change might help?
3. **Edit `train.py`**, commit: `git add train.py && git commit -m "experiment: <desc>"`
4. **Run**: `uv run train.py > run.log 2>&1`
5. **Parse results**: `grep "^val_f1_macro:\|^peak_vram_mb:" run.log`
6. **Log to results.tsv**: `<commit>\t<metric>\t<memory_gb>\t<status>\t<description>`
7. **Keep or revert**: improved → keep; same/worse → `git reset --hard HEAD~1`

**NEVER STOP** until the human interrupts.

---

## Output Format

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

---

## results.tsv Format

```
commit	primary_metric	memory_gb	status	description
a1b2c3d	0.847231	8.2	keep	baseline ConvNeXt-Micro from scratch
b2c3d4e	0.861000	9.1	keep	add stochastic depth 0.2
c3d4e5f	0.843000	8.2	discard	switch to MLP-Mixer (worse)
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
