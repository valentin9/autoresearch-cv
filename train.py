"""
Autoresearch vision training script. Single-GPU, single-file.
Supports: CUDA (RTX 4090), MPS (Apple Silicon M-series), CPU.

Usage:
    uv run train.py

The task is defined by scope.md. Dataset must exist before running.
See prepare.py and program.md for the full protocol.
"""

import os

if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "" and True:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import copy
import math
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from prepare import (
    IMG_SIZE,
    TIME_BUDGET,
    SCOPE_FILE,
    load_scope,
    get_num_classes,
    get_class_names,
    make_dataloader,
    evaluate,
)
import prepare as _prepare

# On macOS, DataLoader workers cause re-execution of train.py via spawn.
# Force single-threaded loading.
import platform

if platform.system() == "Darwin":
    _prepare.NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

scope = load_scope()
TASK_NAME = scope["task_name"]
METRIC = scope["metric"]
NUM_CLASSES = get_num_classes(TASK_NAME)
CLASS_NAMES = get_class_names(TASK_NAME)

print(f"Task:        {TASK_NAME}")
print(f"Classes:     {CLASS_NAMES}")
print(f"Num classes: {NUM_CLASSES}")
print(f"Metric:      {METRIC}")
print(f"Time budget: {TIME_BUDGET}s")
print()

# ---------------------------------------------------------------------------
# Experiment 1 (clean restart): Pretrained ResNet-18 — pipeline ceiling
#
# Dataset: fixed (BW, multi-line, letterbox, correct fonts, ≥50 chars).
# Training fixes applied:
#   - Step-count LR warmup (not time-based)
#   - No grad accumulation (bs=64 direct, matches validated direct test)
#   - No gc.freeze/disable
#   - Best-checkpoint eval fires after step 300 minimum
#   - evaluate() called without extra outer autocast wrapper
#
# 300-step direct test on this dataset: val_f1=0.52, val_acc=0.87.
# Expected after full 20-min run: val_f1 > 0.65.
# ---------------------------------------------------------------------------


class PretrainedClassifier(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=15, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEVICE_BATCH_SIZE = 64
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 50
LABEL_SMOOTHING = 0.1
EVAL_FIRST_STEP = 300  # don't checkpoint before this step
EVAL_EVERY_STEPS = 300  # checkpoint every N steps after that

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(42)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

_autocast_dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=_autocast_dtype)

model = PretrainedClassifier(
    model_name="resnet18", num_classes=NUM_CLASSES, pretrained=True
).to(device)
print(f"Model params: {model.num_params() / 1e6:.2f}M")

if device.type != "mps":
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(
    TASK_NAME, DEVICE_BATCH_SIZE, "train", pin_memory=(device.type == "cuda")
)
images_per_epoch = len(train_loader) * DEVICE_BATCH_SIZE

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=BASE_LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8,
    fused=(device.type == "cuda"),
)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Estimate total steps for cosine schedule (device-specific throughput)
# MPS updated: ~300 imgs/s at bs=64 => ~4.7 steps/sec on M-series
_steps_per_sec = {"cuda": 100.0, "mps": 4.7, "cpu": 0.5}
TOTAL_STEPS = max(1000, int(_steps_per_sec.get(device.type, 1.0) * TIME_BUDGET))

print(f"Estimated total steps: {TOTAL_STEPS}")
print(f"Batch size:            {DEVICE_BATCH_SIZE}")
print(f"Images per epoch:      {images_per_epoch:,}")
print()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def device_synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()


def peak_memory_mb():
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

model.train()
train_iter = iter(train_loader)
t_start_training = time.time()
total_training_time = 0.0
step = 0
total_images = 0
smooth_loss = 0.0
best_metric = -1.0
best_state = None
last_eval_step = -EVAL_EVERY_STEPS  # so first eval fires at step EVAL_FIRST_STEP

while True:
    device_synchronize()
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    try:
        images, labels = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        images, labels = next(train_iter)

    _nb = (
        device.type == "cuda"
    )  # non_blocking only safe on CUDA; causes data races on MPS
    images = images.to(device, non_blocking=_nb)
    labels = labels.to(device, non_blocking=_nb)

    with autocast_ctx:
        logits = model(images)
        loss = criterion(logits, labels)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step-count LR: linear warmup → cosine decay
    if step < WARMUP_STEPS:
        lr = BASE_LR * (step + 1) / WARMUP_STEPS
    else:
        t = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.step()

    device_synchronize()
    dt = time.time() - t0
    total_training_time += dt
    total_images += DEVICE_BATCH_SIZE

    ema_beta = 0.95
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss.item()
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))

    print(
        f"\rstep {step:05d} | loss: {debiased:.4f} | lr: {lr:.2e} | "
        f"imgs/s: {int(DEVICE_BATCH_SIZE / dt):,} | remaining: {max(0, TIME_BUDGET - total_training_time):.0f}s    ",
        end="",
        flush=True,
    )

    # Periodic best-checkpoint eval (not before EVAL_FIRST_STEP)
    if step >= EVAL_FIRST_STEP and (step - last_eval_step) >= EVAL_EVERY_STEPS:
        print()
        print(f"  [eval @ step {step}, t={total_training_time:.0f}s]", flush=True)
        model.eval()
        ckpt_results = evaluate(model, TASK_NAME, DEVICE_BATCH_SIZE, metric=METRIC)
        m = ckpt_results["primary_metric"]
        print(f"  {METRIC}={m:.4f}  acc={ckpt_results['accuracy']:.4f}", flush=True)
        if m > best_metric:
            best_metric = m
            best_state = copy.deepcopy(
                (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()
            )
            print(f"  *** new best: {best_metric:.4f} ***", flush=True)
        model.train()
        last_eval_step = step

    step += 1
    if total_training_time >= TIME_BUDGET:
        break

print()

# Restore best checkpoint
if best_state is not None:
    print(f"Restoring best checkpoint (metric={best_metric:.4f})")
    _m = model._orig_mod if hasattr(model, "_orig_mod") else model
    _m.load_state_dict(best_state)

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
results = evaluate(model, TASK_NAME, DEVICE_BATCH_SIZE, metric=METRIC)

t_end = time.time()
peak_vram = peak_memory_mb()
primary = results["primary_metric"]

print("---")
print(f"val_{METRIC}:        {primary:.6f}")
print(f"val_accuracy:        {results['accuracy']:.6f}")
print(f"val_f1_macro:        {results['f1_macro']:.6f}")
print(f"val_f1_weighted:     {results['f1_weighted']:.6f}")
print(f"val_samples:         {results['num_samples']}")
print(f"training_seconds:    {total_training_time:.1f}")
print(f"total_seconds:       {t_end - t_start:.1f}")
print(f"peak_vram_mb:        {peak_vram:.1f}")
print(f"total_images_k:      {total_images / 1000:.1f}")
print(f"num_steps:           {step}")
_m = model._orig_mod if hasattr(model, "_orig_mod") else model
print(f"num_params_m:        {_m.num_params() / 1e6:.2f}")
