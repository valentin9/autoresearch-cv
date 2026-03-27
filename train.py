"""
Autoresearch vision training script. Single-GPU, single-file.
Supports: CUDA (RTX 4090), MPS (Apple Silicon M-series), CPU.

Usage:
    uv run train.py

The task is defined by scope.md. Dataset must exist before running.
See prepare.py and program.md for the full protocol.
"""

import os
import copy

# Only set CUDA allocator config when CUDA is actually available
if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "" and True:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import time
from dataclasses import dataclass, asdict

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

# On macOS (spawn-based multiprocessing), DataLoader workers cause re-execution of
# train.py which crashes. Force single-threaded loading on macOS.
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
# Experiment 8: ResNet-18 pretrained, no grad accum, constant LR, direct convergence
#
# Root cause debugging: the step-count LR warmup with grad_accum=2 causes slow
# convergence compared to direct training. Key findings:
# - Direct test (bs=64, lr=3e-4 constant): 300 steps → val_f1=0.58
# - train.py (bs=128 eff, warmup 100 steps): still at 0.028 after full 1200s
#
# Fix: remove grad accumulation, use bs=64 directly, use constant LR from step 1
# (small warmup over 50 steps). This matches the working direct test setup.
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
TOTAL_BATCH_SIZE = 64  # NO grad accumulation — matches working direct test
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 50  # short warmup
LABEL_SMOOTHING = 0.1

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
    model_name="resnet18",
    num_classes=NUM_CLASSES,
    pretrained=True,
).to(device)
print(f"Model params: {model.num_params() / 1e6:.2f}M")

if device.type != "mps":
    model = torch.compile(model, dynamic=False)

assert TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=BASE_LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8,
    fused=(device.type == "cuda"),
)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# ---------------------------------------------------------------------------
# Training utilities
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


# Estimate total steps based on device capability
_device_steps_est = {
    "cuda": 100.0,  # RTX 4090 with bs=64
    "mps": 3.0,  # Apple M-series with bs=64 (no grad accum)
    "cpu": 0.5,
}
TOTAL_STEPS = max(1000, int(_device_steps_est.get(device.type, 1.0) * TIME_BUDGET))
EVAL_EVERY_STEPS = max(100, TOTAL_STEPS // 5)

model.train()
train_loader = make_dataloader(
    TASK_NAME, DEVICE_BATCH_SIZE, "train", pin_memory=(device.type == "cuda")
)
images_per_epoch = len(train_loader) * DEVICE_BATCH_SIZE
train_iter = iter(train_loader)

print(f"Estimated total steps: {TOTAL_STEPS}")
print(f"Grad accum steps:    {grad_accum_steps}")
print(f"Images per epoch:    {images_per_epoch:,}")
print(f"Total batch size:    {TOTAL_BATCH_SIZE}")
print()

# Reset optimizer state and model for clean training
optimizer.zero_grad(set_to_none=True)
train_iter = iter(train_loader)

t_start_training = time.time()
total_training_time = 0.0
step = 0
total_images = 0
smooth_loss = 0.0

best_metric = -1.0
best_state = None
last_eval_step = 0
EVAL_EVERY_STEPS = max(50, TOTAL_STEPS // 5)  # eval ~5 times during training

while True:
    device_synchronize()
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)
    batch_loss = 0.0

    for micro_step in range(grad_accum_steps):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(images)
            loss = criterion(logits, labels) / grad_accum_steps

        loss.backward()
        batch_loss += loss.item()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step-count based LR schedule: linear warmup -> cosine decay
    if step < WARMUP_STEPS:
        lr = BASE_LR * (step + 1) / WARMUP_STEPS
    else:
        t = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.step()

    device_synchronize()
    t1 = time.time()
    dt = t1 - t0
    total_training_time += dt
    total_images += TOTAL_BATCH_SIZE

    ema_beta = 0.95
    smooth_loss = (
        ema_beta * smooth_loss + (1 - ema_beta) * batch_loss * grad_accum_steps
    )
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * min(step / max(1, TOTAL_STEPS), 1.0)
    remaining = max(0, TIME_BUDGET - total_training_time)
    imgs_per_sec = int(TOTAL_BATCH_SIZE / dt)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | "
        f"loss: {debiased:.4f} | lr: {lr:.2e} | "
        f"imgs/s: {imgs_per_sec:,} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    # Periodic best-checkpoint evaluation
    if step > 0 and (step - last_eval_step) >= EVAL_EVERY_STEPS:
        print()
        print(f"  [eval @ step {step}, t={total_training_time:.0f}s]", flush=True)
        model.eval()
        with autocast_ctx:
            ckpt_results = evaluate(model, TASK_NAME, DEVICE_BATCH_SIZE, metric=METRIC)
        ckpt_metric = ckpt_results["primary_metric"]
        print(
            f"  {METRIC}: {ckpt_metric:.4f} | acc: {ckpt_results['accuracy']:.4f}",
            flush=True,
        )
        if ckpt_metric > best_metric:
            best_metric = ckpt_metric
            best_state = copy.deepcopy(
                (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()
            )
            print(f"  *** new best: {best_metric:.4f} ***", flush=True)
        model.train()
        last_eval_step = step

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1

    if total_training_time >= TIME_BUDGET:
        break

print()

# Restore best checkpoint before final evaluation
if best_state is not None:
    print(f"Restoring best checkpoint (metric={best_metric:.4f})")
    _m = model._orig_mod if hasattr(model, "_orig_mod") else model
    _m.load_state_dict(best_state)

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
with autocast_ctx:
    results = evaluate(model, TASK_NAME, DEVICE_BATCH_SIZE, metric=METRIC)

t_end = time.time()
peak_vram_mb = peak_memory_mb()

primary_value = results["primary_metric"]
print("---")
print(f"val_{METRIC}:        {primary_value:.6f}")
print(f"val_accuracy:        {results['accuracy']:.6f}")
print(f"val_f1_macro:        {results['f1_macro']:.6f}")
print(f"val_f1_weighted:     {results['f1_weighted']:.6f}")
print(f"val_samples:         {results['num_samples']}")
print(f"training_seconds:    {total_training_time:.1f}")
print(f"total_seconds:       {t_end - t_start:.1f}")
print(f"peak_vram_mb:        {peak_vram_mb:.1f}")
print(f"total_images_k:      {total_images / 1000:.1f}")
print(f"num_steps:           {step}")
_m = model._orig_mod if hasattr(model, "_orig_mod") else model
print(f"num_params_m:        {_m.num_params() / 1e6:.2f}")
