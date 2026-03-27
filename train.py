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
# Experiment 2: ConvNeXt-Micro from scratch
#
# Small ConvNeXt-style network trained entirely from scratch on the font
# detection dataset. Depthwise-separable convolutions, inverted bottlenecks,
# LayerNorm, GELU — no pretrained weights.
#
# Architecture: stem → 4 stages of ConvNeXt blocks → head
#   dims:   [64, 128, 256, 512]
#   depths: [2, 2, 6, 2]  (~8M params)
#
# Hypothesis: from-scratch should show whether f1=1.0 from pretrained was
# genuine or just easy overfitting of the 3200-sample eval subset.
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise conv → LN → pointwise expand → GELU → pointwise contract."""

    def __init__(self, dim, expand=4, layer_scale=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * expand)
        self.pw2 = nn.Linear(dim * expand, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * layer_scale)

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)  # NCHW → NHWC for LayerNorm/Linear
        x = self.norm(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.pw2(x)
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        return residual + x


class ConvNeXtMicro(nn.Module):
    """ConvNeXt-Micro from scratch. ~8M params."""

    def __init__(self, num_classes=15, dims=(64, 128, 256, 512), depths=(2, 2, 6, 2)):
        super().__init__()
        # Stem: patchify 4×4
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6) if False else _LNWrapper(dims[0]),
        )
        # Stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            if i > 0:
                self.downsamples.append(
                    nn.Sequential(
                        _LNWrapper(dims[i - 1]),
                        nn.Conv2d(dims[i - 1], dim, kernel_size=2, stride=2),
                    )
                )
            else:
                self.downsamples.append(nn.Identity())
            self.stages.append(
                nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])
            )
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for down, stage in zip(self.downsamples, self.stages):
            x = down(x)
            x = stage(x)
        x = x.mean(dim=[-2, -1])  # global average pool
        x = self.norm(x)
        return self.head(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class _LNWrapper(nn.Module):
    """LayerNorm wrapper that handles NCHW tensors (normalises over C dim)."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEVICE_BATCH_SIZE = 64
BASE_LR = 4e-4  # reduced from 4e-3 — canonical ConvNeXt LR too high for 30k samples
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 200
LABEL_SMOOTHING = 0.1
EVAL_FIRST_STEP = 300
EVAL_EVERY_STEPS = 300

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

model = ConvNeXtMicro(num_classes=NUM_CLASSES).to(device)
print(f"Model params: {model.num_params() / 1e6:.2f}M")

if device.type == "cuda" and torch.cuda.is_available():
    try:
        model = torch.compile(model, dynamic=False)
    except Exception:
        pass  # skip compile if no C compiler available (e.g. WSL without build-essential)

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
