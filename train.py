"""
Autoresearch vision training script. Single-GPU, single-file.
Supports: CUDA (RTX 4090), MPS (Apple Silicon M-series), CPU.

Usage:
    uv run train.py

The task is defined by scope.md. Dataset must exist before running.
See prepare.py and program.md for the full protocol.
"""

import os

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
METRIC = scope["metric"]  # 'accuracy' | 'f1_macro' | 'f1_weighted'
NUM_CLASSES = get_num_classes(TASK_NAME)
CLASS_NAMES = get_class_names(TASK_NAME)

print(f"Task:        {TASK_NAME}")
print(f"Classes:     {CLASS_NAMES}")
print(f"Num classes: {NUM_CLASSES}")
print(f"Metric:      {METRIC}")
print(f"Time budget: {TIME_BUDGET}s")
print()

# ---------------------------------------------------------------------------
# Experiment 3: Pretrained ResNet-18 via timm (pipeline validation)
#
# Hypothesis: from-scratch models fail to converge in 20 min on MPS.
# Using a pretrained backbone should converge fast and validate the pipeline.
# If this also fails → dataset is broken. If it works → need better from-scratch
# training strategy (e.g. more epochs, different LR, or architecture changes).
#
# NOTE: pretrained weights are used here ONLY as a diagnostic baseline.
# The primary research goal remains from-scratch novel architectures.
# ---------------------------------------------------------------------------


class PretrainedClassifier(nn.Module):
    """Thin wrapper around a timm pretrained backbone + classifier head."""

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 15,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEVICE_BATCH_SIZE = 64
TOTAL_BATCH_SIZE = 128  # smaller effective batch for pretrained fine-tuning
BASE_LR = 1e-3  # higher LR for head, backbone will need lower
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05
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

_DEVICE_PEAK_FLOPS = {
    "cuda": 82.6e12,
    "mps": 14.2e12,
    "cpu": 1.0e12,
}

model = PretrainedClassifier(
    model_name="resnet18",
    num_classes=NUM_CLASSES,
    pretrained=True,
).to(device)
print(f"Model params: {model.num_params() / 1e6:.2f}M")

# torch.compile works on CUDA and CPU; skip on MPS (not yet fully supported)
if device.type != "mps":
    model = torch.compile(model, dynamic=False)

assert TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

train_loader = make_dataloader(
    TASK_NAME, DEVICE_BATCH_SIZE, "train", pin_memory=(device.type == "cuda")
)
images_per_epoch = len(train_loader) * DEVICE_BATCH_SIZE

# Use different LR for backbone vs head (standard fine-tuning)
backbone_params = [
    p for n, p in model.named_parameters() if "head" not in n and "fc" not in n
]
head_params = [p for n, p in model.named_parameters() if "head" in n or "fc" in n]
print(f"Backbone params: {sum(p.numel() for p in backbone_params) / 1e6:.2f}M")
print(f"Head params: {sum(p.numel() for p in head_params) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": BASE_LR * 0.1},  # 10x lower for backbone
        {"params": head_params, "lr": BASE_LR},
    ],
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8,
    fused=(device.type == "cuda"),
)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


def get_lr_scale(step_progress: float) -> float:
    """Cosine with warmup, returns scale factor [0, 1]."""
    warmup_steps = WARMUP_RATIO
    if step_progress < warmup_steps:
        return step_progress / warmup_steps if warmup_steps > 0 else 1.0
    t = (step_progress - warmup_steps) / max(1e-8, 1.0 - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * t))


print(f"Grad accum steps:   {grad_accum_steps}")
print(f"Images per epoch:   {images_per_epoch:,}")
print(f"Total batch size:   {TOTAL_BATCH_SIZE}")
print()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def device_synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()


def peak_memory_mb() -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0.0


t_start_training = time.time()
total_training_time = 0.0
step = 0
total_images = 0
smooth_loss = 0.0
train_iter = iter(train_loader)

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

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lr_scale = get_lr_scale(progress)
    for pg in optimizer.param_groups:
        pg["lr"] = pg["initial_lr"] * lr_scale if "initial_lr" in pg else pg["lr"]

    # Manual LR update based on progress
    warmup_frac = WARMUP_RATIO
    if progress < warmup_frac:
        lr_scale = (progress / warmup_frac) if warmup_frac > 0 else 1.0
    else:
        t = (progress - warmup_frac) / max(1e-8, 1.0 - warmup_frac)
        lr_scale = 0.5 * (1 + math.cos(math.pi * t))

    optimizer.param_groups[0]["lr"] = BASE_LR * 0.1 * lr_scale
    optimizer.param_groups[1]["lr"] = BASE_LR * lr_scale

    optimizer.step()

    device_synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 5:
        total_training_time += dt

    total_images += TOTAL_BATCH_SIZE
    ema_beta = 0.95
    smooth_loss = (
        ema_beta * smooth_loss + (1 - ema_beta) * batch_loss * grad_accum_steps
    )
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)
    imgs_per_sec = int(TOTAL_BATCH_SIZE / dt)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | "
        f"loss: {debiased:.4f} | lr: {optimizer.param_groups[1]['lr']:.2e} | "
        f"imgs/s: {imgs_per_sec:,} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

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


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

scope = load_scope()
TASK_NAME = scope["task_name"]
METRIC = scope["metric"]  # 'accuracy' | 'f1_macro' | 'f1_weighted'
NUM_CLASSES = get_num_classes(TASK_NAME)
CLASS_NAMES = get_class_names(TASK_NAME)

print(f"Task:        {TASK_NAME}")
print(f"Classes:     {CLASS_NAMES}")
print(f"Num classes: {NUM_CLASSES}")
print(f"Metric:      {METRIC}")
print(f"Time budget: {TIME_BUDGET}s")
print()

# ---------------------------------------------------------------------------
# Experiment 2: Smaller ConvNeXt-Nano + improved dataset (larger fonts, centered text)
#
# Hypothesis: baseline failed because:
#   1. Dataset had tiny fonts (16px) making glyphs unreadable
#   2. Model (7M params) too large to converge in 20min on MPS (~100 imgs/s)
#
# Changes:
#   - Dataset regenerated with font_size 28-72px, centered text
#   - Smaller model (2.5M params) = more steps per second
#   - Lower LR (1e-3) — large lr caused instability/plateau in exp 1
#   - Reduced label smoothing (0.05) — less noise in 15-class problem
# ---------------------------------------------------------------------------


@dataclass
class MicroConvConfig:
    img_size: int = IMG_SIZE
    num_classes: int = NUM_CLASSES
    # Smaller model for faster iteration on MPS
    stages: tuple = (
        (32, 2),  # stage 1
        (64, 2),  # stage 2
        (128, 3),  # stage 3
        (256, 2),  # stage 4
    )
    expansion: int = 4
    drop_path_rate: float = 0.05


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    noise.div_(keep_prob)
    return x * noise


class DepthwiseSepConvBlock(nn.Module):
    """
    ConvNeXt-style block:
      depthwise 7x7 -> LayerNorm -> pointwise expand -> GELU -> pointwise contract
    with stochastic depth (drop path).
    """

    def __init__(self, channels: int, expansion: int = 4, drop_path_rate: float = 0.0):
        super().__init__()
        hidden = channels * expansion
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels, bias=False
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pw_expand = nn.Linear(channels, hidden)
        self.pw_contract = nn.Linear(hidden, channels)
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-6)
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        residual = x
        x = self.dw_conv(x)
        # (B, C, H, W) -> (B, H, W, C) for LayerNorm + Linear
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = F.gelu(x)
        x = self.pw_contract(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = drop_path(x, self.drop_path_rate, self.training)
        return residual + x


class MicroConvNet(nn.Module):
    def __init__(self, config: MicroConvConfig):
        super().__init__()
        self.config = config

        # Stem: patchify with 4x4 non-overlapping conv (like ConvNeXt)
        first_channels = config.stages[0][0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, first_channels, kernel_size=4, stride=4, bias=False),
            nn.LayerNorm(first_channels, eps=1e-6),  # applied after permute
        )

        # Total blocks for drop path schedule
        total_blocks = sum(d for _, d in config.stages)
        dp_rates = torch.linspace(0, config.drop_path_rate, total_blocks).tolist()
        block_idx = 0

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = first_channels

        for stage_idx, (out_channels, num_blocks) in enumerate(config.stages):
            # Downsample between stages (except before first)
            if stage_idx > 0:
                self.downsamples.append(
                    nn.Sequential(
                        nn.LayerNorm(
                            in_channels, eps=1e-6
                        ),  # on channel dim after permute
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=2,
                            stride=2,
                            bias=False,
                        ),
                    )
                )
            else:
                self.downsamples.append(None)

            blocks = nn.Sequential(
                *[
                    DepthwiseSepConvBlock(
                        out_channels,
                        expansion=config.expansion,
                        drop_path_rate=dp_rates[block_idx + i],
                    )
                    for i in range(num_blocks)
                ]
            )
            self.stages.append(blocks)
            block_idx += num_blocks
            in_channels = out_channels

        # Head
        self.head_norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.head = nn.Linear(in_channels, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> logits: (B, num_classes)"""
        # Stem
        x = self.stem[0](x)  # (B, C, H/4, W/4)
        x = x.permute(0, 2, 3, 1)
        x = self.stem[1](x)
        x = x.permute(0, 3, 1, 2)

        # Stages
        for stage_idx, (blocks, downsample) in enumerate(
            zip(self.stages, self.downsamples)
        ):
            if downsample is not None:
                # LayerNorm on spatial features: permute -> norm -> permute
                x = x.permute(0, 2, 3, 1)
                x = downsample[0](x)
                x = x.permute(0, 3, 1, 2)
                x = downsample[1](x)
            x = blocks(x)

        # Head
        x = x.mean(dim=(2, 3))  # global average pool: (B, C)
        x = self.head_norm(x)
        x = self.head(x)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Hyperparameters (edit these freely — this is the agent's playground)
# ---------------------------------------------------------------------------

# Training
DEVICE_BATCH_SIZE = 64  # images per forward pass (reduce if OOM)
TOTAL_BATCH_SIZE = 256  # effective batch size (with gradient accumulation)
BASE_LR = 1e-3  # reduced from 4e-3 — exp 1 showed plateau at high LR
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.05
LABEL_SMOOTHING = 0.05  # reduced from 0.1 — less noise for fine-grained task

# Model config (edit freely)
MODEL_CONFIG = MicroConvConfig(
    img_size=IMG_SIZE,
    num_classes=NUM_CLASSES,
    stages=(
        (32, 2),
        (64, 2),
        (128, 3),
        (256, 2),
    ),
    expansion=4,
    drop_path_rate=0.05,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Device selection: CUDA > MPS > CPU
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(42)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# bf16 autocast: supported on CUDA and MPS (Apple Silicon); falls back to fp32 on CPU
_autocast_dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=_autocast_dtype)

# Peak FLOPS constants (for display only)
_DEVICE_PEAK_FLOPS = {
    "cuda": 82.6e12,  # RTX 4090 bf16
    "mps": 14.2e12,  # M3 Pro (approximate, GPU cores × ops)
    "cpu": 1.0e12,  # placeholder
}

model = MicroConvNet(MODEL_CONFIG).to(device)
print(f"Model params: {model.num_params() / 1e6:.2f}M")

# torch.compile works on CUDA and CPU; skip on MPS (not yet fully supported)
if device.type != "mps":
    model = torch.compile(model, dynamic=False)

assert TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

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
    fused=(device.type == "cuda"),  # fused AdamW only available on CUDA
)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


# LR schedule: linear warmup -> cosine decay
def get_lr(step: int, total_steps: int) -> float:
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    if step < warmup_steps:
        return BASE_LR * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return BASE_LR * 0.5 * (1 + math.cos(math.pi * progress))


print(f"Grad accum steps:   {grad_accum_steps}")
print(f"Images per epoch:   {images_per_epoch:,}")
print(f"Total batch size:   {TOTAL_BATCH_SIZE}")
print()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def device_synchronize():
    """Synchronize the current device (CUDA only; MPS/CPU are synchronous)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def peak_memory_mb() -> float:
    """Peak memory allocated on the current device, in MB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0.0


t_start_training = time.time()
total_training_time = 0.0
step = 0
total_images = 0
smooth_loss = 0.0
train_iter = iter(train_loader)

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

    # Gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # LR schedule (estimated total steps from time budget and current step speed)
    # Simple approach: use progress-based schedule from time
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    warmup_frac = WARMUP_RATIO
    if progress < warmup_frac:
        lr = BASE_LR * (progress / warmup_frac) if warmup_frac > 0 else BASE_LR
    else:
        t = (progress - warmup_frac) / max(1e-8, 1.0 - warmup_frac)
        lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * t))

    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.step()

    device_synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 5:
        total_training_time += dt

    total_images += TOTAL_BATCH_SIZE
    ema_beta = 0.95
    smooth_loss = (
        ema_beta * smooth_loss + (1 - ema_beta) * batch_loss * grad_accum_steps
    )
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)
    imgs_per_sec = int(TOTAL_BATCH_SIZE / dt)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | "
        f"loss: {debiased:.4f} | lr: {lr:.2e} | "
        f"imgs/s: {imgs_per_sec:,} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
with autocast_ctx:
    results = evaluate(model, TASK_NAME, DEVICE_BATCH_SIZE, metric=METRIC)

t_end = time.time()
peak_vram_mb = peak_memory_mb()

# ---------------------------------------------------------------------------
# Summary (grep-friendly: each key starts a line with "key: value")
# ---------------------------------------------------------------------------

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
print(
    f"num_params_m:        {(model._orig_mod if hasattr(model, '_orig_mod') else model).num_params() / 1e6:.2f}"
)
