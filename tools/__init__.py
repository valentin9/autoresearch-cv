"""
tools/ — Reusable dataset generation utilities for autoresearch vision tasks.

This package grows over time as the agent builds new dataset generators.
Each module should be reusable across tasks with minimal configuration.

Available modules:
  synthetic    — Base classes and helpers for synthetic image dataset generation
  web_sources  — Downloading images from the web and HuggingFace datasets
  augmentation — Image augmentation pipelines
"""
