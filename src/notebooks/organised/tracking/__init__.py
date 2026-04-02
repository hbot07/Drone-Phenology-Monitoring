"""Organised, modular crown tracking framework.

This package is intentionally notebook-friendly:
- functions are pure where possible
- plotting functions save to disk by default (no implicit display)
"""

from .config import TrackingConfig
from .pipeline import run_tracking_pipeline
