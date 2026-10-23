"""Segmentation models and losses (requires the [seg] extra: torch, transformers<5)."""
from .loss import ActiveBoundaryLoss, JaccardLoss
from .models import SegFormer, SegformerDecodeCore

__all__ = ["SegFormer", "SegformerDecodeCore", "JaccardLoss", "ActiveBoundaryLoss"]
