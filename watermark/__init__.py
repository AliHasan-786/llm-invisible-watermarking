"""Watermark package with lazy exports for optional generation dependencies."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detector import WatermarkDetector
    from .logits_processor import WatermarkLogitsProcessor

__all__ = ["WatermarkDetector", "WatermarkLogitsProcessor"]


def __getattr__(name):
    if name == "WatermarkDetector":
        from .detector import WatermarkDetector

        return WatermarkDetector
    if name == "WatermarkLogitsProcessor":
        from .logits_processor import WatermarkLogitsProcessor

        return WatermarkLogitsProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
