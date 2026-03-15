"""
March Madness 2026 prediction pipeline package.

Provides a production-style structure around data loading, feature
engineering (Elo), model training, submission generation, and the
Gemini ADK-based agent pipeline.
"""

from . import config, state

__all__ = ["config", "state"]

