"""Shared mutable state for the March Madness pipeline tools."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import ELO_CONFIG


# Loaded dataframes: populated by the data loading tool.
DATA: Dict[str, Any] = {}

# Elo ratings keyed by (season, team_id).
ELO: Dict[Tuple[int, int], float] = {}

# Trained sklearn model.
MODEL: Optional[LogisticRegression] = None


def reset_state() -> None:
    """Clear all global state. Useful for tests or repeated runs."""
    DATA.clear()
    ELO.clear()
    global MODEL
    MODEL = None


def default_elo() -> float:
    """Return the default Elo rating."""
    return float(ELO_CONFIG.initial_rating)


def clip_probabilities(probs, low: float = 0.01, high: float = 0.99):
    """Clip prediction probabilities to a reasonable range."""
    return np.clip(probs, low, high)

