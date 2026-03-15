"""Configuration and constants for the March Madness pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    data_dir: str = "./data"
    gemini_model: str = "gemini-2.5-flash"
    current_season: int = 2026


@dataclass(frozen=True)
class EloConfig:
    k_factor: int = 20
    initial_rating: int = 1500
    home_court_advantage: int = 100


APP_CONFIG = AppConfig()
ELO_CONFIG = EloConfig()


def get_google_api_key() -> str:
    """Return the Google API key, raising if it is not set."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable must be set to run the pipeline."
        )
    return api_key

