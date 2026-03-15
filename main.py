"""Entrypoint for running the March Madness prediction pipeline.

This script delegates to the production-grade package under
`march_madness`, which contains the modular implementation:

- `march_madness.data` – data loading
- `march_madness.elo` – Elo feature engineering
- `march_madness.model` – model training
- `march_madness.submission` – submission generation
- `march_madness.agents` – ADK agent wiring
- `march_madness.runner` – high-level pipeline runner
"""

from __future__ import annotations

import warnings

from march_madness.runner import run_sync


def main() -> None:
    """Run the full March Madness pipeline once."""
    warnings.filterwarnings("ignore")
    run_sync()


if __name__ == "__main__":
    main()

