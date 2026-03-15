"""Feature engineering using MMasseyOrdinals rankings.

This turns the raw system-specific rankings (e.g. Sagarin, Pomeroy, RPI)
into per-team, per-season features that can be joined into the model.
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from . import state


def build_massey_features() -> Dict[str, Any]:
    """Aggregate Massey ordinal rankings into team-season features.

    Strategy:
    - Use the men's MMasseyOrdinals file loaded into `state.DATA["m_massey"]`.
    - For each season, find the last available `RankingDayNum` (typically
      very close to tournament time).
    - Restrict to those rows, then pivot to a wide table with:
        index: (Season, TeamID)
        columns: SystemName
        values: OrdinalRank
    - Also compute:
        - `MASSEY_MEAN_RANK`: mean rank across systems for each team.

    The resulting DataFrame is stored as `state.DATA["m_massey_features"]`.
    """
    df = state.DATA["m_massey"]

    # Ensure expected columns exist (schema defined by Kaggle competition)
    required_cols = {
        "Season",
        "RankingDayNum",
        "SystemName",
        "TeamID",
        "OrdinalRank",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"MMasseyOrdinals missing columns: {missing}")

    # For each season, keep only the last available ranking day
    last_day = (
        df.groupby("Season", as_index=False)["RankingDayNum"].max().rename(
            columns={"RankingDayNum": "LastRankingDayNum"}
        )
    )
    df_last = df.merge(
        last_day, on=["Season"], how="inner"
    )
    df_last = df_last[df_last["RankingDayNum"] == df_last["LastRankingDayNum"]]

    # Pivot to wide: one column per system
    pivot = df_last.pivot_table(
        index=["Season", "TeamID"],
        columns="SystemName",
        values="OrdinalRank",
        aggfunc="mean",
    )

    # Compute an overall composite rank (lower is better)
    pivot["MASSEY_MEAN_RANK"] = pivot.mean(axis=1)

    # Store for downstream usage (e.g., join into model features)
    state.DATA["m_massey_features"] = pivot

    return {
        "status": "success",
        "seasons": int(pivot.index.get_level_values("Season").nunique()),
        "teams": int(pivot.index.get_level_values("TeamID").nunique()),
        "systems": [c for c in pivot.columns if c != "MASSEY_MEAN_RANK"],
        "message": "Built per-team Massey-based ranking features at last ranking day.",
    }

