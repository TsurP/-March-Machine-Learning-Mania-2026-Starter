"""Model training utilities for March Madness predictions."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from . import state
from .config import ELO_CONFIG


def _parse_seed(seed_str: str) -> int:
    """Extract numeric seed from string like 'W01', 'X16a' → 1, 16."""
    return int(seed_str[1:3])


def train_prediction_model() -> Dict[str, Any]:
    """Train a logistic regression on Elo/seed/Massey differences.

    Builds training data from historical tournament matchups (2003+).
    Features (Team1 - Team2 where Team1 = lower TeamID):
        - elo_diff: previous-season Elo difference
        - seed_diff: tournament seed difference
        - massey_mean_diff: previous-season MASSEY_MEAN_RANK difference

    Evaluates with 5-fold cross-validation.

    Returns:
        dict: Summary with training size, Brier score, and model info.
    """
    # Seed lookup
    seed_map = {}
    for df in [state.DATA["m_seeds"], state.DATA["w_seeds"]]:
        for _, row in df.iterrows():
            seed_map[(row["Season"], row["TeamID"])] = _parse_seed(row["Seed"])

    # Convenience: Massey composite ranks (men only, previous season)
    massey = state.DATA.get("m_massey_features")
    # Build training set from tournament games
    X, y = [], []
    for t_df in [state.DATA["m_tourney"], state.DATA["w_tourney"]]:
        for _, row in t_df.iterrows():
            season = row["Season"]
            if season < 2003:
                continue

            w_id, l_id = row["WTeamID"], row["LTeamID"]
            prev_season = season - 1
            # Use PREVIOUS season Elo as pre-tournament rating
            w_elo = state.ELO.get((prev_season, w_id), float(ELO_CONFIG.initial_rating))
            l_elo = state.ELO.get((prev_season, l_id), float(ELO_CONFIG.initial_rating))
            w_seed = seed_map.get((season, w_id), 8)
            l_seed = seed_map.get((season, l_id), 8)
            # Massey composite (men's only; fall back to neutral value if missing)
            w_massey = 0.0
            l_massey = 0.0
            if massey is not None and prev_season in massey.index.get_level_values("Season"):
                # Use .get with (season, teamid) if available
                if (prev_season, w_id) in massey.index:
                    w_massey = float(massey.loc[(prev_season, w_id)]["MASSEY_MEAN_RANK"])
                if (prev_season, l_id) in massey.index:
                    l_massey = float(massey.loc[(prev_season, l_id)]["MASSEY_MEAN_RANK"])

            # Convention: team1 = lower ID
            if w_id < l_id:
                X.append(
                    [
                        w_elo - l_elo,
                        l_seed - w_seed,
                        w_massey - l_massey,
                    ]
                )
                y.append(1)
            else:
                X.append(
                    [
                        l_elo - w_elo,
                        w_seed - l_seed,
                        l_massey - w_massey,
                    ]
                )
                y.append(0)

    X_arr, y_arr = np.array(X), np.array(y)

    # Train
    model = LogisticRegression(C=1.0, solver="lbfgs")
    model.fit(X_arr, y_arr)
    state.MODEL = model

    # Cross-val Brier score (lower = better)
    cv_scores = cross_val_score(
        LogisticRegression(C=1.0, solver="lbfgs"),
        X_arr,
        y_arr,
        scoring="neg_brier_score",
        cv=5,
    )
    brier = -cv_scores.mean()

    return {
        "status": "success",
        "training_games": len(y_arr),
        "win_rate_label1": f"{y_arr.mean():.3f}",
        "cv_brier_score": f"{brier:.4f}",
        "coefficients": {
            "elo_diff": f"{model.coef_[0][0]:.6f}",
            "seed_diff": f"{model.coef_[0][1]:.6f}",
            "massey_mean_diff": f"{model.coef_[0][2]:.6f}",
            "intercept": f"{model.intercept_[0]:.6f}",
        },
        "message": f"Model trained on {len(y_arr)} games. CV Brier: {brier:.4f}",
    }

