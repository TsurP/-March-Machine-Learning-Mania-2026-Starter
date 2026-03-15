"""Model validation against historical tournament results."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from . import state
from .config import ELO_CONFIG


def _parse_seed(seed_str: str) -> int:
    """Extract numeric seed from string like 'W01', 'X16a' → 1, 16."""
    return int(seed_str[1:3])


def validate_model_on_history(min_season: int = 2003) -> Dict[str, Any]:
    """Evaluate the current model against historical tournament games.

    Assumes:
        - Elo ratings have been computed and stored in `state.ELO`.
        - Massey features (for men's teams) have been built and stored in
          `state.DATA["m_massey_features"]` (optional but used when present).
        - The global `state.MODEL` has been trained using the same feature
          definition as in `train_prediction_model`.

    For every historical tournament game (men's and women's) with
    Season >= `min_season`, this function:
        - Reconstructs the feature vector:
            [elo_diff, seed_diff, massey_mean_diff]
          using the *previous* season's ratings.
        - Computes the model's predicted probability that the lower-
          numbered TeamID wins.
        - Compares to the actual outcome (1 if lower ID won, else 0).

    Returns:
        dict with overall Brier score, per-gender Brier scores, and counts.
    """
    if state.MODEL is None:
        raise RuntimeError("MODEL is not trained. Run train_prediction_model first.")

    # Seed lookup
    seed_map = {}
    for df in [state.DATA["m_seeds"], state.DATA["w_seeds"]]:
        for _, row in df.iterrows():
            seed_map[(row["Season"], row["TeamID"])] = _parse_seed(row["Seed"])

    massey = state.DATA.get("m_massey_features")

    y_true_all, y_pred_all = [], []
    y_true_m, y_pred_m = [], []
    y_true_w, y_pred_w = [], []

    def _eval_tourney(df, gender: str):
        nonlocal y_true_all, y_pred_all, y_true_m, y_pred_m, y_true_w, y_pred_w
        for _, row in df.iterrows():
            season = row["Season"]
            if season < min_season:
                continue

            w_id, l_id = row["WTeamID"], row["LTeamID"]
            prev_season = season - 1

            # Elo and seeds from prior season / current tournament
            w_elo = state.ELO.get(
                (prev_season, w_id), float(ELO_CONFIG.initial_rating)
            )
            l_elo = state.ELO.get(
                (prev_season, l_id), float(ELO_CONFIG.initial_rating)
            )
            w_seed = seed_map.get((season, w_id), 8)
            l_seed = seed_map.get((season, l_id), 8)

            # Massey composite (men's only)
            w_massey = 0.0
            l_massey = 0.0
            if gender == "M" and massey is not None:
                if prev_season in massey.index.get_level_values("Season"):
                    if (prev_season, w_id) in massey.index:
                        w_massey = float(
                            massey.loc[(prev_season, w_id)]["MASSEY_MEAN_RANK"]
                        )
                    if (prev_season, l_id) in massey.index:
                        l_massey = float(
                            massey.loc[(prev_season, l_id)]["MASSEY_MEAN_RANK"]
                        )

            # Define Team1 as lower ID to match training convention
            if w_id < l_id:
                x = np.array([[w_elo - l_elo, l_seed - w_seed, w_massey - l_massey]])
                y_true = 1
            else:
                x = np.array([[l_elo - w_elo, w_seed - l_seed, l_massey - w_massey]])
                y_true = 0

            prob = float(state.MODEL.predict_proba(x)[0][1])
            prob = float(np.clip(prob, 0.01, 0.99))

            y_true_all.append(y_true)
            y_pred_all.append(prob)
            if gender == "M":
                y_true_m.append(y_true)
                y_pred_m.append(prob)
            else:
                y_true_w.append(y_true)
                y_pred_w.append(prob)

    _eval_tourney(state.DATA["m_tourney"], gender="M")
    _eval_tourney(state.DATA["w_tourney"], gender="W")

    def _brier(y_true, y_pred):
        if not y_true:
            return None
        yt = np.array(y_true, dtype=float)
        yp = np.array(y_pred, dtype=float)
        return float(np.mean((yp - yt) ** 2))

    result = {
        "status": "success",
        "min_season": min_season,
        "total_games": len(y_true_all),
        "mens_games": len(y_true_m),
        "womens_games": len(y_true_w),
        "brier_overall": _brier(y_true_all, y_pred_all),
        "brier_mens": _brier(y_true_m, y_pred_m),
        "brier_womens": _brier(y_true_w, y_pred_w),
        "message": (
            "Validation against historical tournament results complete. "
            "Scores are in Brier units (lower is better)."
        ),
    }

    return result

