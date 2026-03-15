"""Submission file generation for Kaggle."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from . import state
from .config import ELO_CONFIG


def _parse_seed(seed_str: str) -> int:
    """Extract numeric seed from string like 'W01', 'X16a' → 1, 16."""
    return int(seed_str[1:3])


def generate_submission(output_path: str = "/kaggle/working/submission.csv") -> Dict[str, Any]:
    """Generate predictions for every possible matchup and save submission.csv.

    For each row in the sample submission, extracts the two team IDs,
    computes features (Elo diff, seed diff), and predicts P(Team1 wins).
    Falls back to 0.5 if a team has no Elo rating.

    Returns:
        dict: Summary with number of predictions and output path.
    """
    sub = state.DATA["sample_sub"].copy()

    # Massey composite ranks (men only, previous season)
    massey = state.DATA.get("m_massey_features")

    # Seed lookup for ALL seasons (not just current)
    seed_map = {}
    for df in [state.DATA["m_seeds"], state.DATA["w_seeds"]]:
        for _, row in df.iterrows():
            seed_map[(row["Season"], row["TeamID"])] = _parse_seed(row["Seed"])

    preds = []
    for _, row in sub.iterrows():
        parts = row["ID"].split("_")
        season = int(parts[0])
        t1, t2 = int(parts[1]), int(parts[2])  # t1 < t2 by construction

        # Use prior season's Elo and Massey as pre-tournament rating
        prev_season = season - 1

        e1 = state.ELO.get((prev_season, t1), float(ELO_CONFIG.initial_rating))
        e2 = state.ELO.get((prev_season, t2), float(ELO_CONFIG.initial_rating))
        s1 = seed_map.get((season, t1), 8)
        s2 = seed_map.get((season, t2), 8)

        m1 = 0.0
        m2 = 0.0
        if massey is not None and prev_season in massey.index.get_level_values("Season"):
            if (prev_season, t1) in massey.index:
                m1 = float(massey.loc[(prev_season, t1)]["MASSEY_MEAN_RANK"])
            if (prev_season, t2) in massey.index:
                m2 = float(massey.loc[(prev_season, t2)]["MASSEY_MEAN_RANK"])

        features = np.array([[e1 - e2, s2 - s1, m1 - m2]])
        prob = state.MODEL.predict_proba(features)[0][1]
        prob = float(np.clip(prob, 0.01, 0.99))
        preds.append(prob)

    sub["Pred"] = preds
    sub.to_csv(output_path, index=False)

    return {
        "status": "success",
        "num_predictions": len(preds),
        "mean_pred": f"{np.mean(preds):.4f}",
        "std_pred": f"{np.std(preds):.4f}",
        "output_path": output_path,
        "message": f"Submission saved to {output_path} with {len(preds)} predictions.",
    }

