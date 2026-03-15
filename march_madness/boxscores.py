"""Team-level box score feature engineering from detailed results."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from . import state


def compute_team_boxscores() -> Dict[str, Any]:
    """Aggregate men's regular season detailed results into team box scores.

    Uses `MRegularSeasonDetailedResults.csv` (already loaded into
    `state.DATA["m_regular_detail"]`) to compute per-team, per-season
    shooting percentages, rebounding, assists, turnovers, steals, blocks,
    and fouls.

    The aggregated frame is stored in `state.DATA["m_team_boxscores"]`
    with index (Season, TeamID).
    """
    df = state.DATA["m_regular_detail"]

    # Stats we will aggregate for each team in each game
    stat_pairs = {
        "FGM": ("WFGM", "LFGM"),
        "FGA": ("WFGA", "LFGA"),
        "FGM3": ("WFGM3", "LFGM3"),
        "FGA3": ("WFGA3", "LFGA3"),
        "FTM": ("WFTM", "LFTM"),
        "FTA": ("WFTA", "LFTA"),
        "OR": ("WOR", "LOR"),
        "DR": ("WDR", "LDR"),
        "Ast": ("WAst", "LAst"),
        "TO": ("WTO", "LTO"),
        "Stl": ("WStl", "LStl"),
        "Blk": ("WBlk", "LBlk"),
        "PF": ("WPF", "LPF"),
    }

    rows = []
    for _, row in df.iterrows():
        season = int(row["Season"])

        # Winning team stats
        w_team = int(row["WTeamID"])
        w_stats = {"Season": season, "TeamID": w_team}
        # Losing team stats
        l_team = int(row["LTeamID"])
        l_stats = {"Season": season, "TeamID": l_team}

        for base, (w_col, l_col) in stat_pairs.items():
            w_val = float(row[w_col])
            l_val = float(row[l_col])
            w_stats[base] = w_val
            l_stats[base] = l_val

        rows.append(w_stats)
        rows.append(l_stats)

    import pandas as pd

    per_game = pd.DataFrame(rows)

    grouped = per_game.groupby(["Season", "TeamID"]).sum(numeric_only=True)

    # Derived percentages and totals
    grouped["FG_PCT"] = grouped["FGM"] / grouped["FGA"].replace(0, np.nan)
    grouped["FG3_PCT"] = grouped["FGM3"] / grouped["FGA3"].replace(0, np.nan)
    grouped["FT_PCT"] = grouped["FTM"] / grouped["FTA"].replace(0, np.nan)
    grouped["TR"] = grouped["OR"] + grouped["DR"]

    # Replace NaNs from 0-division with 0.0
    grouped = grouped.fillna(0.0)

    state.DATA["m_team_boxscores"] = grouped

    # Simple summary for the tool response
    num_seasons = grouped.index.get_level_values("Season").nunique()
    num_teams = grouped.index.get_level_values("TeamID").nunique()

    return {
        "status": "success",
        "seasons": num_seasons,
        "teams": num_teams,
        "features": list(grouped.columns),
        "message": "Aggregated men's detailed regular season results into team box scores.",
    }

