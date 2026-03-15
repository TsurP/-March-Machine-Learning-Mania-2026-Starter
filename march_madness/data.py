"""Data loading utilities for the March Madness competition."""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from .config import APP_CONFIG
from . import state


def load_competition_data() -> Dict[str, Any]:
    """Load all March Madness competition CSV files and return a summary.

    Reads men's and women's team info, regular season results,
    tournament results, tournament seeds, and the sample submission file.

    Returns:
        dict: Summary with status, dataset sizes, and a message.
    """
    data_dir = APP_CONFIG.data_dir

    state.DATA["m_teams"] = pd.read_csv(f"{data_dir}/MTeams.csv")
    state.DATA["w_teams"] = pd.read_csv(f"{data_dir}/WTeams.csv")
    state.DATA["m_regular"] = pd.read_csv(
        f"{data_dir}/MRegularSeasonCompactResults.csv"
    )
    # Detailed regular season results (men's) for box-score style stats
    state.DATA["m_regular_detail"] = pd.read_csv(
        f"{data_dir}/MRegularSeasonDetailedResults.csv"
    )
    state.DATA["w_regular"] = pd.read_csv(
        f"{data_dir}/WRegularSeasonCompactResults.csv"
    )
    state.DATA["m_tourney"] = pd.read_csv(f"{data_dir}/MNCAATourneyCompactResults.csv")
    state.DATA["w_tourney"] = pd.read_csv(f"{data_dir}/WNCAATourneyCompactResults.csv")
    state.DATA["m_seeds"] = pd.read_csv(f"{data_dir}/MNCAATourneySeeds.csv")
    state.DATA["w_seeds"] = pd.read_csv(f"{data_dir}/WNCAATourneySeeds.csv")
    # Team–conference membership and conference tourney games
    state.DATA["m_team_conferences"] = pd.read_csv(
        f"{data_dir}/MTeamConferences.csv"
    )
    state.DATA["w_team_conferences"] = pd.read_csv(
        f"{data_dir}/WTeamConferences.csv"
    )
    state.DATA["m_conf_tourney"] = pd.read_csv(
        f"{data_dir}/MConferenceTourneyGames.csv"
    )
    state.DATA["w_conf_tourney"] = pd.read_csv(
        f"{data_dir}/WConferenceTourneyGames.csv"
    )
    # Men's Massey-style ordinal rankings from many systems
    state.DATA["m_massey"] = pd.read_csv(f"{data_dir}/MMasseyOrdinals.csv")
    state.DATA["sample_sub"] = pd.read_csv(f"{data_dir}/SampleSubmissionStage1.csv")

    m_regular = state.DATA["m_regular"]
    w_regular = state.DATA["w_regular"]

    return {
        "status": "success",
        "seasons": f"{m_regular['Season'].min()}-{m_regular['Season'].max()}",
        "mens_teams": len(state.DATA["m_teams"]),
        "womens_teams": len(state.DATA["w_teams"]),
        "regular_season_games": len(m_regular) + len(w_regular),
        "tourney_games": len(state.DATA["m_tourney"]) + len(state.DATA["w_tourney"]),
        "submission_rows": len(state.DATA["sample_sub"]),
        "message": "All data loaded successfully. Ready for feature engineering.",
    }

