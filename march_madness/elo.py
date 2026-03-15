"""Elo rating computation for March Madness teams."""

from __future__ import annotations

from typing import Dict, Tuple, Any, Tuple as Tup

import pandas as pd

from .config import ELO_CONFIG
from . import state


def _run_elo(
    regular_df: pd.DataFrame, tourney_df: pd.DataFrame
) -> Tup[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Run Elo over all games and also compute strength of schedule.

    Returns:
        season_elos: final Elo per (season, team)
        sos: average opponent Elo per (season, team)
    """
    elo: Dict[int, float] = {}
    season_elos: Dict[Tuple[int, int], float] = {}
    sos: Dict[Tuple[int, int], float] = {}

    # Track opponent strength within a season
    opp_elo_sum: Dict[int, float] = {}
    games_played: Dict[int, int] = {}

    all_games = pd.concat([regular_df, tourney_df]).sort_values(["Season", "DayNum"])
    prev_season = None
    # Pre-compute max regular-season DayNum per season to weight late games more
    max_day_by_season = (
        regular_df.groupby("Season")["DayNum"].max().to_dict()
        if not regular_df.empty
        else {}
    )

    for _, row in all_games.iterrows():
        season = row["Season"]
        if season != prev_season and prev_season is not None:
            # Close out previous season: snapshot Elo and SOS, then regress ratings
            for tid, r in elo.items():
                season_elos[(prev_season, tid)] = r
            for tid, total_opp in opp_elo_sum.items():
                gp = games_played.get(tid, 0)
                if gp > 0:
                    sos[(prev_season, tid)] = total_opp / gp
            opp_elo_sum = {}
            games_played = {}

            elo = {
                tid: 0.75 * r + 0.25 * ELO_CONFIG.initial_rating
                for tid, r in elo.items()
            }
        prev_season = season

        w_id, l_id = row["WTeamID"], row["LTeamID"]
        w_elo = elo.get(w_id, float(ELO_CONFIG.initial_rating))
        l_elo = elo.get(l_id, float(ELO_CONFIG.initial_rating))

        # Update strength-of-schedule trackers using pre-game opponent Elo
        opp_elo_sum[w_id] = opp_elo_sum.get(w_id, 0.0) + l_elo
        games_played[w_id] = games_played.get(w_id, 0) + 1
        opp_elo_sum[l_id] = opp_elo_sum.get(l_id, 0.0) + w_elo
        games_played[l_id] = games_played.get(l_id, 0) + 1

        # Home court adjustment
        w_loc = row.get("WLoc", "N")
        if w_loc == "H":
            w_adj = w_elo + ELO_CONFIG.home_court_advantage
        elif w_loc == "A":
            w_adj = w_elo - ELO_CONFIG.home_court_advantage
        else:
            w_adj = w_elo

        # Expected win probability & update
        exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_adj) / 400.0))

        # Give more weight to late-season games:
        # scale K from 0.5*K (start of season) up to 1.5*K (end of regular season).
        max_day = max_day_by_season.get(season)
        if max_day and max_day > 0:
            phase = min(max(row["DayNum"] / max_day, 0.0), 1.0)
            k_eff = ELO_CONFIG.k_factor * (0.5 + phase)
        else:
            k_eff = ELO_CONFIG.k_factor

        elo[w_id] = w_elo + k_eff * (1.0 - exp_w)
        elo[l_id] = l_elo + k_eff * (0.0 - (1.0 - exp_w))

    if prev_season is not None:
        for tid, r in elo.items():
            season_elos[(prev_season, tid)] = r
        for tid, total_opp in opp_elo_sum.items():
            gp = games_played.get(tid, 0)
            if gp > 0:
                sos[(prev_season, tid)] = total_opp / gp

    return season_elos, sos


def compute_elo_ratings() -> Dict[str, Any]:
    """Compute Elo ratings for all men's and women's teams across all seasons.

    Uses a standard Elo system with K-factor, home court advantage,
    and between-season regression toward the mean.

    Returns:
        dict: Summary with top-rated teams and total ratings computed.
    """
    m_elos, m_sos = _run_elo(state.DATA["m_regular"], state.DATA["m_tourney"])
    w_elos, w_sos = _run_elo(state.DATA["w_regular"], state.DATA["w_tourney"])
    state.ELO.update(m_elos)
    state.ELO.update(w_elos)

    # Combine SOS maps and expose as a DataFrame for downstream features
    import pandas as pd

    sos_all = {}
    sos_all.update(m_sos)
    sos_all.update(w_sos)
    if sos_all:
        sos_index = pd.MultiIndex.from_tuples(
            sos_all.keys(), names=["Season", "TeamID"]
        )
        sos_df = pd.DataFrame({"SOS": list(sos_all.values())}, index=sos_index)
        state.DATA["sos"] = sos_df

    # Conference-level strength, using final Elo and SOS
    def _build_conf_strength(
        elos: Dict[Tuple[int, int], float],
        sos: Dict[Tuple[int, int], float],
        team_confs: pd.DataFrame,
        gender: str,
    ) -> pd.DataFrame:
        if team_confs.empty:
            return pd.DataFrame()
        # Choose conference identifier column
        conf_col = "ConfAbbrev" if "ConfAbbrev" in team_confs.columns else "ConfID"
        tc = team_confs[["Season", "TeamID", conf_col]].copy()
        # Build per-team frame from Elo and SOS dicts
        idx = pd.MultiIndex.from_tuples(elos.keys(), names=["Season", "TeamID"])
        team_df = pd.DataFrame({"Elo": list(elos.values())}, index=idx)
        if sos:
            sos_idx = pd.MultiIndex.from_tuples(sos.keys(), names=["Season", "TeamID"])
            sos_df = pd.DataFrame({"SOS": list(sos.values())}, index=sos_idx)
            team_df = team_df.join(sos_df, how="left")
        team_df = team_df.reset_index()
        merged = team_df.merge(tc, on=["Season", "TeamID"], how="left")
        # Some teams (e.g., very old seasons) may lack conference info
        merged = merged.dropna(subset=[conf_col])
        grouped = (
            merged.groupby(["Season", conf_col])
            .agg(
                conf_mean_elo=("Elo", "mean"),
                conf_mean_sos=("SOS", "mean"),
                num_teams=("TeamID", "nunique"),
            )
            .reset_index()
        )
        grouped["Gender"] = gender
        grouped = grouped.set_index(["Season", conf_col, "Gender"])
        return grouped

    m_conf_strength = _build_conf_strength(
        m_elos, m_sos, state.DATA["m_team_conferences"], gender="M"
    )
    w_conf_strength = _build_conf_strength(
        w_elos, w_sos, state.DATA["w_team_conferences"], gender="W"
    )
    if not m_conf_strength.empty or not w_conf_strength.empty:
        conf_df = pd.concat([m_conf_strength, w_conf_strength], axis=0)
        state.DATA["conference_strength"] = conf_df

    # Top teams for display
    m_names = dict(
        zip(state.DATA["m_teams"]["TeamID"], state.DATA["m_teams"]["TeamName"])
    )
    w_names = dict(
        zip(state.DATA["w_teams"]["TeamID"], state.DATA["w_teams"]["TeamName"])
    )
    latest_m = max(s for s, _ in m_elos.keys())
    latest_w = max(s for s, _ in w_elos.keys())
    top_m = sorted(
        [(tid, r) for (s, tid), r in m_elos.items() if s == latest_m],
        key=lambda x: -x[1],
    )[:5]
    top_w = sorted(
        [(tid, r) for (s, tid), r in w_elos.items() if s == latest_w],
        key=lambda x: -x[1],
    )[:5]

    return {
        "status": "success",
        "total_ratings": len(state.ELO),
        "top_mens": [f"{m_names.get(t, t)}: {r:.0f}" for t, r in top_m],
        "top_womens": [f"{w_names.get(t, t)}: {r:.0f}" for t, r in top_w],
        "message": f"Elo computed through {latest_m} (men) and {latest_w} (women).",
    }

