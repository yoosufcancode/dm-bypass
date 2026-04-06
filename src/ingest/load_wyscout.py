"""
Wyscout Open Dataset loader and normalizer.

Converts Wyscout 2017/18 event JSON format to the internal schema used
by all feature engineering functions — identical column names and semantics
to what clean_events() produced for StatsBomb data.

Dataset: Pappalardo et al. (2019) "A public data set of spatio-temporal
match events in soccer competitions", Scientific Data.
https://figshare.com/collections/Soccer_match_event_dataset/4415000/5

Expected directory layout:
    data/raw/wyscout/
        events/
            events_Spain.json
            events_England.json
            events_France.json
            events_Germany.json
            events_Italy.json
        matches/
            matches_Spain.json
            ...
        players.json
        teams.json
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Set


# ── Coordinate normalization ────────────────────────────────────────────────
# Wyscout: x ∈ [0,100], y ∈ [0,100]
# Internal (StatsBomb-equivalent): x ∈ [0,120], y ∈ [0,80]
_X_SCALE = 1.2
_Y_SCALE = 0.8

# ── Wyscout event type IDs ──────────────────────────────────────────────────
_EID_DUEL         = 1
_EID_FOUL         = 2
_EID_FREE_KICK    = 3
_EID_GK_REFLEX    = 4
_EID_INTERRUPTION = 5
_EID_OFFSIDE      = 6
_EID_OTHERS       = 7
_EID_PASS         = 8
_EID_SAVE         = 9
_EID_SHOT         = 10

# ── Duel sub-event IDs ──────────────────────────────────────────────────────
_DUEL_GROUND_ATT  = 10   # Ground attacking duel (attacker's event)
_DUEL_GROUND_DEF  = 11   # Ground defending duel (defender's event)
_DUEL_AERIAL      = 12   # Aerial duel
_DUEL_SLIDE       = 13   # Sliding tackle

# ── Pass sub-event IDs ──────────────────────────────────────────────────────
_PASS_CORNER      = 30
_PASS_FK_CROSS    = 31
_PASS_FK          = 32
_PASS_THROW       = 33
_PASS_GOAL_KICK   = 34
_PASS_PENALTY     = 35
_PASS_SIMPLE      = 84
_PASS_HIGH        = 85
_PASS_HEAD        = 86
_PASS_SMART       = 87   # "smart pass" = through-ball equivalent
_PASS_LAUNCH      = 88
_PASS_CROSS       = 89

# ── Others-on-ball sub-event IDs ────────────────────────────────────────────
_OTHERS_CLEARANCE = 70
_OTHERS_TOUCH     = 71
_OTHERS_HEAD      = 72
_OTHERS_ACCEL     = 73   # Acceleration = carry

# ── Tag IDs ─────────────────────────────────────────────────────────────────
_TAG_ACCURATE     = 1801
_TAG_NOT_ACCURATE = 1802
_TAG_GOAL_ASSIST  = 401
_TAG_KEY_PASS     = 402
_TAG_THROUGH_BALL = 703
_TAG_YELLOW       = 31
_TAG_RED          = 45
_TAG_INTERCEPTION = 1301

# Midfielder role code in Wyscout lineup
_MIDFIELDER_ROLE = "MD"


# ── Tag helpers ─────────────────────────────────────────────────────────────

def _has_tag(tags, tag_id: int) -> bool:
    if not isinstance(tags, list):
        return False
    return any(t.get("id") == tag_id for t in tags)


def _get_origin(positions) -> tuple:
    """First position entry is always the origin (start location)."""
    if not isinstance(positions, list) or len(positions) == 0:
        return None, None
    p = positions[0]
    return p.get("x"), p.get("y")


def _get_dest(positions) -> tuple:
    """Second position entry is always the destination (end location)."""
    if not isinstance(positions, list) or len(positions) < 2:
        return None, None
    p = positions[1]
    return p.get("x"), p.get("y")


def _norm_x(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return float(x) * _X_SCALE


def _norm_y(y) -> float:
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return np.nan
    return float(y) * _Y_SCALE


# ── Possession synthesis ─────────────────────────────────────────────────────

def synthesize_possession(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign synthetic possession IDs by detecting team changes.

    A new possession begins when the teamId changes from the previous event
    (excluding interruptions and offsides which don't transfer possession).
    """
    # Sort by period then time within period. matchPeriod = "1H"/"2H".
    period_order = {"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5}
    df = df.copy()
    df["_period_sort"] = df["matchPeriod"].map(period_order).fillna(9)
    df = df.sort_values(["_period_sort", "eventSec"]).drop(columns=["_period_sort"])
    poss_id = 1
    prev_team = None
    prev_eid = None
    ids = []

    for _, row in df.iterrows():
        curr_team = row.get("teamId")
        curr_eid = row.get("eventId")
        # Interruptions and offsides don't constitute possession changes
        if (
            prev_team is not None
            and curr_team != prev_team
            and prev_eid not in (_EID_INTERRUPTION, _EID_OFFSIDE)
            and curr_eid not in (_EID_INTERRUPTION, _EID_OFFSIDE)
        ):
            poss_id += 1
        ids.append(poss_id)
        prev_team = curr_team
        prev_eid = curr_eid

    df["possession"] = ids
    # possession_team_id = the team that owns this possession (first non-interruption event)
    poss_team = (
        df[~df["eventId"].isin([_EID_INTERRUPTION, _EID_OFFSIDE])]
        .groupby("possession")["teamId"]
        .first()
    )
    df["possession_team_id"] = df["possession"].map(poss_team)
    return df


# ── Main normalizer ──────────────────────────────────────────────────────────

def clean_wyscout_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a Wyscout events DataFrame to the internal schema.

    Input: DataFrame from json_normalize of a Wyscout events JSON file,
    already augmented with synthetic possession columns.

    Output: DataFrame with identical column semantics to what
    clean_events() produces for StatsBomb data.
    """
    rows = []
    prev_team_id = None

    for _, r in raw_df.iterrows():
        eid  = r.get("eventId")
        sub  = r.get("subEventId")
        tags = r.get("tags", [])
        pos  = r.get("positions", [])

        ox, oy = _get_origin(pos)
        dx, dy = _get_dest(pos)

        acc = _has_tag(tags, _TAG_ACCURATE)
        not_acc = _has_tag(tags, _TAG_NOT_ACCURATE)

        nx  = _norm_x(ox)
        ny  = _norm_y(oy)
        ndx = _norm_x(dx)
        ndy = _norm_y(dy)

        # ── Map event type ─────────────────────────────────────────────────
        type_name          = "Unknown"
        duel_type          = None
        duel_outcome       = None
        dribble_outcome    = None
        pass_outcome_name  = None   # None = completed (StatsBomb convention)
        pass_cross         = False
        pass_through_ball  = False
        pass_switch        = False
        pass_shot_assist   = False
        pass_goal_assist   = False
        pass_type_name     = None
        pass_body_part     = None
        foul_card          = None

        team_id = r.get("teamId")

        if eid == _EID_PASS:
            type_name = "Pass"
            pass_outcome_name = None if acc else ("Incomplete" if not_acc else None)
            pass_cross         = sub == _PASS_CROSS
            pass_through_ball  = _has_tag(tags, _TAG_THROUGH_BALL) or sub == _PASS_SMART
            pass_shot_assist   = _has_tag(tags, _TAG_KEY_PASS)
            pass_goal_assist   = _has_tag(tags, _TAG_GOAL_ASSIST)
            pass_body_part     = "Head" if sub == _PASS_HEAD else None
            if sub == _PASS_CORNER:
                pass_type_name = "Corner"
            elif sub in (_PASS_FK_CROSS, _PASS_FK):
                pass_type_name = "Free Kick"
            elif sub == _PASS_THROW:
                pass_type_name = "Throw-in"
            elif sub == _PASS_GOAL_KICK:
                pass_type_name = "Goal Kick"
            # Detect switch: large lateral (y) displacement
            if ndx is not np.nan and ndy is not np.nan and not np.isnan(ndx) and not np.isnan(ndy):
                if not np.isnan(ny) and abs(ndy - ny) > 32:
                    pass_switch = True

        elif eid == _EID_SHOT:
            type_name = "Shot"

        elif eid == _EID_DUEL:
            if sub == _DUEL_GROUND_ATT:
                if acc:
                    type_name = "Dribble"
                    dribble_outcome = "Complete"
                else:
                    type_name = "Dispossessed"
            elif sub == _DUEL_GROUND_DEF:
                if _has_tag(tags, _TAG_INTERCEPTION):
                    type_name = "Interception"
                else:
                    type_name = "Duel"
                    duel_type = "Tackle"
                    duel_outcome = "Won" if acc else "Lost In Play"
            elif sub == _DUEL_AERIAL:
                type_name = "Duel"
                duel_type = "Aerial Won" if acc else "Aerial Lost"
                duel_outcome = "Won" if acc else "Lost In Play"
            elif sub == _DUEL_SLIDE:
                type_name = "Duel"
                duel_type = "Tackle"
                duel_outcome = "Won" if acc else "Lost In Play"
            else:
                type_name = "Duel"
                duel_outcome = "Won" if acc else "Lost In Play"

        elif eid == _EID_FOUL:
            type_name = "Foul Committed"
            if _has_tag(tags, _TAG_YELLOW):
                foul_card = "Yellow Card"
            elif _has_tag(tags, _TAG_RED):
                foul_card = "Red Card"

        elif eid == _EID_OTHERS:
            if sub == _OTHERS_ACCEL:
                type_name = "Carry"
            elif sub == _OTHERS_CLEARANCE:
                type_name = "Clearance"
            else:
                # Ball Recovery: first touch when team just regained possession
                if prev_team_id is not None and team_id != prev_team_id:
                    type_name = "Ball Recovery"
                else:
                    type_name = "Touch"

        elif eid in (_EID_INTERRUPTION, _EID_OFFSIDE):
            type_name = "Interruption"

        elif eid in (_EID_GK_REFLEX, _EID_SAVE, _EID_FREE_KICK):
            type_name = "Goalkeeper" if eid in (_EID_GK_REFLEX, _EID_SAVE) else "Free Kick Start"

        else:
            type_name = f"Unknown_{eid}"

        prev_team_id = team_id

        # ── Build normalized row ────────────────────────────────────────────
        end_loc = [ndx, ndy] if (not np.isnan(ndx) and not np.isnan(ndy)) else None

        row_out = {
            "match_id":              r.get("matchId"),
            "team_id":               team_id,
            "team_name":             r.get("_team_name"),
            "player_id":             r.get("playerId"),
            "player_name":           r.get("_player_name"),
            "type_name":             type_name,
            "period":                1 if r.get("matchPeriod") == "1H" else 2,
            "timestamp":             pd.to_timedelta(r.get("eventSec", 0), unit="s"),
            "minute":                int(r.get("eventSec", 0) // 60),
            "second":                int(r.get("eventSec", 0) % 60),
            "possession":            r.get("possession"),
            "possession_team_id":    r.get("possession_team_id"),
            "possession_team_name":  r.get("_poss_team_name"),
            "duration":              0.0,
            "under_pressure":        False,   # not available in Wyscout
            "counterpress":          False,   # not available in Wyscout
            "id":                    r.get("id"),
            # Coordinates
            "x":                     nx,
            "y":                     ny,
            # Pass
            "pass.end_location":     end_loc if type_name == "Pass" else None,
            "pass_end_location":     end_loc if type_name == "Pass" else None,
            "pass.length":           None,
            "pass_length":           None,
            "pass.outcome.name":     pass_outcome_name,
            "pass_outcome_name":     pass_outcome_name,
            "pass.cross":            pass_cross,
            "pass_cross":            pass_cross,
            "pass.through_ball":     pass_through_ball,
            "pass_through_ball":     pass_through_ball,
            "pass.switch":           pass_switch,
            "pass_switch":           pass_switch,
            "pass.shot_assist":      pass_shot_assist,
            "pass_shot_assist":      pass_shot_assist,
            "pass.goal_assist":      pass_goal_assist,
            "pass_goal_assist":      pass_goal_assist,
            "pass.type.name":        pass_type_name,
            "pass_type_name":        pass_type_name,
            "pass.body_part.name":   pass_body_part,
            "pass_body_part_name":   pass_body_part,
            "pass.progressive":      False,
            "pass_progressive":      False,
            "pass.carry_id":         None,
            "pass_carry_id":         None,
            # Carry
            "carry.end_location":    end_loc if type_name == "Carry" else None,
            "carry_end_location":    end_loc if type_name == "Carry" else None,
            "carry.id":              None,
            "carry_id":              None,
            # Duel
            "duel.type.name":        duel_type,
            "duel_type":             duel_type,
            "duel.outcome.name":     duel_outcome,
            "duel_outcome":          duel_outcome,
            "duel.tackle":           None,
            "duel_tackle":           None,
            # Dribble
            "dribble.outcome.name":  dribble_outcome,
            "dribble_outcome_name":  dribble_outcome,
            # Foul
            "foul_committed.card.name":  foul_card,
            "foul_committed_card_name":  foul_card,
            "foul_committed.type.name":  None,
            "foul_committed_type_name":  None,
            "foul_won.advantage":    False,
            "foul_won_advantage":    False,
            # Shot
            "shot.statsbomb_xg":     np.nan,   # not available in Wyscout
            "shot_statsbomb_xg":     np.nan,
            "shot.key_pass_id":      None,
            "shot_key_pass_id":      None,
            "shot.carry_id":         None,
            "shot_carry_id":         None,
            # Block
            "block.deflection":      False,
            "block_deflection":      False,
            "block.block_type":      None,
            "block_block_type":      None,
            # Tactics
            "tactics.formation":     None,
            "tactics.lineup":        None,
            "tactics_formation":     None,
            "tactics_lineup":        None,
            # Play pattern (inferred from pass type)
            "play_pattern.name":     _infer_play_pattern(eid, sub, pass_type_name),
            "play_pattern_name":     _infer_play_pattern(eid, sub, pass_type_name),
            # Take-on
            "take_on.outcome.name":  None,
            "take_on_outcome_name":  None,
            # 50/50
            "50_50.outcome.name":    None,
            "50_50_outcome_name":    None,
            # Related events (not available)
            "related_events":        [],
            # Outcome coalesced
            "outcome_name":          duel_outcome or pass_outcome_name,
            # Team alias used by existing code
            "team.id":               team_id,
            "possession_team.id":    r.get("possession_team_id"),
        }
        rows.append(row_out)

    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_timedelta(out["timestamp"])
    return out


def _infer_play_pattern(eid, sub, pass_type_name) -> Optional[str]:
    if pass_type_name == "Corner":
        return "From Corner"
    if pass_type_name == "Free Kick":
        return "From Free Kick"
    if pass_type_name == "Throw-in":
        return "From Throw In"
    if pass_type_name == "Goal Kick":
        return "From Goal Kick"
    return "Regular Play"


# ── File loaders ─────────────────────────────────────────────────────────────

def load_wyscout_events(wyscout_dir: Path, league: str = "Spain") -> pd.DataFrame:
    """Load raw Wyscout events for a league from events_{league}.json."""
    path = wyscout_dir / "events" / f"events_{league}.json"
    if not path.exists():
        raise FileNotFoundError(f"Wyscout events file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    return df


def load_wyscout_matches(wyscout_dir: Path, league: str = "Spain") -> pd.DataFrame:
    """Load match metadata from matches_{league}.json."""
    path = wyscout_dir / "matches" / f"matches_{league}.json"
    if not path.exists():
        raise FileNotFoundError(f"Wyscout matches file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_wyscout_players(wyscout_dir: Path) -> Dict[int, dict]:
    """Load players.json and return dict keyed by wyId."""
    path = wyscout_dir / "players.json"
    if not path.exists():
        raise FileNotFoundError(f"Wyscout players file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["wyId"]: p for p in data}


def load_wyscout_teams(wyscout_dir: Path) -> Dict[int, dict]:
    """Load teams.json and return dict keyed by wyId."""
    path = wyscout_dir / "teams.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {t["wyId"]: t for t in data}


# ── Midfielder detection ─────────────────────────────────────────────────────

def get_midfielder_ids_wyscout(
    match_row: pd.Series,
    team_id: int,
    players_dict: Dict[int, dict],
) -> Set[int]:
    """
    Return player IDs classified as midfielders for a given team in a match.

    Wyscout lineup entries don't carry role codes, so we look up each player's
    role from players.json (role.code2 == "MD").
    """
    midfielder_ids: Set[int] = set()
    teams_data = match_row.get("teamsData", {})
    if not isinstance(teams_data, dict):
        return midfielder_ids

    # teamsData keys may be int or string
    team_entry = teams_data.get(team_id) or teams_data.get(str(team_id))
    if not team_entry:
        return midfielder_ids

    formation = team_entry.get("formation", {})
    lineup = formation.get("lineup", [])
    bench  = formation.get("bench", [])

    for entry in lineup + bench:
        pid = entry.get("playerId")
        if pid is None:
            continue
        pid = int(pid)
        player_info = players_dict.get(pid, {})
        role_info = player_info.get("role", {})
        if role_info.get("code2") == _MIDFIELDER_ROLE:
            midfielder_ids.add(pid)

    return midfielder_ids


def get_all_team_ids_in_match(match_row: pd.Series) -> list:
    """Return both team IDs from a match teamsData dict."""
    teams_data = match_row.get("teamsData", {})
    if not isinstance(teams_data, dict):
        return []
    result = []
    for k in teams_data.keys():
        try:
            result.append(int(k))
        except (ValueError, TypeError):
            pass
    return result


# ── Enrichment helpers ───────────────────────────────────────────────────────

def enrich_events_with_names(
    events_df: pd.DataFrame,
    players_dict: Dict[int, dict],
    teams_dict: Dict[int, dict],
) -> pd.DataFrame:
    """Add _player_name and _team_name columns to the raw events DataFrame."""
    events_df = events_df.copy()

    def _player_name(pid):
        if pd.isna(pid):
            return None
        info = players_dict.get(int(pid), {})
        return info.get("shortName") or info.get("lastName")

    def _team_name(tid):
        if pd.isna(tid):
            return None
        info = teams_dict.get(int(tid), {})
        return info.get("name")

    events_df["_player_name"] = events_df["playerId"].apply(_player_name)
    events_df["_team_name"]   = events_df["teamId"].apply(_team_name)
    return events_df


def build_poss_team_name(
    events_df: pd.DataFrame,
    teams_dict: Dict[int, dict],
) -> pd.DataFrame:
    """Add _poss_team_name based on possession_team_id."""
    events_df = events_df.copy()

    def _team_name(tid):
        if pd.isna(tid):
            return None
        info = teams_dict.get(int(tid), {})
        return info.get("name")

    events_df["_poss_team_name"] = events_df["possession_team_id"].apply(_team_name)
    return events_df


# ── Public entry point ───────────────────────────────────────────────────────

def load_and_clean_match(
    match_events_df: pd.DataFrame,
    match_row: pd.Series,
    players_dict: Dict[int, dict],
    teams_dict: Dict[int, dict],
) -> pd.DataFrame:
    """
    Full pipeline for one match: add names, synthesize possession, normalize.

    Parameters
    ----------
    match_events_df : pd.DataFrame
        Raw Wyscout events for a single matchId (from load_wyscout_events).
    match_row : pd.Series
        Row from matches DataFrame for this match.
    players_dict, teams_dict : dicts
        Loaded from load_wyscout_players / load_wyscout_teams.

    Returns
    -------
    pd.DataFrame
        Normalized events in internal schema.
    """
    df = match_events_df.copy()
    df = enrich_events_with_names(df, players_dict, teams_dict)
    df = synthesize_possession(df)
    df = build_poss_team_name(df, teams_dict)
    return clean_wyscout_events(df)
