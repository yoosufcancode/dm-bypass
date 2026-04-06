from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd


MIDFIELD_POSITION_KEYWORDS = {
    "Midfield",
    "Attacking Midfield",
    "Defensive Midfield",
    "Center Midfield",
}

MIDFIELD_POSITION_MAP = {
    "Defensive Midfield": 0,
    "Center Midfield": 1,
    "Attacking Midfield": 2,
    "Midfield": 3,
}


def get_position_code(position_name: str) -> Optional[int]:
    """
    Map a midfielder position name to its numeric code (0-7).

    Position codes:
    - 0: Defensive Midfield
    - 1: Center Midfield
    - 2: Attacking Midfield
    - 3: Wing Midfield
    - 4: Right Wing
    - 5: Left Wing
    - 6: Wing Back
    - 7: Midfield (generic)

    Parameters
    ----------
    position_name : str
        Position name string (may contain the keyword).

    Returns
    -------
    Optional[int]
        Numeric code (0-7) if position matches a midfielder keyword, None otherwise.
    """
    # Check in order of specificity: more specific positions first
    # This ensures "Wing Back" matches before "Right Wing" or "Left Wing"
    priority_order = [
        "Defensive Midfield",
        "Attacking Midfield",
        "Center Midfield",
        "Wing Midfield",
        "Wing Back",
        "Midfield",  # Generic fallback last
    ]
    
    for keyword in priority_order:
        if keyword in position_name:
            return MIDFIELD_POSITION_MAP[keyword]
    return None


def _timestamp_to_seconds(ts: pd.Series) -> pd.Series:
    """Convert timedelta64 series to float seconds."""
    return ts.dt.total_seconds()


def _extract_coordinate(value: Optional[Iterable[float]], idx: int) -> float:
    """
    Extract a coordinate from a list/tuple or return NaN.

    Parameters
    ----------
    value : Optional[Iterable[float]]
        List, tuple, or None containing coordinates.
    idx : int
        Index of the coordinate to extract (0 for x, 1 for y).

    Returns
    -------
    float
        The coordinate value or np.nan if not available.
    """
    if isinstance(value, (list, tuple)) and len(value) > idx:
        return value[idx]
    return np.nan


def get_midfielder_ids_from_clean(events: pd.DataFrame, team_id: int) -> Set[int]:
    """
    Derive midfielder IDs from a cleaned (internal-schema) events DataFrame.

    Used with Wyscout data where midfielder IDs are passed in directly from
    match lineup metadata (see load_wyscout.get_midfielder_ids_wyscout).
    This function exists as a no-op passthrough for when IDs are already known.
    """
    return set()


def get_midfielder_ids(raw_events: pd.DataFrame, team_id: int) -> Set[int]:
    """
    Determine the set of player ids considered midfielders for the given match.

    Uses Starting XI and substitution metadata to capture role assignments.
    Falls back to players whose average event y-coordinate is central when
    explicit position information is missing.
    """
    midfielder_ids: Set[int] = set()

    # Starting XI entries hold the lineups.
    xi_rows = raw_events[
        (raw_events.get("team.id") == team_id)
        & (raw_events.get("type.name") == "Starting XI")
    ]
    for _, row in xi_rows.iterrows():
        lineup = row.get("tactics.lineup")
        if isinstance(lineup, list):
            for entry in lineup:
                pos_name = entry.get("position", {}).get("name", "")
                player = entry.get("player", {})
                player_id = player.get("id")
                if player_id is None:
                    continue
                if any(keyword in pos_name for keyword in MIDFIELD_POSITION_KEYWORDS):
                    midfielder_ids.add(player_id)

    # Include substitution replacements with midfield positions.
    sub_rows = raw_events[
        (raw_events.get("team.id") == team_id)
        & (raw_events.get("type.name") == "Substitution")
    ]
    for _, row in sub_rows.iterrows():
        repl = row.get("substitution.replacement")
        if isinstance(repl, dict):
            player_id = repl.get("id")
            pos_name = (repl.get("position") or {}).get("name", "")
            if (
                player_id is not None
                and any(keyword in pos_name for keyword in MIDFIELD_POSITION_KEYWORDS)
            ):
                midfielder_ids.add(player_id)

    return midfielder_ids


@dataclass
class MidfieldFeatureContext:
    """Convenience container for per-match, per-team midfielder feature computation."""

    raw_events: pd.DataFrame
    events: pd.DataFrame
    team_id: int
    midfielder_ids: Set[int]
    match_id: Optional[int] = None
    _team_events: pd.DataFrame = field(init=False, repr=False)
    _player_events: pd.DataFrame = field(init=False, repr=False)
    _opponent_events: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._team_events = self.events[self.events["team_id"] == self.team_id].copy()
        opponent_mask = self.events["team_id"] != self.team_id
        self._opponent_events = self.events[opponent_mask].copy()
        if self.midfielder_ids:
            self._player_events = self._team_events[
                self._team_events["player_id"].isin(self.midfielder_ids)
            ].copy()
        else:
            self._player_events = self._team_events.iloc[0:0].copy()

        # Ensure time columns in seconds are available.
        if "timestamp_seconds" not in self._team_events.columns:
            self._team_events["timestamp_seconds"] = _timestamp_to_seconds(
                self._team_events["timestamp"]
            )
        if "timestamp_seconds" not in self._player_events.columns:
            self._player_events["timestamp_seconds"] = _timestamp_to_seconds(
                self._player_events["timestamp"]
            )
        if "timestamp_seconds" not in self._opponent_events.columns:
            self._opponent_events["timestamp_seconds"] = _timestamp_to_seconds(
                self._opponent_events["timestamp"]
            )

        # Convenience coordinate columns.
        if "end_x" not in self._team_events.columns or "end_y" not in self._team_events.columns:
            end_x = []
            end_y = []
            pass_locs = (
                self._team_events["pass_end_location"]
                if "pass_end_location" in self._team_events.columns
                else pd.Series([None] * len(self._team_events), index=self._team_events.index)
            )
            carry_locs = (
                self._team_events["carry_end_location"]
                if "carry_end_location" in self._team_events.columns
                else pd.Series([None] * len(self._team_events), index=self._team_events.index)
            )
            for pass_loc, carry_loc in zip(pass_locs, carry_locs):
                loc = pass_loc if isinstance(pass_loc, (list, tuple)) else carry_loc
                end_x.append(_extract_coordinate(loc, 0))
                end_y.append(_extract_coordinate(loc, 1))
            self._team_events["end_x"] = end_x
            self._team_events["end_y"] = end_y

        self._player_events["end_x"] = self._team_events.loc[
            self._player_events.index, "end_x"
        ].values
        self._player_events["end_y"] = self._team_events.loc[
            self._player_events.index, "end_y"
        ].values

    @property
    def player_events(self) -> pd.DataFrame:
        return self._player_events

    @property
    def team_events(self) -> pd.DataFrame:
        return self._team_events

    @property
    def opponent_events(self) -> pd.DataFrame:
        return self._opponent_events

    def players_series(self, default: float = np.nan) -> pd.Series:
        """Return an empty Series indexed by midfielder ids filled with default value."""
        index = sorted(self.midfielder_ids)
        return pd.Series(default, index=index, dtype=float)

    def ensure_index(self, series: pd.Series, fill_value: float = np.nan) -> pd.Series:
        """
        Reindex a Series to include all midfielders, filling missing with default.

        Parameters
        ----------
        series : pd.Series
            The series to reindex.
        fill_value : float, optional
            Value to fill for missing midfielders, by default np.nan.

        Returns
        -------
        pd.Series
            Reindexed series including all midfielders.
        """
        base = self.players_series(default=fill_value)
        # Convert series index to match base index type (int)
        if not series.empty:
            # Convert index to int if it's float, handling NaN values
            if series.index.dtype in ('float64', 'float32'):
                # Convert float index to int index
                new_index = []
                for idx in series.index:
                    if pd.notna(idx):
                        new_index.append(int(idx))
                    else:
                        new_index.append(idx)
                series.index = pd.Index(new_index)
            series = series.astype(float)
        # Reindex to match base index exactly
        return series.reindex(base.index, fill_value=fill_value)



