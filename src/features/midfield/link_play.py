from __future__ import annotations

import pandas as pd

from .context import MidfieldFeatureContext


def third_man_runs(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Approximate third-man runs as carries or passes immediately following a teammate pass
    that switched flank within same possession.

    A third-man run is identified when a player receives the ball after a teammate
    pass that switched at least 20 meters laterally (y-coordinate change).

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with third-man run counts.
    """
    events = ctx.team_events.sort_values("timestamp_seconds")
    counts = ctx.players_series(default=0.0)

    for _, possession in events.groupby("possession"):
        possession = possession.reset_index(drop=True)
        for i in range(1, len(possession)):
            current = possession.loc[i]
            prev = possession.loc[i - 1]
            if prev["team_id"] != ctx.team_id or current["team_id"] != ctx.team_id:
                continue
            if prev["type_name"] != "Pass":
                continue
            prev_y = prev["y"]
            curr_y = current["y"]
            if pd.notna(prev_y) and pd.notna(curr_y):
                if abs(prev_y - curr_y) >= 20:
                    player_id = current["player_id"]
                    if player_id in counts.index:
                        counts.loc[player_id] += 1.0
    return counts


def wall_pass_events(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count wall pass events (give-and-go sequences) for each midfielder.

    A wall pass is identified as: Player A passes, Player B receives/passes/carries,
    then Player A receives again, all within 3 seconds in the same possession.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with wall pass event counts.
    """
    events = ctx.team_events.sort_values("timestamp_seconds")
    counts = ctx.players_series(default=0.0)

    for _, possession in events.groupby("possession"):
        possession = possession.reset_index(drop=True)
        for i in range(len(possession) - 2):
            first = possession.loc[i]
            second = possession.loc[i + 1]
            third = possession.loc[i + 2]
            if (
                first["team_id"] == ctx.team_id
                and second["team_id"] == ctx.team_id
                and third["team_id"] == ctx.team_id
            ):
                if (
                    first["player_id"] == third["player_id"]
                    and first["player_id"] in counts.index
                    and first["type_name"] == "Pass"
                    and second["type_name"] in {"Pass", "Carry"}
                    and third["timestamp_seconds"] - first["timestamp_seconds"] <= 3
                ):
                    counts.loc[first["player_id"]] += 1.0
    return counts

