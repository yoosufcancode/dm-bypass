from pydantic import BaseModel
from typing import Optional


class ReplacementRequest(BaseModel):
    league: str                   # target team's league, e.g. "England"
    team: str                     # target team name, e.g. "Manchester United"
    top_n: int = 5
    min_matches: int = 4          # minimum half-match rows — aligned with notebook MIN_MATCHES=4


class ScoutingFeature(BaseModel):
    feature: str
    gradient: float
    direction: str                # "look for LOW" or "look for HIGH"
    p_value: float
    sign_stable: bool
    confidence_tier: str


class SquadPlayer(BaseModel):
    player_name: str
    tactical_role: str
    position_bucket: str
    average_position_x: Optional[float] = None
    bypass_score: float
    halves_played: int
    bypasses_per_half: float
    is_weak: bool = False
    weakness_reason: str = ""


class ReplacementCandidate(BaseModel):
    rank: int
    player_name: str
    team: str
    league: str
    tactical_role: str
    position_bucket: str
    average_position_x: Optional[float] = None
    bypass_score: float
    improvement: float
    bypasses_per_half: float
    feature_comparison: dict = {}


class PlayerReplacementResult(BaseModel):
    target_player: SquadPlayer
    match_filter: str
    replacements: list[ReplacementCandidate]


class ReplacementResult(BaseModel):
    league: str
    team: str
    model_selected: str
    spearman_test: float
    spearman_train: float
    scouting_features: list[ScoutingFeature]
    squad: list[SquadPlayer]
    recommendations: list[PlayerReplacementResult]
