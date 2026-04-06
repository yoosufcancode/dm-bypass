"""Stage 6: Find replacement midfielders using tactical role + scouting gradient matching."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent.parent.parent
ALL_LEAGUES = ["Spain", "England", "France", "Germany", "Italy"]


def run_replacement_analysis(
    league: str,
    team: str,
    top_n: int,
    min_matches: int = 4,   # aligned with notebook (was 10)
    progress_cb: Callable[[int, str], None] = lambda p, m: None,
) -> dict:
    """
    Full replacement analysis pipeline:
      1. Load team CSV + cross-league candidate pool
      2. Train/test split on team data, run scouting evaluation
      3. Score all players in the pool
      4. For each target-team player: match by tactical_role + position_bucket
         + bypass ceiling filter
      5. Return structured dict matching ReplacementResult schema
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from scripts.player_recommendations import (
        load_half_match,
        get_feature_cols,
        prepare_xy,
        run_scouting_evaluation,
        score_all_players,
        flag_weak_players,
        TARGET_HALF,
    )
    from sklearn.model_selection import train_test_split

    processed_dir = ROOT / "data" / "processed"
    csv_path = processed_dir / f"wyscout_{league}_features.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Features CSV not found: {csv_path}\n"
            f"Run ingestion for league '{league}' first."
        )

    # ── Step 1: Load half-match data ─────────────────────────────────────────
    progress_cb(5, f"Loading half-match data for {team} ({league})")
    df_team = load_half_match(csv_path, team=team)

    progress_cb(10, "Building cross-league candidate pool")
    league_frames = []
    for lg in ALL_LEAGUES:
        lg_path = processed_dir / f"wyscout_{lg}_features.csv"
        if not lg_path.exists():
            continue
        try:
            df_lg = load_half_match(lg_path)
            df_lg["league"] = lg
            league_frames.append(df_lg)
        except Exception:
            continue

    if not league_frames:
        raise RuntimeError("No league feature CSVs found. Run ingestion first.")

    df_all = pd.concat(league_frames, ignore_index=True)

    # ── Step 2: Train/test split on team data, scouting evaluation ───────────
    progress_cb(20, f"Running scouting evaluation on {team} data")
    feat_cols = get_feature_cols(df_team)
    X_sc, y, imp, scaler = prepare_xy(df_team, feat_cols, TARGET_HALF, min_passes=5)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.15, random_state=42
    )

    scout_result = run_scouting_evaluation(X_train, y_train, X_test, y_test)
    scouting_grads   = scout_result["scouting_grads"]
    grad_df          = scout_result["grad_df"]
    best_model_name  = scout_result["best_model_name"]
    spearman_test    = float(scout_result["spearman_test"])
    spearman_train   = float(scout_result["spearman_train"])

    if not scouting_grads:
        raise RuntimeError(
            "No scouting features passed stability + significance filters. "
            "Cannot rank players. Try providing more data."
        )

    # Build scouting_features list for the response
    scouting_features = []
    for _, row in grad_df.iterrows():
        scouting_features.append({
            "feature":         str(row["Feature"]),
            "gradient":        float(row["Gradient (dy/dx)"]),
            "direction":       str(row["scouting_direction"]),
            "p_value":         float(row["p_value"]),
            "sign_stable":     bool(row["sign_stable"]),
            "confidence_tier": str(row["confidence_tier"]),
        })

    # ── Step 3: Score all players ─────────────────────────────────────────────
    progress_cb(40, "Scoring all players in the candidate pool")
    scored = score_all_players(
        df_all, scouting_grads, team,
        min_matches=min_matches,
        min_bypasses=3.0,   # notebook: candidates must avg ≥ 3 bypasses/half
    )

    # ── Step 4: Build squad list with weak-player flagging ───────────────────
    progress_cb(55, "Building squad profile")
    team_players = scored[scored["source"] == team].copy()
    team_players = flag_weak_players(team_players)   # adds is_weak + weakness_reason
    feats = list(scouting_grads.keys())

    squad = []
    for _, row in team_players.sort_values("bypass_score").iterrows():
        avg_x = (
            float(row["average_position_x"])
            if "average_position_x" in row and pd.notna(row.get("average_position_x"))
            else None
        )
        squad.append({
            "player_name":        str(row["player_name"]),
            "tactical_role":      str(row.get("tactical_role", "Unknown")),
            "position_bucket":    str(row["position_bucket"]),
            "average_position_x": avg_x,
            "bypass_score":       float(round(row["bypass_score"], 2)),
            "halves_played":      int(row["matches_played"]),
            "bypasses_per_half":  float(round(row[TARGET_HALF], 3)),
            "is_weak":            bool(row.get("is_weak", False)),
            "weakness_reason":    str(row.get("weakness_reason", "")),
        })

    # ── Step 5: Find replacements per player ──────────────────────────────────
    progress_cb(65, f"Finding top-{top_n} replacements per player")
    candidates = scored[scored["source"] == "Other"].copy()

    # Role-level bypass ceiling from candidate pool
    role_bypass_medians = (
        candidates.groupby("tactical_role")[TARGET_HALF]
        .median()
        .to_dict()
    )

    recommendations = []

    for _, player in team_players.sort_values("bypass_score").iterrows():
        role   = player.get("tactical_role", "Unknown")
        bucket = player["position_bucket"]

        bypass_ceiling = role_bypass_medians.get(role, candidates[TARGET_HALF].median())

        # Priority 1: exact match — same role AND same position bucket
        pool = candidates[
            (candidates["tactical_role"] == role) &
            (candidates["position_bucket"] == bucket)
        ].copy()
        match_label = f"{role} + {bucket}"

        # Priority 2: same role only
        if len(pool) < top_n:
            role_only = candidates[
                (candidates["tactical_role"] == role) &
                (~candidates.index.isin(pool.index))
            ].copy()
            pool = pd.concat([pool, role_only])
            if len(role_only) > 0:
                match_label = f"{role} (pos relaxed)"

        # Priority 3: same position bucket only
        if pool.empty:
            pool = candidates[candidates["position_bucket"] == bucket].copy()
            match_label = f"{bucket} (role relaxed)"

        # Priority 4: full pool
        if pool.empty:
            pool = candidates.copy()
            match_label = "full pool"

        # Apply bypass ceiling
        pool_filtered = pool[pool[TARGET_HALF] <= bypass_ceiling].copy()
        if pool_filtered.empty:
            ceiling_relaxed = pool[TARGET_HALF].quantile(0.33)
            pool_filtered = pool[pool[TARGET_HALF] <= ceiling_relaxed].copy()
            match_label += f" [bypass ≤ {ceiling_relaxed:.1f} relaxed]"
        else:
            match_label += f" [bypass ≤ {bypass_ceiling:.1f}]"

        # Only keep candidates with lower raw bypass score than target player
        pool_filtered = pool_filtered[
            pool_filtered["raw_bypass_score"] < player["raw_bypass_score"]
        ].copy()
        pool_filtered["improvement"] = (
            player["raw_bypass_score"] - pool_filtered["raw_bypass_score"]
        )
        pool_filtered = pool_filtered.nsmallest(top_n, "raw_bypass_score")

        # Build target_player object
        target_avg_x = (
            float(player["average_position_x"])
            if "average_position_x" in player and pd.notna(player.get("average_position_x"))
            else None
        )
        target_player_obj = {
            "player_name":        str(player["player_name"]),
            "tactical_role":      str(role),
            "position_bucket":    str(bucket),
            "average_position_x": target_avg_x,
            "bypass_score":       float(round(player["bypass_score"], 2)),
            "halves_played":      int(player["matches_played"]),
            "bypasses_per_half":  float(round(player[TARGET_HALF], 3)),
        }

        # Build replacement candidates list
        replacement_list = []
        for rank, (_, row) in enumerate(pool_filtered.iterrows(), 1):
            cand_avg_x = (
                float(row["average_position_x"])
                if "average_position_x" in row and pd.notna(row.get("average_position_x"))
                else None
            )

            # Feature comparison: candidate vs target for each scouting feature
            feature_comparison = {}
            for feat in feats:
                if feat in row and feat in player:
                    feature_comparison[feat] = {
                        "candidate": float(round(row[feat], 4)) if pd.notna(row.get(feat)) else None,
                        "target":    float(round(player[feat], 4)) if pd.notna(player.get(feat)) else None,
                        "delta":     float(round(row[feat] - player[feat], 4))
                                     if pd.notna(row.get(feat)) and pd.notna(player.get(feat))
                                     else None,
                    }

            replacement_list.append({
                "rank":               rank,
                "player_name":        str(row["player_name"]),
                "team":               str(row["team_name"]),
                "league":             str(row.get("league", "—")),
                "tactical_role":      str(row.get("tactical_role", "Unknown")),
                "position_bucket":    str(row["position_bucket"]),
                "average_position_x": cand_avg_x,
                "bypass_score":       float(round(row["bypass_score"], 2)),
                "improvement":        float(round(row["improvement"], 4)),
                "bypasses_per_half":  float(round(row[TARGET_HALF], 3)),
                "feature_comparison": feature_comparison,
            })

        recommendations.append({
            "target_player": target_player_obj,
            "match_filter":  match_label,
            "replacements":  replacement_list,
        })

    progress_cb(100, "Replacement analysis complete")

    return {
        "league":           league,
        "team":             team,
        "model_selected":   best_model_name,
        "spearman_test":    round(spearman_test, 4),
        "spearman_train":   round(spearman_train, 4),
        "scouting_features": scouting_features,
        "squad":            squad,
        "recommendations":  recommendations,
    }
