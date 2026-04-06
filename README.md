# DM-Bypass — Midfield Bypass Prediction & Replacement Scouting

A research-grade football analytics pipeline that quantifies how often a team's midfield is bypassed, identifies which individual midfielders contribute to that vulnerability, and surfaces cross-league replacement candidates using gradient-weighted player scoring.

---

## Research Context

**Bypass** — an opponent action (through-ball, carry, or long pass) that advances the ball from behind the midfield line to the space in front of it, effectively removing the midfield defensive block in a single move.

The core thesis: if a scouting model can reliably rank matches by their bypass concession rate using only **player-inherent spatial and skill features** (zone coverage, interceptions, tempo, passing style), it can be inverted as a scouting direction — "look for players with LOW values on these features" — to identify structurally similar replacements from other leagues.

### Data Source

All event data comes from the **Wyscout Open Dataset** (2017/18 season, five top European leagues). The Barcelona multi-season analysis uses the Wyscout Spain dataset for seasons 2012/13 through 2015/16.

---

## Project Structure

```
dm-bypass/
│
├── data/
│   ├── raw/                        # Wyscout JSON event files (gitignored)
│   │   └── events/{league}/{team}/
│   └── processed/                  # Feature CSVs, gradient analysis outputs
│
├── notebooks/
│   ├── EDA2.ipynb                  # Exploratory data analysis on match-level features
│   ├── feature_selection.ipynb     # 4-method consensus feature selection
│   ├── model_building.ipynb        # MLR / Ridge / Lasso with temporal split
│   └── Best_player.ipynb           # Bayesian-shrinkage player scoring + replacement finder
│
├── scripts/
│   ├── model.py                    # Model builders (build_mlr, build_ridge, build_lasso, LOOCV)
│   ├── scouting_evaluation.py      # Two-model scouting evaluation (predictive + gradient Ridge)
│   ├── player_recommendations.py   # End-to-end: load → cluster → score → recommend
│   ├── process_barcelona_3seasons.py
│   ├── process_barcelona_all_seasons.py
│   ├── process_all_teams.py
│   ├── run_pipeline.py             # CLI entry point for batch processing
│   └── download_wyscout.py
│
├── src/
│   └── features/
│       ├── main_feature.py         # Top-level feature orchestrator
│       └── midfield/               # 17 feature modules (one per category)
│           ├── passing.py
│           ├── spatial.py
│           ├── defensive.py
│           ├── defensive_phase.py
│           ├── duels.py
│           ├── carrying.py
│           ├── progression.py
│           ├── possession_tempo.py
│           ├── pressure_resistance.py
│           ├── link_play.py
│           ├── receiving.py
│           ├── set_pieces.py
│           ├── shot_creation.py
│           ├── attacking_creation.py
│           ├── discipline.py
│           ├── context.py          # Opponent context features (opp_ppda, opp_formation, …)
│           └── independent_var.py  # Target: bypasses_per_halftime
│
├── production/
│   └── backend/                    # FastAPI 6-stage pipeline API
│       ├── api/
│       ├── schemas/
│       ├── services/
│       └── workers/
│
├── models/                         # Saved .pkl model files (gitignored)
├── requirements.txt                # Research / notebook dependencies
└── events_data.md                  # Field-by-field Wyscout event schema reference
```

---

## Feature Engineering

Features are computed per **player per match-half** from raw Wyscout event sequences. There are **~80 features** across 17 modules before selection.

### Feature Categories

| Module | Example Features |
|--------|-----------------|
| **Spatial** | `midfield_zone_coverage_x`, `width_variance`, `average_position_x/y` |
| **Passing** | `pass_completion_rate`, `under_pressure_pass_share`, `weak_foot_pass_share`, `tempo_index` |
| **Defensive** | `interceptions`, `bypass_channel_defensive_actions`, `avg_defensive_x_on_deep_opp` |
| **Defensive Phase** | `midfield_presence_on_deep_opp`, `defensive_shape_compactness`* |
| **Duels** | `duel_win_rate`, `aerial_duel_win_rate` |
| **Carrying** | `progressive_carries`, `carry_distance` |
| **Progression** | `final_third_entries_by_pass`, `progressive_passes` |
| **Possession/Tempo** | `possessions_involved`, `possession_time_seconds` |
| **Pressure Resistance** | `passes_under_pressure`, `turnovers` |
| **Opponent Context** | `opp_ppda`, `opp_long_ball_rate`, `opp_avg_pass_length`, `score_diff_start_of_half` |

\* `defensive_shape_compactness` and `zone14_touches` are dropped before feature selection — they are team structural metrics or positional role artefacts, not player-inherent causal signals.

### Target Variable

- **`bypasses_per_halftime`** — count of opponent bypasses in that half-match (sum across the three midfielders in team-level analysis, or individual count for player-level scoring)

---

## Analysis Pipeline (Notebooks)

The four notebooks run in sequence and build on each other.

### 1. `EDA2.ipynb` — Exploratory Analysis

- Loads match-level aggregated features for Barcelona (39 matches)
- Descriptive statistics, missing value audit, correlation heatmap
- Distribution of `bypasses_per_halftime`
- Identifies feature pairs with high collinearity

### 2. `feature_selection.ipynb` — 4-Method Consensus

Aggregates the player-half data to **one row per match-half** (all three midfielders combined using semantic rules), then runs:

| Method | Implementation |
|--------|----------------|
| Univariate | `SelectKBest` with `f_regression` + `mutual_info_regression` + absolute Pearson correlation, combined as `(F_norm + MI_norm + |Corr|) / 3` |
| Random Forest | `RandomForestRegressor(n_estimators=100, random_state=42)` feature importances |
| RFE | `RFE(RandomForestRegressor(n_estimators=50), n_features_to_select=N_FEATURES, step=1)` |
| **Consensus** | `selection_count × 0.4 + combined_univariate × 0.3 + rf_norm × 0.3` |

Where `selection_count` is how many of the three methods (Univariate top-N, RF top-N, RFE support) selected that feature.

**Output** — 15 selected features, e.g.:
```
midfield_zone_coverage_x, possessions_involved, average_position_x,
penalty_area_deliveries, avg_defensive_x_on_deep_opp, interceptions,
opp_ppda, bypass_channel_defensive_actions, opp_avg_pass_length,
final_third_entries_by_pass, width_variance, tempo_index,
average_position_y, under_pressure_pass_share, weak_foot_pass_share
```

Exports both a **match-level dataset** (for model training) and a **player-level dataset** (for scouting scoring).

### 3. `model_building.ipynb` — Regression Models

- **Temporal train/test split**: seasons 2012/13–2014/15 → train; 2015/16 → test (no data leakage)
- StandardScaler fit on training data only
- Three models: `LinearRegression`, `RidgeCV(alphas=logspace(-3,3,50))`, `LassoCV`
- Feature reduction analysis: tests counts [5, 6, 7, 8, 10, 12, 15] to find optimal N by CV RMSE
- Final model: Ridge on best N features
- Evaluation: Spearman ρ (primary), R², RMSE, MAE — both 5-fold CV and temporal test set
- Calls `scripts/scouting_evaluation.py` → `evaluate_scouting_model()` to produce `mlr_gradient_analysis.csv`

**Scouting evaluation two-model approach:**
1. **Predictive model** (Ridge on selected features) — evaluated by Spearman ρ, tells us how well the model ranks matches
2. **Scouting Ridge** (Ridge on ALL player features, opponent context excluded) — produces gradient weights `∂bypasses/∂feature` with p-values and sign stability across CV folds; these weights become the scouting directions

### 4. `Best_player.ipynb` — Player Scoring & Replacement

Loads `mlr_gradient_analysis.csv` and the player-level selected-feature datasets.

**Scoring pipeline:**

1. **Aggregate** — one row per player across all available seasons (mean of features, count of match-halves)
2. **Bayesian shrinkage** — adjusts feature estimates toward the pool mean proportional to sample size:
   ```
   adjusted = (n × observed + k × pool_mean) / (n + k)
   k = pool median of matches_played  (≈ 4.0 in Barcelona data)
   ```
   Players with few halves are shrunk toward average; players with many halves retain their observed values.
3. **Z-score** — standardise each shrunk feature across the pool (ddof=1)
4. **Bypass score** — `z_features @ gradient_vector`, then converted to pool percentile × 100
   - Lower percentile = better (fewer bypasses predicted by that player's feature profile)
5. **Weak-player flagging** — target-team players above their position-type median bypass score are flagged `WEAK`
6. **Replacement ranking** — candidates from "Other" teams filtered by `min_matches ≥ 4` and `bypasses_per_halftime ≥ 3`, sorted by `raw_bypass_score` (gradient-weighted composite, not percentile)

---

## Production API

The `production/backend/` directory contains a **FastAPI** implementation of the same pipeline, designed to be driven from a web frontend.

See **[production/backend/README.md](production/backend/README.md)** for full API reference, installation, and configuration.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r production/backend/requirements_production.txt

# Run from the project root
uvicorn production.backend.api.main:app --reload --port 8000
```

| URL | |
|-----|-|
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/static/index.html` | Frontend |

---

## Running the Research Scripts

### Process a single league

```bash
python scripts/process_all_teams.py --league Spain
```

### Run the full Barcelona multi-season pipeline

```bash
python scripts/process_barcelona_all_seasons.py
```

### Get player recommendations (CLI)

```bash
python scripts/player_recommendations.py --league Spain --team Barcelona --top-n 5
```

### Run model building

```bash
# From project root, after processing Barcelona features
python -c "
import sys; sys.path.insert(0, '.')
from scripts.scouting_evaluation import evaluate_scouting_model
# ... (see model_building.ipynb for full invocation)
"
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Half-match granularity** | Doubles the effective sample size vs full-match; captures tactical half-time adjustments |
| **Team-level aggregation for training** | Bypass concession is a collective midfield property; aggregating the trio gives cleaner signal than individual rows |
| **Player-level scoring with Bayesian shrinkage** | Scouting must operate on individual profiles; shrinkage prevents low-sample players from dominating rankings |
| **Temporal train/test split** | Avoids data leakage across seasons; a random split would let future matches inform past predictions |
| **Scouting Ridge (not Lasso) for gradient weights** | Ridge retains all features with non-zero coefficients; Lasso arbitrarily zeroes correlated features, producing sparse but unstable scouting profiles |
| **RFE with fixed `n_features_to_select`** | Ensures the selected set is exactly N features; RFECV auto-selects optimally but ignores the interpretability budget |
| **Consensus scoring: `count×0.4 + univariate×0.3 + rf×0.3`** | Rewards features that appear across multiple independent methods; the count weight rewards multi-method consensus over any single method's top score |
| **`min_matches = 4`** | Minimum 4 half-match observations (≈ 2 full matches) to include a candidate; balances data reliability against pool size |

---

## Requirements

```
Python 3.11+
pandas, numpy, scikit-learn, scipy, statsmodels
fastapi, uvicorn, pydantic, pydantic-settings
```

Full lists:
- Research pipeline: `requirements.txt`
- Production API: `production/backend/requirements_production.txt`
