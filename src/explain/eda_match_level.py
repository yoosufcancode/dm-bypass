"""
Exploratory Data Analysis: Match-Level Composite Features vs Bypasses

This script is structured for notebook execution - each section can be run as a separate cell.
Copy each cell into a Jupyter notebook cell for interactive analysis.
"""

# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    print("✓ Visualization libraries loaded")
except ImportError:
    HAS_VIZ = False
    print("⚠ Warning: matplotlib/seaborn not available. Visualizations will be skipped.")


# ============================================================================
# CELL 2: Load Data
# ============================================================================
csv_path = Path("data/processed/Barcelona_2014_2015_features.csv")
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} rows from {csv_path}")
print(f"Unique matches: {df['match_id'].nunique()}")
print(f"Unique players: {df['player_id'].nunique()}")
print(f"Columns: {len(df.columns)}")
print(f"\nFirst few rows:")
df.head()


# ============================================================================
# CELL 3: Define Aggregation Strategy
# ============================================================================
# Metadata columns
metadata_cols = ['match_id', 'team_id', 'team_name', 'season', 'bypasses_per_match']

# Feature columns (exclude metadata)
feature_cols = [c for c in df.columns if c not in metadata_cols and 
               c not in ['player_id', 'player_name', 'midfielder_type', 'computed_at']]

# Define aggregation strategy
# Counts/volumes: sum
count_features = [
    'possessions_involved', 'passes_attempted', 'progressive_passes',
    'final_third_entries_by_pass', 'key_passes', 'carries_attempted',
    'progressive_carries', 'successful_dribbles', 'carries_leading_to_shot',
    'carries_leading_to_key_pass', 'final_third_carries', 'penalty_area_carries',
    'pressures_applied', 'ball_recoveries', 'interceptions', 'tackles_won',
    'counterpress_actions', 'blocked_passes', 'blocked_shots',
    'clearance_followed_by_recovery', 'zone_entries', 'pressured_touches',
    'third_man_runs', 'wall_pass_events', 'shot_creating_actions',
    'fouls_committed', 'fouls_suffered', 'tactical_fouls', 'advantage_fouls_won',
    'set_piece_involvements', 'set_piece_duels_won', 'defensive_set_piece_clearances',
    'line_breaking_receipts', 'zone14_touches', 'penalty_area_deliveries',
    'switches_completed', 'aerial_duels_contested', 'fifty_fiftys_won',
    'sliding_tackles', 'ball_receipts_total', 'central_lane_receipts',
    'one_touch_passes', 'secondary_shot_assists', 'expected_assists'
]

# Rates/percentages: mean
rate_features = [
    'pass_completion_rate', 'under_pressure_pass_share', 'pressured_carry_success_rate',
    'press_to_interception_chain', 'pressure_to_self_recovery',
    'pressures_to_turnover_rate', 'pressured_touch_retention_rate',
    'corner_delivery_accuracy', 'cross_accuracy', 'aerial_duel_win_rate',
    'sliding_tackle_success_rate', 'weak_foot_pass_share',
    'pressured_retention_rate'
]

# Totals/distances: sum
total_features = [
    'possession_time_seconds', 'carry_distance_total', 'expected_threat_added',
    'xg_chain'
]

# Averages: mean
avg_features = [
    'tempo_index', 'average_position_x', 'average_position_y',
    'width_variance'
]

# Special: turnovers (sum)
special_features = ['turnovers']

print(f"Count features: {len(count_features)}")
print(f"Rate features: {len(rate_features)}")
print(f"Total features: {len(total_features)}")
print(f"Average features: {len(avg_features)}")


# ============================================================================
# CELL 4: Aggregate Features Per Match
# ============================================================================
# Build aggregation dictionary
agg_dict = {}

# Metadata: take first (same for all players in match)
for col in ['team_id', 'team_name', 'season', 'bypasses_per_match']:
    agg_dict[col] = 'first'

# Count features: sum
for col in count_features:
    if col in feature_cols:
        agg_dict[col] = 'sum'

# Rate features: mean
for col in rate_features:
    if col in feature_cols:
        agg_dict[col] = 'mean'

# Total features: sum
for col in total_features:
    if col in feature_cols:
        agg_dict[col] = 'sum'

# Average features: mean
for col in avg_features:
    if col in feature_cols:
        agg_dict[col] = 'mean'

# Special features
for col in special_features:
    if col in feature_cols:
        agg_dict[col] = 'sum'

# Handle any remaining features with mean
for col in feature_cols:
    if col not in agg_dict:
        agg_dict[col] = 'mean'

# Group by match and aggregate
match_df = df.groupby('match_id').agg(agg_dict).reset_index()

# Add number of midfielders per match
midfielder_counts = df.groupby('match_id')['player_id'].count().reset_index()
midfielder_counts.columns = ['match_id', 'num_midfielders']
match_df = match_df.merge(midfielder_counts, on='match_id')

print(f"Aggregated to {len(match_df)} matches")
print(f"Match-level features shape: {match_df.shape}")
print(f"\nFirst few rows:")
match_df.head()


# ============================================================================
# CELL 5: Basic Statistics
# ============================================================================
print("="*80)
print("BASIC STATISTICS")
print("="*80)

print(f"\nNumber of matches: {len(match_df)}")
print(f"\nBypasses per match:")
print(match_df['bypasses_per_match'].describe())

print(f"\nNumber of midfielders per match:")
print(match_df['num_midfielders'].describe())

print(f"\nBypasses distribution:")
print(f"  Mean: {match_df['bypasses_per_match'].mean():.2f}")
print(f"  Std: {match_df['bypasses_per_match'].std():.2f}")
print(f"  Min: {match_df['bypasses_per_match'].min()}")
print(f"  Max: {match_df['bypasses_per_match'].max()}")
print(f"  Median: {match_df['bypasses_per_match'].median():.2f}")
print(f"  Q1: {match_df['bypasses_per_match'].quantile(0.25):.2f}")
print(f"  Q3: {match_df['bypasses_per_match'].quantile(0.75):.2f}")


# ============================================================================
# CELL 6: Correlation Analysis
# ============================================================================
print("="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Select numeric feature columns (exclude metadata)
feature_cols_for_corr = [c for c in match_df.columns 
                        if c not in ['match_id', 'team_id', 'team_name', 'season', 
                                    'num_midfielders', 'bypasses_per_match']]

# Calculate correlations with bypasses
correlations = {}
for col in feature_cols_for_corr:
    if match_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        corr = match_df[col].corr(match_df['bypasses_per_match'])
        if not np.isnan(corr):
            correlations[col] = corr

# Sort by absolute correlation
sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\nTotal features analyzed: {len(correlations)}")
print(f"\nTop 20 features most correlated with bypasses_per_match:")
print(f"{'Feature':<40} {'Correlation':>12}")
print("-" * 54)
for feature, corr in sorted_corrs[:20]:
    print(f"{feature:<40} {corr:>12.4f}")

print(f"\nTop 10 positive correlations:")
positive_corrs = [(f, c) for f, c in sorted_corrs if c > 0][:10]
for feature, corr in positive_corrs:
    print(f"  {feature:<40} {corr:>12.4f}")

print(f"\nTop 10 negative correlations:")
negative_corrs = [(f, c) for f, c in sorted_corrs if c < 0][:10]
for feature, corr in negative_corrs:
    print(f"  {feature:<40} {corr:>12.4f}")


# ============================================================================
# CELL 7: Feature Category Analysis
# ============================================================================
print("="*80)
print("FEATURE CATEGORY ANALYSIS")
print("="*80)

categories = {
    'Possession & Tempo': ['possessions_involved', 'possession_time_seconds', 'tempo_index', 'turnovers'],
    'Passing': ['passes_attempted', 'pass_completion_rate', 'progressive_passes', 
               'final_third_entries_by_pass', 'key_passes', 'under_pressure_pass_share'],
    'Carrying': ['carries_attempted', 'progressive_carries', 'carry_distance_total',
                'successful_dribbles', 'final_third_carries', 'penalty_area_carries'],
    'Defensive Actions': ['pressures_applied', 'ball_recoveries', 'interceptions',
                        'tackles_won', 'blocked_passes', 'blocked_shots'],
    'Pressure & Recovery': ['press_to_interception_chain', 'counterpress_actions',
                          'pressure_to_self_recovery', 'pressures_to_turnover_rate'],
    'Spatial': ['average_position_x', 'average_position_y', 'width_variance', 'zone_entries'],
    'Attacking Creation': ['shot_creating_actions', 'expected_threat_added',
                        'secondary_shot_assists', 'expected_assists', 'xg_chain'],
    'Set Pieces': ['set_piece_involvements', 'corner_delivery_accuracy',
                  'set_piece_duels_won', 'defensive_set_piece_clearances'],
    'Progression': ['line_breaking_receipts', 'zone14_touches', 'penalty_area_deliveries',
                   'switches_completed', 'cross_accuracy'],
    'Duels': ['aerial_duels_contested', 'aerial_duel_win_rate', 'fifty_fiftys_won',
             'sliding_tackles', 'sliding_tackle_success_rate'],
    'Receiving': ['ball_receipts_total', 'central_lane_receipts', 'one_touch_passes',
                 'weak_foot_pass_share'],
    'Discipline': ['fouls_committed', 'fouls_suffered', 'tactical_fouls', 'advantage_fouls_won']
}

category_corrs = {}
for category, features in categories.items():
    cat_corrs = []
    for feat in features:
        if feat in correlations:
            cat_corrs.append(abs(correlations[feat]))
    if cat_corrs:
        category_corrs[category] = {
            'mean_abs_corr': np.mean(cat_corrs),
            'max_abs_corr': np.max(cat_corrs),
            'features': len(cat_corrs)
        }

print(f"\n{'Category':<30} {'Mean |Corr|':>12} {'Max |Corr|':>12} {'Features':>10}")
print("-" * 66)
for cat, stats in sorted(category_corrs.items(), key=lambda x: x[1]['mean_abs_corr'], reverse=True):
    print(f"{cat:<30} {stats['mean_abs_corr']:>12.4f} {stats['max_abs_corr']:>12.4f} {stats['features']:>10}")


# ============================================================================
# CELL 8: High vs Low Bypass Match Comparison
# ============================================================================
print("="*80)
print("HIGH vs LOW BYPASS MATCH COMPARISON")
print("="*80)

median_bypasses = match_df['bypasses_per_match'].median()
high_bypass = match_df[match_df['bypasses_per_match'] >= median_bypasses]
low_bypass = match_df[match_df['bypasses_per_match'] < median_bypasses]

print(f"\nMedian bypasses: {median_bypasses:.0f}")
print(f"High bypass matches (>= {median_bypasses:.0f}): {len(high_bypass)} matches")
print(f"Low bypass matches (< {median_bypasses:.0f}): {len(low_bypass)} matches")

# Compare key features between high and low bypass matches
key_features = [
    'passes_attempted', 'pass_completion_rate', 'progressive_passes',
    'carries_attempted', 'pressures_applied', 'ball_recoveries',
    'interceptions', 'possession_time_seconds', 'tempo_index',
    'shot_creating_actions', 'zone_entries'
]

print(f"\n{'Feature':<30} {'High Bypass':>15} {'Low Bypass':>15} {'Difference':>12}")
print("-" * 72)
for feat in key_features:
    if feat in match_df.columns:
        high_mean = high_bypass[feat].mean()
        low_mean = low_bypass[feat].mean()
        diff = high_mean - low_mean
        print(f"{feat:<30} {high_mean:>15.2f} {low_mean:>15.2f} {diff:>12.2f}")


# ============================================================================
# CELL 9: Visualization - Bypasses Distribution
# ============================================================================
if HAS_VIZ:
    plt.figure(figsize=(10, 6))
    plt.hist(match_df['bypasses_per_match'], bins=15, edgecolor='black', alpha=0.7)
    plt.xlabel('Bypasses per Match', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Bypasses per Match (39 matches)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping visualization (matplotlib not available)")


# ============================================================================
# CELL 10: Visualization - Top Correlations
# ============================================================================
if HAS_VIZ:
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    features = [f[0] for f in sorted_corrs]
    corr_values = [f[1] for f in sorted_corrs]
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if c < 0 else 'green' for c in corr_values]
    plt.barh(range(len(features)), corr_values, color=colors, alpha=0.7)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Correlation with Bypasses per Match', fontsize=12)
    plt.title('Top 15 Features Correlated with Bypasses per Match', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
else:
    print("Skipping visualization (matplotlib not available)")


# ============================================================================
# CELL 11: Visualization - Scatter Plots for Top Features
# ============================================================================
if HAS_VIZ:
    top_features = [f[0] for f in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:6]]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feat in enumerate(top_features):
        if feat in match_df.columns:
            axes[idx].scatter(match_df[feat], match_df['bypasses_per_match'], 
                            alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
            axes[idx].set_xlabel(feat, fontsize=10)
            axes[idx].set_ylabel('Bypasses per Match', fontsize=10)
            axes[idx].set_title(f'{feat}\n(r={correlations[feat]:.3f})', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(match_df[feat].dropna(), 
                          match_df.loc[match_df[feat].notna(), 'bypasses_per_match'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(match_df[feat].min(), match_df[feat].max(), 100)
            axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    plt.suptitle('Top 6 Features vs Bypasses per Match', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping visualization (matplotlib not available)")


# ============================================================================
# CELL 12: Visualization - Correlation Heatmap
# ============================================================================
if HAS_VIZ:
    key_features = [
        'bypasses_per_match', 'passes_attempted', 'pass_completion_rate',
        'progressive_passes', 'carries_attempted', 'pressures_applied',
        'ball_recoveries', 'interceptions', 'possession_time_seconds',
        'tempo_index', 'shot_creating_actions', 'zone_entries'
    ]
    
    available_features = [f for f in key_features if f in match_df.columns]
    corr_matrix = match_df[available_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Key Features vs Bypasses', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping visualization (matplotlib not available)")


# ============================================================================
# CELL 13: Save Aggregated Match Data
# ============================================================================
output_dir = Path("data/explain/match_level_eda")
output_dir.mkdir(parents=True, exist_ok=True)
output_csv = output_dir / "match_level_features.csv"
match_df.to_csv(output_csv, index=False)
print(f"Saved aggregated match data to: {output_csv}")
print(f"Shape: {match_df.shape}")


# ============================================================================
# CELL 14: Additional Analysis - Match-by-Match Breakdown
# ============================================================================
print("="*80)
print("MATCH-BY-MATCH BREAKDOWN")
print("="*80)

# Show matches with highest and lowest bypasses
print("\nTop 5 matches with highest bypasses:")
top_matches = match_df.nlargest(5, 'bypasses_per_match')[['match_id', 'bypasses_per_match', 'num_midfielders']]
print(top_matches.to_string(index=False))

print("\nTop 5 matches with lowest bypasses:")
bottom_matches = match_df.nsmallest(5, 'bypasses_per_match')[['match_id', 'bypasses_per_match', 'num_midfielders']]
print(bottom_matches.to_string(index=False))


# ============================================================================
# CELL 15: Summary Statistics Table
# ============================================================================
print("="*80)
print("SUMMARY STATISTICS TABLE")
print("="*80)

# Create summary table for key features
summary_features = [
    'bypasses_per_match', 'num_midfielders', 'passes_attempted', 
    'pass_completion_rate', 'progressive_passes', 'carries_attempted',
    'pressures_applied', 'ball_recoveries', 'interceptions',
    'possession_time_seconds', 'tempo_index', 'shot_creating_actions'
]

summary_df = match_df[summary_features].describe()
print("\nSummary statistics for key features:")
print(summary_df.round(2))
