"""
Exploratory Data Analysis (EDA) for Barcelona 2014/2015 Midfielder Features

This script performs comprehensive EDA on the processed features dataset,
structured as notebook cells for easy execution and understanding.
"""

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
    # Set style for better visualizations
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
except ImportError:
    HAS_VIZ = False
    print("Warning: matplotlib/seaborn not available. Visualizations will be skipped.")
    print("Install with: pip install matplotlib seaborn")

# ============================================================================
# CELL 1: Data Loading and Basic Information
# ============================================================================

def load_data():
    """Load the features CSV file."""
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "Barcelona_2014_2015_features.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print("=" * 80)
    print("CELL 1: DATA LOADING AND BASIC INFORMATION")
    print("=" * 80)
    print(f"\n✓ Data loaded successfully from: {data_path}")
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return df

# ============================================================================
# CELL 2: Data Types and Missing Values
# ============================================================================

def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    print("\n" + "=" * 80)
    print("CELL 2: MISSING VALUES ANALYSIS")
    print("=" * 80)
    
    # Identify metadata vs feature columns
    metadata_cols = ['player_id', 'player_name', 'midfielder_type', 'match_id', 
                     'team_id', 'team_name', 'season', 'computed_at']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"\nMetadata columns: {len(metadata_cols)}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Missing values analysis
    missing = df[feature_cols].isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    print("\nMissing Values Summary:")
    print("-" * 80)
    missing_nonzero = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_nonzero) > 0:
        print(missing_nonzero.to_string())
    else:
        print("  ✓ No missing values found!")
    
    # Visualize missing values
    if HAS_VIZ and len(missing_nonzero) > 0:
        plt.figure(figsize=(12, 8))
        missing_nonzero['Missing %'].plot(kind='barh')
        plt.xlabel('Missing Percentage (%)')
        plt.title('Missing Values by Feature')
        plt.tight_layout()
        plt.savefig(Path(__file__).parent.parent.parent / "data" / "processed" / "eda_missing_values.png", dpi=150)
        print(f"\n✓ Missing values plot saved to: data/processed/eda_missing_values.png")
        plt.close()
    elif not HAS_VIZ and len(missing_nonzero) > 0:
        print("\n⚠ Visualization skipped (matplotlib not available)")
    
    return metadata_cols, feature_cols

# ============================================================================
# CELL 3: Descriptive Statistics
# ============================================================================

def descriptive_statistics(df, feature_cols):
    """Generate descriptive statistics for all features."""
    print("\n" + "=" * 80)
    print("CELL 3: DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    # Basic statistics
    numeric_features = df[feature_cols].select_dtypes(include=[np.number])
    
    print(f"\nNumeric Features: {len(numeric_features.columns)}")
    print("\nSummary Statistics:")
    print("-" * 80)
    stats = numeric_features.describe().T
    stats['range'] = stats['max'] - stats['min']
    stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']]
    print(stats.round(2).to_string())
    
    # Save to CSV
    stats_path = Path(__file__).parent.parent.parent / "data" / "processed" / "eda_descriptive_stats.csv"
    stats.to_csv(stats_path)
    print(f"\n✓ Full statistics saved to: {stats_path}")
    
    return numeric_features

# ============================================================================
# CELL 4: Midfielder Type Analysis
# ============================================================================

def analyze_midfielder_types(df):
    """Analyze distribution and characteristics by midfielder type."""
    print("\n" + "=" * 80)
    print("CELL 4: MIDFIELDER TYPE ANALYSIS")
    print("=" * 80)
    
    type_names = {
        0: 'Defensive Midfield',
        1: 'Center Midfield',
        2: 'Attacking Midfield',
        3: 'Wing Midfield',
        4: 'Right Wing',
        5: 'Left Wing',
        6: 'Wing Back',
        7: 'Midfield (Generic)'
    }
    
    df['midfielder_type_name'] = df['midfielder_type'].map(type_names)
    
    print("\nMidfielder Type Distribution:")
    print("-" * 80)
    type_counts = df['midfielder_type_name'].value_counts().sort_index()
    for type_code, count in type_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {type_code:30s}: {count:3d} ({pct:5.1f}%)")
    
    # Key features by type
    key_features = ['passes_attempted', 'pass_completion_rate', 'carries_attempted',
                    'pressures_applied', 'ball_recoveries', 'average_position_x', 
                    'average_position_y']
    
    print("\n\nAverage Key Features by Midfielder Type:")
    print("-" * 80)
    type_means = df.groupby('midfielder_type_name')[key_features].mean().round(2)
    print(type_means.to_string())
    
    # Visualization
    if not HAS_VIZ:
        print("\n⚠ Visualization skipped (matplotlib not available)")
        return type_names
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Type distribution
    type_counts.plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Midfielder Type Distribution')
    axes[0, 0].set_xlabel('Midfielder Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Passes by type
    df.boxplot(column='passes_attempted', by='midfielder_type_name', ax=axes[0, 1])
    axes[0, 1].set_title('Passes Attempted by Type')
    axes[0, 1].set_xlabel('')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Position heatmap
    position_data = df.groupby('midfielder_type_name')[['average_position_x', 'average_position_y']].mean()
    axes[1, 0].scatter(position_data['average_position_x'], position_data['average_position_y'], 
                       s=200, alpha=0.6)
    for idx, row in position_data.iterrows():
        axes[1, 0].annotate(idx, (row['average_position_x'], row['average_position_y']))
    axes[1, 0].set_xlabel('Average Position X')
    axes[1, 0].set_ylabel('Average Position Y')
    axes[1, 0].set_title('Average Position by Midfielder Type')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pressures by type
    df.boxplot(column='pressures_applied', by='midfielder_type_name', ax=axes[1, 1])
    axes[1, 1].set_title('Pressures Applied by Type')
    axes[1, 1].set_xlabel('')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent.parent / "data" / "processed" / "eda_midfielder_types.png", dpi=150)
    print(f"\n✓ Midfielder type analysis plot saved to: data/processed/eda_midfielder_types.png")
    plt.close()
    
    return type_names

# ============================================================================
# CELL 5: Feature Distributions
# ============================================================================

def analyze_feature_distributions(df, feature_cols):
    """Analyze distributions of key features."""
    print("\n" + "=" * 80)
    print("CELL 5: FEATURE DISTRIBUTIONS")
    print("=" * 80)
    
    key_features = ['passes_attempted', 'pass_completion_rate', 'carries_attempted',
                    'pressures_applied', 'ball_recoveries', 'possessions_involved',
                    'possession_time_seconds', 'progressive_passes', 'progressive_carries']
    
    available_features = [f for f in key_features if f in feature_cols]
    
    # Create distribution plots
    if not HAS_VIZ:
        print("\n⚠ Visualization skipped (matplotlib not available)")
        return
    
    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(available_features):
        ax = axes[idx]
        data = df[feature].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent.parent / "data" / "processed" / "eda_feature_distributions.png", dpi=150)
    print(f"\n✓ Feature distributions plot saved to: data/processed/eda_feature_distributions.png")
    plt.close()
    
    # Print skewness and kurtosis
    print("\nFeature Skewness and Kurtosis:")
    print("-" * 80)
    skew_kurt = pd.DataFrame({
        'Skewness': [df[f].skew() for f in available_features],
        'Kurtosis': [df[f].kurtosis() for f in available_features]
    }, index=available_features)
    print(skew_kurt.round(2).to_string())

# ============================================================================
# CELL 6: Correlation Analysis
# ============================================================================

def analyze_correlations(df, feature_cols):
    """Analyze correlations between features."""
    print("\n" + "=" * 80)
    print("CELL 6: CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Select numeric features
    numeric_features = df[feature_cols].select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_features.corr()
    
    # Find highly correlated pairs
    print("\nHighly Correlated Feature Pairs (|r| > 0.7):")
    print("-" * 80)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7 and not np.isnan(corr_val):
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
        high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        print("  No highly correlated pairs found (|r| > 0.7)")
    
    # Visualize correlation matrix for key features
    key_features = ['passes_attempted', 'pass_completion_rate', 'carries_attempted',
                    'pressures_applied', 'ball_recoveries', 'possessions_involved',
                    'possession_time_seconds', 'progressive_passes', 'progressive_carries',
                    'average_position_x', 'average_position_y']
    
    available_key = [f for f in key_features if f in numeric_features.columns]
    
    if len(available_key) > 1:
        key_corr = numeric_features[available_key].corr()
        
        if HAS_VIZ:
            plt.figure(figsize=(14, 12))
            mask = np.triu(np.ones_like(key_corr, dtype=bool))
            sns.heatmap(key_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                        center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix: Key Features')
            plt.tight_layout()
            plt.savefig(Path(__file__).parent.parent.parent / "data" / "processed" / "eda_correlation_matrix.png", dpi=150)
            print(f"\n✓ Correlation matrix plot saved to: data/processed/eda_correlation_matrix.png")
            plt.close()
        else:
            print("\n⚠ Visualization skipped (matplotlib not available)")
    
    # Save full correlation matrix
    corr_path = Path(__file__).parent.parent.parent / "data" / "processed" / "eda_correlation_matrix.csv"
    corr_matrix.to_csv(corr_path)
    print(f"✓ Full correlation matrix saved to: {corr_path}")

# ============================================================================
# CELL 7: Feature Relationships
# ============================================================================

def analyze_feature_relationships(df):
    """Analyze relationships between key features."""
    print("\n" + "=" * 80)
    print("CELL 7: FEATURE RELATIONSHIPS")
    print("=" * 80)
    
    # Scatter plots for key relationships
    if not HAS_VIZ:
        print("\n⚠ Visualization skipped (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Passes vs Carries
    axes[0, 0].scatter(df['passes_attempted'], df['carries_attempted'], 
                       alpha=0.6, s=50, c=df['midfielder_type'], cmap='viridis')
    axes[0, 0].set_xlabel('Passes Attempted')
    axes[0, 0].set_ylabel('Carries Attempted')
    axes[0, 0].set_title('Passes vs Carries (colored by midfielder type)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pass completion rate vs Pressures
    axes[0, 1].scatter(df['pass_completion_rate'], df['pressures_applied'],
                       alpha=0.6, s=50, c=df['midfielder_type'], cmap='viridis')
    axes[0, 1].set_xlabel('Pass Completion Rate')
    axes[0, 1].set_ylabel('Pressures Applied')
    axes[0, 1].set_title('Pass Completion vs Pressures')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Possessions vs Time
    axes[1, 0].scatter(df['possessions_involved'], df['possession_time_seconds'],
                      alpha=0.6, s=50, c=df['midfielder_type'], cmap='viridis')
    axes[1, 0].set_xlabel('Possessions Involved')
    axes[1, 0].set_ylabel('Possession Time (seconds)')
    axes[1, 0].set_title('Possessions vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Position scatter
    axes[1, 1].scatter(df['average_position_x'], df['average_position_y'],
                      alpha=0.6, s=50, c=df['midfielder_type'], cmap='viridis')
    axes[1, 1].set_xlabel('Average Position X')
    axes[1, 1].set_ylabel('Average Position Y')
    axes[1, 1].set_title('Average Position on Pitch (colored by type)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent.parent / "data" / "processed" / "eda_feature_relationships.png", dpi=150)
    print(f"\n✓ Feature relationships plot saved to: data/processed/eda_feature_relationships.png")
    plt.close()

# ============================================================================
# CELL 8: Player-Level Analysis
# ============================================================================

def analyze_players(df):
    """Analyze individual player performance."""
    print("\n" + "=" * 80)
    print("CELL 8: PLAYER-LEVEL ANALYSIS")
    print("=" * 80)
    
    # Aggregate by player
    player_stats = df.groupby('player_name').agg({
        'match_id': 'count',
        'passes_attempted': 'mean',
        'pass_completion_rate': 'mean',
        'carries_attempted': 'mean',
        'pressures_applied': 'mean',
        'ball_recoveries': 'mean',
        'possessions_involved': 'mean',
        'possession_time_seconds': 'mean',
        'progressive_passes': 'mean',
        'progressive_carries': 'mean'
    }).round(2)
    
    player_stats.columns = ['Matches', 'Avg_Passes', 'Avg_Pass_Comp_Rate', 'Avg_Carries',
                           'Avg_Pressures', 'Avg_Recoveries', 'Avg_Possessions',
                           'Avg_Possession_Time', 'Avg_Prog_Passes', 'Avg_Prog_Carries']
    player_stats = player_stats.sort_values('Matches', ascending=False)
    
    print("\nPlayer Performance Summary (sorted by number of matches):")
    print("-" * 80)
    print(player_stats.to_string())
    
    # Top performers
    print("\n\nTop 5 Players by Key Metrics:")
    print("-" * 80)
    
    metrics = {
        'Most Passes': 'Avg_Passes',
        'Best Pass Accuracy': 'Avg_Pass_Comp_Rate',
        'Most Pressures': 'Avg_Pressures',
        'Most Recoveries': 'Avg_Recoveries',
        'Most Progressive Passes': 'Avg_Prog_Passes'
    }
    
    for metric_name, metric_col in metrics.items():
        top_5 = player_stats.nlargest(5, metric_col)[[metric_col]]
        print(f"\n{metric_name}:")
        print(top_5.to_string())
    
    # Save player stats
    player_stats_path = Path(__file__).parent.parent.parent / "data" / "processed" / "eda_player_stats.csv"
    player_stats.to_csv(player_stats_path)
    print(f"\n✓ Player statistics saved to: {player_stats_path}")
    
    # Visualization
    if not HAS_VIZ:
        print("\n⚠ Visualization skipped (matplotlib not available)")
        return
    
    top_players = player_stats.head(10)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Passes and carries
    x = np.arange(len(top_players))
    width = 0.35
    axes[0, 0].bar(x - width/2, top_players['Avg_Passes'], width, label='Passes', alpha=0.8)
    axes[0, 0].bar(x + width/2, top_players['Avg_Carries'], width, label='Carries', alpha=0.8)
    axes[0, 0].set_xlabel('Player')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Top 10 Players: Passes vs Carries')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(top_players.index, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Pass completion rate
    axes[0, 1].barh(range(len(top_players)), top_players['Avg_Pass_Comp_Rate'], alpha=0.8)
    axes[0, 1].set_yticks(range(len(top_players)))
    axes[0, 1].set_yticklabels(top_players.index)
    axes[0, 1].set_xlabel('Pass Completion Rate')
    axes[0, 1].set_title('Top 10 Players: Pass Completion Rate')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Pressures
    axes[1, 0].barh(range(len(top_players)), top_players['Avg_Pressures'], alpha=0.8, color='orange')
    axes[1, 0].set_yticks(range(len(top_players)))
    axes[1, 0].set_yticklabels(top_players.index)
    axes[1, 0].set_xlabel('Pressures Applied')
    axes[1, 0].set_title('Top 10 Players: Pressures Applied')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Progressive actions
    axes[1, 1].bar(x - width/2, top_players['Avg_Prog_Passes'], width, label='Prog Passes', alpha=0.8)
    axes[1, 1].bar(x + width/2, top_players['Avg_Prog_Carries'], width, label='Prog Carries', alpha=0.8)
    axes[1, 1].set_xlabel('Player')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Top 10 Players: Progressive Actions')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(top_players.index, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent.parent / "data" / "processed" / "eda_player_analysis.png", dpi=150)
    print(f"\n✓ Player analysis plot saved to: data/processed/eda_player_analysis.png")
    plt.close()

# ============================================================================
# CELL 9: Match-Level Patterns
# ============================================================================

def analyze_match_patterns(df):
    """Analyze patterns across matches."""
    print("\n" + "=" * 80)
    print("CELL 9: MATCH-LEVEL PATTERNS")
    print("=" * 80)
    
    # Aggregate by match
    match_stats = df.groupby('match_id').agg({
        'player_id': 'count',
        'passes_attempted': 'sum',
        'carries_attempted': 'sum',
        'pressures_applied': 'sum',
        'ball_recoveries': 'sum',
        'possessions_involved': 'mean',
        'possession_time_seconds': 'mean'
    }).round(2)
    
    match_stats.columns = ['Midfielders', 'Total_Passes', 'Total_Carries', 'Total_Pressures',
                          'Total_Recoveries', 'Avg_Possessions', 'Avg_Possession_Time']
    
    print("\nMatch-Level Statistics:")
    print("-" * 80)
    print(match_stats.describe().round(2).to_string())
    
    # Match activity over time (if we had match dates, we'd use those)
    print(f"\nTotal Matches Analyzed: {len(match_stats)}")
    print(f"Average Midfielders per Match: {match_stats['Midfielders'].mean():.1f}")
    print(f"Total Passes across all matches: {match_stats['Total_Passes'].sum():.0f}")
    print(f"Total Carries across all matches: {match_stats['Total_Carries'].sum():.0f}")
    
    # Save match stats
    match_stats_path = Path(__file__).parent.parent.parent / "data" / "processed" / "eda_match_stats.csv"
    match_stats.to_csv(match_stats_path)
    print(f"\n✓ Match statistics saved to: {match_stats_path}")

# ============================================================================
# CELL 10: Summary and Insights
# ============================================================================

def generate_summary(df, metadata_cols, feature_cols):
    """Generate summary insights."""
    print("\n" + "=" * 80)
    print("CELL 10: SUMMARY AND INSIGHTS")
    print("=" * 80)
    
    print("\n📊 Dataset Overview:")
    print("-" * 80)
    print(f"  • Total observations: {len(df):,}")
    print(f"  • Unique players: {df['player_name'].nunique()}")
    print(f"  • Unique matches: {df['match_id'].nunique()}")
    print(f"  • Midfielder types: {df['midfielder_type'].nunique()}")
    print(f"  • Total features: {len(feature_cols)}")
    
    print("\n🎯 Key Insights:")
    print("-" * 80)
    
    # Top insights
    numeric_features = df[feature_cols].select_dtypes(include=[np.number])
    
    # Most variable features
    most_variable = numeric_features.std().nlargest(5)
    print("\n  Most Variable Features (highest std dev):")
    for feat, std_val in most_variable.items():
        print(f"    • {feat}: {std_val:.2f}")
    
    # Features with most zeros
    zero_counts = (numeric_features == 0).sum()
    most_zeros = zero_counts.nlargest(5)
    print("\n  Features with Most Zero Values:")
    for feat, count in most_zeros.items():
        pct = (count / len(df)) * 100
        print(f"    • {feat}: {count} zeros ({pct:.1f}%)")
    
    # Average performance
    print("\n  Average Performance Metrics:")
    print(f"    • Average passes per match: {df['passes_attempted'].mean():.1f}")
    print(f"    • Average pass completion rate: {df['pass_completion_rate'].mean():.2%}")
    print(f"    • Average carries per match: {df['carries_attempted'].mean():.1f}")
    print(f"    • Average pressures per match: {df['pressures_applied'].mean():.1f}")
    
    print("\nEDA Complete!")
    print("=" * 80)
    print("\nAll plots and statistics have been saved to:")
    print("  • data/processed/eda_*.png (visualizations)")
    print("  • data/processed/eda_*.csv (statistics)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function - runs all EDA cells."""
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS - BARCELONA 2014/2015 MIDFIELDER FEATURES")
    print("=" * 80)
    
    # Cell 1: Load data
    df = load_data()
    
    # Cell 2: Missing values
    metadata_cols, feature_cols = analyze_missing_values(df)
    
    # Cell 3: Descriptive statistics
    numeric_features = descriptive_statistics(df, feature_cols)
    
    # Cell 4: Midfielder types
    type_names = analyze_midfielder_types(df)
    
    # Cell 5: Feature distributions
    analyze_feature_distributions(df, feature_cols)
    
    # Cell 6: Correlations
    analyze_correlations(df, feature_cols)
    
    # Cell 7: Feature relationships
    analyze_feature_relationships(df)
    
    # Cell 8: Player analysis
    analyze_players(df)
    
    # Cell 9: Match patterns
    analyze_match_patterns(df)
    
    # Cell 10: Summary
    generate_summary(df, metadata_cols, feature_cols)
    
    print("\n" + "=" * 80)
    print("EDA PROCESS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

