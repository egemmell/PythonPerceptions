#!/usr/bin/env python3
"""
Analyze differences between winning and losing images
Handles images appearing in multiple games (as both winners and losers)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def load_data(merged_file, games_file=None):
    """
    Load merged perception+detection data and games data
    
    Args:
        merged_file: Path to percep_det_merged.csv (from clean_and_merge_detections.py)
        games_file: Path to games CSV (if different from merged file)
    
    Returns:
        percep_det: DataFrame with perception + detection data
        games: DataFrame with winner/loser pairs (by idx)
    """
    
    # Load merged perception + detection data
    percep_det = pd.read_csv(merged_file)
    
    print(f"Loaded perception+detection data: {percep_det.shape}")
    print(f"  Unique images: {percep_det['img_id'].nunique()}")
    print(f"  Index range: {percep_det['idx'].min()} to {percep_det['idx'].max()}")
    
    # If games file provided separately
    if games_file:
        games = pd.read_csv(games_file)
    else:
        # Try to infer games file location
        merged_path = Path(merged_file)
        games_path = merged_path.parent / 'games_data.csv'
        if games_path.exists():
            games = pd.read_csv(games_path)
        else:
            # Try to find gamedata.csv
            games_path = merged_path.parent / 'gamedata.csv'
            if games_path.exists():
                games = pd.read_csv(games_path)
            else:
                raise FileNotFoundError("Could not find games data. Please provide games_file argument.")
    
    print(f"Loaded games data: {len(games)} games")
    
    return percep_det, games


def analyze_pairwise_differences(percep_det, games):
    """
    Analyze differences between winner and loser in each game
    
    Each image may appear in multiple games (as winner or loser)
    Each row in output represents one game
    
    Args:
        percep_det: DataFrame with idx, img_id, and feature columns
        games: DataFrame with winner (idx) and loser (idx) columns
    
    Returns:
        df_diff: DataFrame with differences for each game
        feature_cols: List of feature column names
    """
    
    # Identify feature columns
    metadata_cols = ['img_id', 'idx', 'month', 'year', 'wins', 'games', 'win_probability']
    feature_cols = [col for col in percep_det.columns if col not in metadata_cols]
    
    print(f"\nAnalyzing {len(games)} games")
    print(f"Features ({len(feature_cols)}): {feature_cols}\n")
    
    # Set idx as index for faster lookup
    percep_by_idx = percep_det.set_index('idx')
    
    differences = []
    matched_games = 0
    missing_indices = []
    
    # For each game, calculate difference between winner and loser
    for game_id, row in games.iterrows():
        winner_idx = row['winner']
        loser_idx = row['loser']
        
        # Try to find the images
        try:
            winner_data = percep_by_idx.loc[winner_idx]
            loser_data = percep_by_idx.loc[loser_idx]
            matched_games += 1
            
            # Calculate differences (winner - loser)
            diff_row = {
                'game_id': game_id,
                'winner_idx': winner_idx,
                'loser_idx': loser_idx,
                'winner_img_id': winner_data['img_id'],
                'loser_img_id': loser_data['img_id']
            }
            
            for feature in feature_cols:
                winner_val = winner_data[feature]
                loser_val = loser_data[feature]
                diff_row[f'{feature}_diff'] = winner_val - loser_val
                diff_row[f'{feature}_winner'] = winner_val
                diff_row[f'{feature}_loser'] = loser_val
            
            differences.append(diff_row)
            
        except KeyError as e:
            missing_idx = winner_idx if winner_idx not in percep_by_idx.index else loser_idx
            missing_indices.append(missing_idx)
    
    df_diff = pd.DataFrame(differences)
    
    print(f"Matched games: {matched_games}/{len(games)}")
    if missing_indices:
        print(f"Missing indices: {len(set(missing_indices))} unique indices")
    
    # Calculate image statistics
    all_indices_in_games = set(games['winner'].unique()) | set(games['loser'].unique())
    print(f"\nImage participation in games:")
    print(f"  Total unique images in games: {len(all_indices_in_games)}")
    print(f"  Images in percep_det: {len(percep_det)}")
    
    winner_counts = games['winner'].value_counts()
    loser_counts = games['loser'].value_counts()
    print(f"  Images appearing as winners: {len(winner_counts)}")
    print(f"  Images appearing as losers: {len(loser_counts)}")
    print(f"  Images appearing in both roles: {len(set(winner_counts.index) & set(loser_counts.index))}")
    print(f"  Max games for single image: {max(winner_counts.max(), loser_counts.max())}")
    
    return df_diff, feature_cols


def summarize_differences(df_diff, feature_cols):
    """
    Summarize feature differences between winners and losers
    
    Returns statistics and p-values for each feature
    """
    
    print("\n" + "="*80)
    print("FEATURE DIFFERENCES: Winner vs Loser (Across All Games)")
    print("="*80)
    
    results = []
    
    for feature in feature_cols:
        diff_col = f'{feature}_diff'
        
        # Get differences
        differences = df_diff[diff_col]
        
        # Calculate statistics
        mean_diff = differences.mean()
        std_diff = differences.std()
        median_diff = differences.median()
        
        # Count directions
        winner_more = (differences > 0).sum()
        loser_more = (differences < 0).sum()
        equal = (differences == 0).sum()
        
        # Paired t-test (one-sample t-test on differences)
        t_stat, p_value = stats.ttest_1samp(differences, 0)
        
        # Effect size (Cohen's d)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Store results
        results.append({
            'Feature': feature,
            'Mean_Difference': mean_diff,
            'Std_Difference': std_diff,
            'Median_Difference': median_diff,
            'Winner_More': winner_more,
            'Loser_More': loser_more,
            'Equal': equal,
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Cohens_d': cohens_d
        })
        
        # Print detailed results
        print(f"\n{feature.upper()}")
        print(f"  Mean difference: {mean_diff:+.3f} ± {std_diff:.3f}")
        print(f"  Median difference: {median_diff:+.1f}")
        print(f"  Winners have more: {winner_more} ({winner_more/len(df_diff)*100:.1f}%)")
        print(f"  Losers have more: {loser_more} ({loser_more/len(df_diff)*100:.1f}%)")
        print(f"  Equal: {equal} ({equal/len(df_diff)*100:.1f}%)")
        print(f"  t-test p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        print(f"  Cohen's d: {cohens_d:.3f}")
    
    return pd.DataFrame(results)


def create_visualizations(df_diff, feature_cols, output_dir='results'):
    """
    Create visualization comparing features
    """
    
    output_dir = Path(output_dir) / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating visualizations in {output_dir}...")
    
    # 1. Distribution of differences for each feature
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle('Distribution of Feature Differences (Winner - Loser)', fontsize=14, fontweight='bold')
    
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        diff_col = f'{feature}_diff'
        
        data = df_diff[diff_col]
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        ax.set_xlabel('Difference')
        ax.set_ylabel('Count')
        ax.set_title(feature)
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(feature_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_differences_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: feature_differences_distribution.png")
    plt.close()
    
    # 2. Box plot of differences
    if len(feature_cols) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data_for_box = [df_diff[f'{feature}_diff'].values for feature in feature_cols]
        
        bp = ax.boxplot(data_for_box, labels=feature_cols, patch_artist=True)
        
        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_ylabel('Difference (Winner - Loser)')
        ax.set_title('Feature Count Differences: Winner vs Loser')
        ax.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_differences_boxplot.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: feature_differences_boxplot.png")
        plt.close()
    
    # 3. Scatter plot: Winner vs Loser counts
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle('Feature Counts: Winner vs Loser', fontsize=14, fontweight='bold')
    
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        
        winner_col = f'{feature}_winner'
        loser_col = f'{feature}_loser'
        
        ax.scatter(df_diff[loser_col], df_diff[winner_col], alpha=0.5, s=20)
        
        # Add diagonal line (equal counts)
        min_val = min(df_diff[loser_col].min(), df_diff[winner_col].min())
        max_val = max(df_diff[loser_col].max(), df_diff[winner_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal')
        
        ax.set_xlabel(f'Loser {feature}')
        ax.set_ylabel(f'Winner {feature}')
        ax.set_title(feature)
        ax.grid(alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(len(feature_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_scatter_winner_vs_loser.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: feature_scatter_winner_vs_loser.png")
    plt.close()


def save_detailed_results(df_diff, results_df, feature_cols, output_dir='results'):
    """Save detailed results to CSV"""
    
    output_dir = Path(output_dir) / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary statistics
    results_df.to_csv(output_dir / 'feature_comparison_summary.csv', index=False)
    print(f"Saved: feature_comparison_summary.csv")
    
    # Save all pairwise differences
    df_diff.to_csv(output_dir / 'all_game_differences.csv', index=False)
    print(f"Saved: all_game_differences.csv ({len(df_diff)} games)")
    
    # Sort by most significant features
    results_sorted = results_df.sort_values('P_Value')
    results_sorted.to_csv(output_dir / 'features_by_significance.csv', index=False)
    print(f"Saved: features_by_significance.csv")


def print_key_findings(results_df):
    """Print key findings"""
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Features where winners have significantly more
    significant = results_df[results_df['P_Value'] < 0.05].sort_values('Mean_Difference', ascending=False)
    
    if len(significant) > 0:
        print("\nFeatures where WINNERS have significantly MORE (p < 0.05):")
        for _, row in significant.iterrows():
            if row['Mean_Difference'] > 0:
                print(f"  {row['Feature']:20s}: +{row['Mean_Difference']:.3f} (p={row['P_Value']:.4f}, d={row['Cohens_d']:.3f})")
    
    # Features where losers have significantly more
    losers_sig = results_df[results_df['P_Value'] < 0.05].sort_values('Mean_Difference')
    
    if len(losers_sig) > 0:
        print("\nFeatures where LOSERS have significantly MORE (p < 0.05):")
        for _, row in losers_sig.iterrows():
            if row['Mean_Difference'] < 0:
                print(f"  {row['Feature']:20s}: {row['Mean_Difference']:.3f} (p={row['P_Value']:.4f}, d={row['Cohens_d']:.3f})")
    
    # Effect sizes
    print("\nLargest effect sizes (Cohen's d):")
    largest_effects = results_df.reindex(results_df['Cohens_d'].abs().argsort(ascending=False))
    for _, row in largest_effects.head(5).iterrows():
        direction = "Winners more" if row['Cohens_d'] > 0 else "Losers more"
        print(f"  {row['Feature']:20s}: d={row['Cohels_d']:+.3f} ({direction})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze differences between winning and losing images')
    parser.add_argument('--merged', type=str, default='results/percep_det_merged.csv',
                       help='Path to merged perception+detection data')
    parser.add_argument('--games', type=str, default=None,
                       help='Path to games CSV (optional, tries to find automatically)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    percep_det, games = load_data(args.merged, args.games)
    
    # Analyze differences
    print("\nAnalyzing pairwise differences...")
    df_diff, feature_cols = analyze_pairwise_differences(percep_det, games)
    
    # Summarize
    print("\nSummarizing results...")
    results_df = summarize_differences(df_diff, feature_cols)
    
    # Visualize
    print("\nGenerating visualizations...")
    create_visualizations(df_diff, feature_cols, args.output)
    
    # Save
    print("\nSaving results...")
    save_detailed_results(df_diff, results_df, feature_cols, args.output)
    
    # Key findings
    print_key_findings(results_df)
    
    print("\n" + "="*80)
    print(f"Analysis complete! Check {args.output}/analysis/ for results")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
