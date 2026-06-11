#!/usr/bin/env python3
"""
Analyze differences between winning and losing images using high-confidence detections only
Filters detections_detailed.csv by confidence threshold, then aggregates and compares
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def load_detailed_detections(detailed_file):
    """Load detailed detections with confidence scores"""
    df = pd.read_csv(detailed_file)
    print(f"Loaded detailed detections: {len(df)} detections")
    print(f"  Confidence range: {df['confidence'].min():.3f} - {df['confidence'].max():.3f}")
    print(f"  Mean confidence: {df['confidence'].mean():.3f}")
    return df

def load_perception_data(perception_file):
    """Load perception data with img_id and idx (metadata only)"""
    df = pd.read_csv(perception_file)
    print(f"Loaded perception data: {len(df)} images")
    
    # Keep only metadata columns (not detection columns)
    metadata_cols = ['img_id', 'idx', 'month', 'year', 'wins', 'games', 'win_probability']
    df = df[[col for col in metadata_cols if col in df.columns]]
    
    # Create mapping from img_id to idx
    img_to_idx = dict(zip(df['img_id'], df['idx']))
    return df, img_to_idx

def load_games(games_file):
    """Load games data"""
    df = pd.read_csv(games_file)
    print(f"Loaded games: {len(df)} games")
    return df

def aggregate_by_confidence(df_detailed, img_to_idx, conf_threshold):
    """
    Aggregate detailed detections by image, filtering by confidence threshold
    
    Args:
        df_detailed: DataFrame with one row per detection (has image_path, class_name, confidence)
        img_to_idx: Dictionary mapping img_id to idx
        conf_threshold: Minimum confidence to include
    
    Returns:
        DataFrame with counts per image (only high-confidence detections)
    """
    
    print(f"\nFiltering detections by confidence >= {conf_threshold}...")
    
    # Filter to high-confidence only
    df_high_conf = df_detailed[df_detailed['confidence'] >= conf_threshold].copy()
    
    print(f"  Detections above threshold: {len(df_high_conf)}/{len(df_detailed)} ({len(df_high_conf)/len(df_detailed)*100:.1f}%)")
    
    # Remove .jpg/.jpeg from image_path to get img_id
    df_high_conf['img_id'] = df_high_conf['image_path'].str.replace('.jpg', '', regex=False)
    df_high_conf['img_id'] = df_high_conf['img_id'].str.replace('.jpeg', '', regex=False)
    
    # Add idx from mapping
    df_high_conf['idx'] = df_high_conf['img_id'].map(img_to_idx)
    
    # Group by image and class to count detections
    grouped = df_high_conf.groupby(['idx', 'class_name']).size().reset_index(name='count')
    
    # Pivot to get one column per class
    df_agg = grouped.pivot(index='idx', columns='class_name', values='count').fillna(0).astype(int)
    
    print(f"  Aggregated to {len(df_agg)} images with high-confidence detections")
    print(f"  Classes: {list(df_agg.columns)}")
    
    return df_agg

def analyze_with_confidence(detailed_file, perception_file, games_file, conf_threshold=0.5):
    """
    Complete analysis pipeline with confidence filtering
    """
    
    print("="*80)
    print(f"ANALYSIS: High-Confidence Detections (confidence >= {conf_threshold})")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df_detailed = load_detailed_detections(detailed_file)
    df_perception, img_to_idx = load_perception_data(perception_file)
    games = load_games(games_file)
    
    # Aggregate by confidence
    df_high_conf = aggregate_by_confidence(df_detailed, img_to_idx, conf_threshold)
    
    # Get all feature columns
    feature_cols = list(df_high_conf.columns)
    
    # Match with perception data
    print(f"\nMatching with perception data...")
    df_merged = df_perception.set_index('idx').join(df_high_conf, how='inner')
    df_merged = df_merged.reset_index()
    
    print(f"  Merged: {len(df_merged)} images with both perception and high-conf detections")
    
    # Analyze pairwise differences
    print(f"\nAnalyzing {len(games)} games...")
    differences = []
    matched = 0
    
    df_by_idx = df_merged.set_index('idx')
    
    for game_id, row in games.iterrows():
        winner_idx = row['winner']
        loser_idx = row['loser']
        
        if winner_idx not in df_by_idx.index or loser_idx not in df_by_idx.index:
            continue
        
        matched += 1
        winner_data = df_by_idx.loc[winner_idx]
        loser_data = df_by_idx.loc[loser_idx]
        
        diff_row = {
            'game_id': game_id,
            'winner_idx': winner_idx,
            'loser_idx': loser_idx
        }
        
        for feature in feature_cols:
            winner_val = winner_data[feature] if feature in winner_data.index else 0
            loser_val = loser_data[feature] if feature in loser_data.index else 0
            diff_row[f'{feature}_diff'] = winner_val - loser_val
            diff_row[f'{feature}_winner'] = winner_val
            diff_row[f'{feature}_loser'] = loser_val
        
        differences.append(diff_row)
    
    df_diff = pd.DataFrame(differences)
    print(f"  Matched games: {matched}/{len(games)}")
    
    # Run statistics
    print("\n" + "="*80)
    print(f"RESULTS: Confidence >= {conf_threshold}")
    print("="*80)
    
    results = []
    
    for feature in feature_cols:
        diff_col = f'{feature}_diff'
        diffs = df_diff[diff_col]
        
        mean_diff = diffs.mean()
        std_diff = diffs.std()
        median_diff = diffs.median()
        
        winner_more = (diffs > 0).sum()
        loser_more = (diffs < 0).sum()
        equal = (diffs == 0).sum()
        
        t_stat, p_value = stats.ttest_1samp(diffs, 0)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
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
        
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"\n{feature}")
        print(f"  Mean: {mean_diff:+.3f} ± {std_diff:.3f} | Median: {median_diff:+.1f}")
        print(f"  Winners more: {winner_more:5d} ({winner_more/len(df_diff)*100:5.1f}%) | "
              f"Losers more: {loser_more:5d} ({loser_more/len(df_diff)*100:5.1f}%) | "
              f"Equal: {equal:5d} ({equal/len(df_diff)*100:5.1f}%)")
        print(f"  p={p_value:.4f} {sig} | d={cohens_d:.3f}")
    
    return pd.DataFrame(results), df_diff

def compare_thresholds(detailed_file, perception_file, games_file):
    """
    Compare results across multiple confidence thresholds
    """
    
    print("\n" + "="*80)
    print("COMPARING MULTIPLE CONFIDENCE THRESHOLDS")
    print("="*80)
    
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    all_results = {}
    
    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"Threshold: {threshold}")
        print(f"{'='*80}")
        
        try:
            results_df, df_diff = analyze_with_confidence(
                detailed_file, perception_file, games_file, threshold
            )
            all_results[threshold] = results_df
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Create summary comparison
    print("\n" + "="*80)
    print("SUMMARY: Effect Sizes (Cohen's d) by Threshold")
    print("="*80)
    
    if len(all_results) == 0:
        print("No results to compare - check for errors above")
        return
    
    for feature in all_results[list(all_results.keys())[0]]['Feature'].values:
        print(f"\n{feature}")
        for threshold in thresholds:
            if threshold in all_results:
                row = all_results[threshold][all_results[threshold]['Feature'] == feature]
                if len(row) > 0:
                    d = row['Cohens_d'].values[0]
                    p = row['P_Value'].values[0]
                    sig = '*' if p < 0.05 else ''
                    print(f"  {threshold}: d={d:+.3f} (p={p:.4f}){sig}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze with high-confidence detections')
    parser.add_argument('--detailed', type=str, default='results/yolo_coco/detections_detailed.csv',
                       help='Path to detailed detections CSV')
    parser.add_argument('--perception', type=str, default='results/percep_det_merged.csv',
                       help='Path to perception data CSV')
    parser.add_argument('--games', type=str, default='results/games_data.csv',
                       help='Path to games CSV')
    parser.add_argument('--threshold', type=float, default=0.50,
                       help='Confidence threshold (default 0.50)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple thresholds')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_thresholds(args.detailed, args.perception, args.games)
    else:
        results_df, df_diff = analyze_with_confidence(
            args.detailed, args.perception, args.games, args.threshold
        )
        
        # Save results
        output_dir = Path('results') / 'analysis' / f'confidence_{args.threshold}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_dir / 'results_summary.csv', index=False)
        df_diff.to_csv(output_dir / 'all_game_differences.csv', index=False)
        
        print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main()
