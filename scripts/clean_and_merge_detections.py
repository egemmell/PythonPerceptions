#!/usr/bin/env python3
"""
Clean and merge YOLO detection data with perception data
Equivalent to the R code provided by the user
"""

import pandas as pd
from pathlib import Path

def clean_and_merge_data(
    detection_file='results/yolo_coco/detections_summary.csv',
    games_file='results/gamedata.csv',
    perception_file='results/clean_perception_data11.24.2023.csv',
    output_dir='results'
):
    """
    Clean YOLO detection data and merge with perception data
    
    Args:
        detection_file: Path to YOLO detection summary CSV
        games_file: Path to gamedata CSV
        perception_file: Path to perception data CSV
        output_dir: Directory to save merged data
    
    Returns:
        percep_det: Merged dataframe with both perception and detection data
        games: Games dataframe
    """
    
    print("Loading data...")
    
    # Load detection data
    det = pd.read_csv(detection_file)
    print(f"  Detections: {len(det)} images")
    
    # Load games data
    games = pd.read_csv(games_file)
    print(f"  Games: {len(games)} game pairs")
    
    # Load perception data
    perceptions = pd.read_csv(perception_file)
    print(f"  Perceptions (raw): {len(perceptions)} rows")
    
    # ========== CLEAN DETECTION DATA ==========
    print("\nCleaning detection data...")
    
    # Remove ".jpg" from image_path to create img_id
    det['img_id'] = det['image_path'].str.replace('.jpg', '', regex=False)
    det['img_id'] = det['img_id'].str.replace('.jpeg', '', regex=False)
    
    # Drop the image_path column (no longer needed)
    det = det.drop('image_path', axis=1)
    
    print(f"  Created img_id column")
    print(f"  Detection data shape: {det.shape}")
    
    # ========== CLEAN PERCEPTION DATA ==========
    print("\nCleaning perception data...")
    
    # Select specific columns
    perceptions = perceptions[['idx', 'img_id', 'month', 'year', 'wins', 'games', 'win_probability']]
    print(f"  Selected columns: {list(perceptions.columns)}")
    
    # Remove underscores from img_id
    perceptions['img_id'] = perceptions['img_id'].str.replace('_', '', regex=False)
    
    print(f"  Removed underscores from img_id")
    
    # Filter to only include games > 0
    initial_count = len(perceptions)
    perceptions = perceptions[perceptions['games'] > 0]
    filtered_count = len(perceptions)
    print(f"  Filtered games > 0: {initial_count} -> {filtered_count} rows")
    
    # ========== MERGE DATA ==========
    print("\nMerging perception and detection data...")
    
    # Merge on img_id
    percep_det = pd.merge(perceptions, det, on='img_id', how='left')
    
    print(f"  Merged data shape: {percep_det.shape}")
    print(f"  Columns: {list(percep_det.columns)}")
    
    # Check for mismatches
    unmatched = percep_det[percep_det.isnull().any(axis=1)]
    if len(unmatched) > 0:
        print(f"\n  ⚠️  WARNING: {len(unmatched)} rows with missing detection data")
        print(f"     These images may not have YOLO detections")
    else:
        print(f"  ✓ All perception records matched with detection data")
    
    # ========== SAVE OUTPUT ==========
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'percep_det_merged.csv'
    percep_det.to_csv(output_file, index=False)
    print(f"\nSaved merged data to: {output_file}")
    
    # Also save games for reference
    games_output = output_dir / 'games_data.csv'
    games.to_csv(games_output, index=False)
    print(f"Saved games data to: {games_output}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total records in percep_det: {len(percep_det)}")
    print(f"Unique images: {percep_det['img_id'].nunique()}")
    print(f"Total games: {len(games)}")
    print(f"Date range: {percep_det['year'].min()}-{percep_det['year'].max()}")
    print(f"Win probability range: {percep_det['win_probability'].min():.2f} - {percep_det['win_probability'].max():.2f}")
    
    # Check image participation
    winner_indices = set(games['winner'].unique())
    loser_indices = set(games['loser'].unique())
    all_game_indices = winner_indices | loser_indices
    
    print(f"\nImage participation in games:")
    print(f"  Images that appear as winners: {len(winner_indices)}")
    print(f"  Images that appear as losers: {len(loser_indices)}")
    print(f"  Images in both roles: {len(winner_indices & loser_indices)}")
    print(f"  Total unique images in games: {len(all_game_indices)}")
    
    # Check for images appearing multiple times
    winner_counts = games['winner'].value_counts()
    loser_counts = games['loser'].value_counts()
    
    print(f"\nImage frequency in games:")
    print(f"  Max times as winner: {winner_counts.max()}")
    print(f"  Max times as loser: {loser_counts.max()}")
    print(f"  Images appearing 5+ times: {(winner_counts >= 5).sum() + (loser_counts >= 5).sum()}")
    
    # Detection feature columns
    feature_cols = [col for col in percep_det.columns 
                   if col not in ['img_id', 'idx', 'month', 'year', 'wins', 'games', 'win_probability']]
    print(f"\nDetected feature columns ({len(feature_cols)}):")
    for col in sorted(feature_cols):
        print(f"  - {col}")
    
    print(f"\nFirst few rows:")
    print(percep_det.head())
    print("="*80 + "\n")
    
    return percep_det, games


def validate_data(percep_det, games):
    """
    Validate the merged data
    """
    print("Validating data...")
    
    # Check for duplicate img_ids
    duplicates = percep_det[percep_det.duplicated(subset=['img_id'], keep=False)]
    if len(duplicates) > 0:
        print(f"  ⚠️  {len(duplicates)} duplicate img_ids found")
    else:
        print(f"  ✓ No duplicate img_ids")
    
    # Check if all game indices exist in percep_det
    all_indices = set(percep_det['idx'].unique())
    game_indices = set(games['winner'].unique()) | set(games['loser'].unique())
    missing = game_indices - all_indices
    
    if len(missing) > 0:
        print(f"  ⚠️  {len(missing)} game indices not found in perception data")
        print(f"     This may cause issues with pairwise analysis")
    else:
        print(f"  ✓ All game indices found in perception data")
    
    # Check for NaN values
    nan_counts = percep_det.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"\n  NaN values found:")
        print(nan_counts[nan_counts > 0])
    else:
        print(f"  ✓ No NaN values in data")
    
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean and merge YOLO detection with perception data')
    parser.add_argument('--detections', type=str, default='results/yolo_coco/detections_summary.csv',
                       help='Path to YOLO detection summary CSV')
    parser.add_argument('--games', type=str, default='results/gamedata.csv',
                       help='Path to games data CSV')
    parser.add_argument('--perceptions', type=str, default='results/clean_perception_data11.24.2023.csv',
                       help='Path to perception data CSV')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run cleaning and merging
    percep_det, games = clean_and_merge_data(
        detection_file=args.detections,
        games_file=args.games,
        perception_file=args.perceptions,
        output_dir=args.output
    )
    
    # Validate
    validate_data(percep_det, games)
    
    return percep_det, games


if __name__ == '__main__':
    percep_det, games = main()
