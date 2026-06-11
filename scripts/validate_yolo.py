#!/usr/bin/env python3
"""
Validate YOLO predictions against ground truth data
Compares YOLO detections to manually annotated ground truth
Calculates precision, recall, F1 score, and other metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml

def load_ground_truth(gt_file):
    """Load ground truth annotations"""
    df_gt = pd.read_csv(gt_file)
    print(f"Loaded ground truth: {len(df_gt)} images")
    print(f"Columns: {list(df_gt.columns)}")
    return df_gt

def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_yolo_predictions(image_paths, model, conf_threshold=0.25, device='cpu'):
    """
    Run YOLO on images and return counts per image
    Normalizes class names to match ground truth format
    
    Returns DataFrame with same structure as ground truth
    """
    
    predictions = []
    
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            device=device,
            verbose=False
        )[0]
        
        # Count by class
        class_counts = {}
        if results.boxes is not None:
            for box in results.boxes:
                class_name = model.names[int(box.cls[0])]
                # Normalize class name
                class_name = normalize_class_name(class_name)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Get image filename
        img_name = Path(img_path).name
        
        # Create row
        row = {'image_path': img_name}
        row.update(class_counts)
        predictions.append(row)
    
    # Convert to DataFrame and fill missing classes with 0
    df_pred = pd.DataFrame(predictions)
    
    # Fill NaN with 0
    df_pred = df_pred.fillna(0).astype({col: int for col in df_pred.columns if col != 'image_path'})
    
    print(f"Generated predictions for {len(df_pred)} images")
    
    return df_pred

def normalize_class_name(name):
    """Normalize class names: lowercase and strip whitespace"""
    return name.lower().strip()

def align_dataframes(df_gt, df_pred):
    """
    Align ground truth and predictions to have same columns
    Keep only images that appear in both datasets
    Normalize class names and filter to only ground truth classes
    """
    
    print("\nAligning ground truth and predictions...")
    
    # Remove .jpg from image_path in predictions if needed
    df_pred['image_path'] = df_pred['image_path'].str.replace('.jpg', '', regex=False)
    df_pred['image_path'] = df_pred['image_path'].str.replace('.jpeg', '', regex=False)
    
    df_gt_clean = df_gt.copy()
    df_gt_clean['image_path'] = df_gt_clean['image_path'].str.replace('.jpg', '', regex=False)
    df_gt_clean['image_path'] = df_gt_clean['image_path'].str.replace('.jpeg', '', regex=False)
    
    # Find common images
    gt_images = set(df_gt_clean['image_path'].unique())
    pred_images = set(df_pred['image_path'].unique())
    common_images = gt_images & pred_images
    
    print(f"  Ground truth images: {len(gt_images)}")
    print(f"  Prediction images: {len(pred_images)}")
    print(f"  Common images: {len(common_images)}")
    
    # Filter to common images
    df_gt_aligned = df_gt_clean[df_gt_clean['image_path'].isin(common_images)].reset_index(drop=True)
    df_pred_aligned = df_pred[df_pred['image_path'].isin(common_images)].reset_index(drop=True)
    
    # Get ground truth classes and normalize them
    gt_classes = set(df_gt_aligned.columns) - {'image_path'}
    gt_classes_normalized = {normalize_class_name(col): col for col in gt_classes}
    
    print(f"\n  Ground truth classes (original): {sorted(gt_classes)}")
    print(f"  Ground truth classes (normalized): {sorted(gt_classes_normalized.keys())}")
    
    # Normalize prediction column names
    pred_classes = set(df_pred_aligned.columns) - {'image_path'}
    pred_classes_normalized = {normalize_class_name(col): col for col in pred_classes}
    
    print(f"  Prediction classes (original): {sorted(pred_classes)}")
    print(f"  Prediction classes (normalized): {sorted(pred_classes_normalized.keys())}")
    
    # Find intersection - only keep classes in ground truth
    matching_classes = sorted(set(gt_classes_normalized.keys()) & set(pred_classes_normalized.keys()))
    
    print(f"\n  Matching classes (in both GT and predictions): {matching_classes}")
    
    # Rename columns to normalized names in both dataframes
    gt_rename = {col: normalize_class_name(col) for col in df_gt_aligned.columns if col != 'image_path'}
    pred_rename = {col: normalize_class_name(col) for col in df_pred_aligned.columns if col != 'image_path'}
    
    df_gt_aligned = df_gt_aligned.rename(columns=gt_rename)
    df_pred_aligned = df_pred_aligned.rename(columns=pred_rename)
    
    # Keep only matching classes
    cols_to_keep = ['image_path'] + matching_classes
    
    # Add missing columns with zeros
    for col in matching_classes:
        if col not in df_gt_aligned.columns:
            df_gt_aligned[col] = 0
        if col not in df_pred_aligned.columns:
            df_pred_aligned[col] = 0
    
    # Select and sort columns
    df_gt_aligned = df_gt_aligned[cols_to_keep]
    df_pred_aligned = df_pred_aligned[cols_to_keep]
    
    # Sort by image_path for alignment
    df_gt_aligned = df_gt_aligned.sort_values('image_path').reset_index(drop=True)
    df_pred_aligned = df_pred_aligned.sort_values('image_path').reset_index(drop=True)
    
    print(f"\n  Final classes for validation: {matching_classes}")
    print(f"  Images in aligned validation set: {len(df_gt_aligned)}")
    
    return df_gt_aligned, df_pred_aligned, matching_classes

def calculate_metrics(df_gt, df_pred, class_names):
    """
    Calculate metrics for count-based validation:
    - Accuracy_±1: % of images where prediction is within ±1 of ground truth
    - Recall: Detection rate (total predicted / total ground truth)
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Square Error
    """
    
    results = []
    
    for class_name in class_names:
        gt_col = df_gt[class_name].values
        pred_col = df_pred[class_name].values
        
        # Accuracy: % of images where count is within ±1
        correct = np.abs(gt_col - pred_col) <= 1
        accuracy = correct.sum() / len(correct) * 100
        
        # MAE and RMSE
        mae = np.mean(np.abs(gt_col - pred_col))
        rmse = np.sqrt(np.mean((gt_col - pred_col) ** 2))
        
        # Recall: Detection rate (total predicted / total ground truth)
        gt_total = gt_col.sum()
        pred_total = pred_col.sum()
        if gt_total > 0:
            recall = pred_total / gt_total * 100
        else:
            recall = 100.0 if pred_total == 0 else 0.0
        
        results.append({
            'Class': class_name,
            'GT_Count': int(gt_total),
            'Pred_Count': int(pred_total),
            'Accuracy_±1': f"{accuracy:.1f}%",
            'Recall': f"{recall:.1f}%",
            'MAE': f"{mae:.2f}",
            'RMSE': f"{rmse:.2f}"
        })
    
    return pd.DataFrame(results)

def calculate_image_accuracy(df_gt, df_pred, class_names):
    """
    Calculate per-image accuracy
    """
    
    # For each image, calculate how many classes match
    matches = 0
    total = 0
    
    per_image_accuracy = []
    
    for idx in range(len(df_gt)):
        gt_row = df_gt.iloc[idx][class_names].values
        pred_row = df_pred.iloc[idx][class_names].values
        
        # Count exact matches
        exact_matches = (gt_row == pred_row).sum()
        accuracy = exact_matches / len(class_names) * 100
        
        # Also count lenient matches (±1)
        lenient_matches = (np.abs(gt_row - pred_row) <= 1).sum()
        lenient_accuracy = lenient_matches / len(class_names) * 100
        
        per_image_accuracy.append({
            'image_path': df_gt.iloc[idx]['image_path'],
            'exact_match_accuracy': accuracy,
            'lenient_match_accuracy': lenient_accuracy
        })
    
    df_img_acc = pd.DataFrame(per_image_accuracy)
    
    return df_img_acc

def print_summary(metrics_df, img_acc_df):
    """Print validation summary with clean metrics"""
    
    print("\n" + "="*100)
    print("VALIDATION RESULTS BY CLASS")
    print("="*100)
    
    # Display with better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(metrics_df.to_string(index=False))
    
    # Explain metrics
    print("\n" + "="*100)
    print("METRICS EXPLAINED")
    print("="*100)
    print("""
Accuracy_±1 = % of images where predicted count is within ±1 of ground truth
Recall      = Detection rate (total objects predicted / total objects in ground truth)
MAE         = Mean Absolute Error (average per-image error in object counts)
RMSE        = Root Mean Square Error (penalizes larger errors more heavily)
    """)
    
    # Sort by recall to show detection capability
    print("\n" + "="*100)
    print("CLASSES RANKED BY RECALL (Detection Capability)")
    print("="*100)
    
    # Extract numeric recall for sorting
    metrics_df['Recall_numeric'] = metrics_df['Recall'].str.rstrip('%').astype(float)
    sorted_by_recall = metrics_df.sort_values('Recall_numeric', ascending=False)[
        ['Class', 'GT_Count', 'Pred_Count', 'Accuracy_±1', 'Recall', 'MAE', 'RMSE']
    ]
    print(sorted_by_recall.to_string(index=False))
    
    # Per-image accuracy
    print("\n" + "="*100)
    print("PER-IMAGE ACCURACY SUMMARY")
    print("="*100)
    print(f"Exact match accuracy (all classes correct):")
    print(f"  Mean: {img_acc_df['exact_match_accuracy'].mean():.1f}%")
    print(f"  Median: {img_acc_df['exact_match_accuracy'].median():.1f}%")
    print(f"  Std: {img_acc_df['exact_match_accuracy'].std():.1f}%")
    
    print(f"\nLenient accuracy (±1 object tolerance):")
    print(f"  Mean: {img_acc_df['lenient_match_accuracy'].mean():.1f}%")
    print(f"  Median: {img_acc_df['lenient_match_accuracy'].median():.1f}%")
    print(f"  Std: {img_acc_df['lenient_match_accuracy'].std():.1f}%")
    
    # Images with perfect predictions
    perfect = (img_acc_df['exact_match_accuracy'] == 100).sum()
    near_perfect = (img_acc_df['lenient_match_accuracy'] == 100).sum()
    
    print(f"\nImages with perfect predictions: {perfect}/{len(img_acc_df)} ({perfect/len(img_acc_df)*100:.1f}%)")
    print(f"Images with perfect lenient predictions: {near_perfect}/{len(img_acc_df)} ({near_perfect/len(img_acc_df)*100:.1f}%)")
    
    # Worst cases
    print(f"\nImages with lowest accuracy:")
    worst = img_acc_df.nsmallest(5, 'lenient_match_accuracy')
    for _, row in worst.iterrows():
        print(f"  {row['image_path']:40s}: {row['lenient_match_accuracy']:5.1f}%")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate YOLO predictions against ground truth')
    parser.add_argument('--ground-truth', type=str, default='ground_truth_latest.csv',
                       help='Path to ground truth CSV')
    parser.add_argument('--image-dir', type=str, default='data/images',
                       help='Directory containing images')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file')
    parser.add_argument('--threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default='results/validation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("YOLO VALIDATION AGAINST GROUND TRUTH")
    print("="*80)
    
    # Load ground truth
    print("\nLoading ground truth...")
    df_gt = load_ground_truth(args.ground_truth)
    
    # Load config and model
    print("Loading YOLO model...")
    config = load_config(args.config)
    model_config = config['models']['yolo_coco']
    model = YOLO(model_config['model_name'])
    
    # Get image paths from ground truth
    image_dir = Path(args.image_dir)
    gt_images = df_gt['image_path'].values
    image_paths = [image_dir / img for img in gt_images]
    image_paths = [p for p in image_paths if p.exists()]
    
    print(f"\nFound {len(image_paths)}/{len(gt_images)} images in directory")
    
    # Run YOLO
    print("\nRunning YOLO predictions...")
    df_pred = run_yolo_predictions(image_paths, model, args.threshold, model_config['device'])
    
    # Align dataframes
    df_gt_aligned, df_pred_aligned, classes = align_dataframes(df_gt, df_pred)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_df = calculate_metrics(df_gt_aligned, df_pred_aligned, classes)
    img_acc_df = calculate_image_accuracy(df_gt_aligned, df_pred_aligned, classes)
    
    # Print summary
    print_summary(metrics_df, img_acc_df)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_df.to_csv(output_dir / 'class_metrics.csv', index=False)
    img_acc_df.to_csv(output_dir / 'image_accuracy.csv', index=False)
    df_gt_aligned.to_csv(output_dir / 'ground_truth_aligned.csv', index=False)
    df_pred_aligned.to_csv(output_dir / 'predictions_aligned.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")

if __name__ == '__main__':
    main()
