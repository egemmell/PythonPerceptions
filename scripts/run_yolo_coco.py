#!/usr/bin/env python3
"""
Run YOLO (COCO) inference on street view images
Saves both summary counts and detailed per-detection information
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_inference(image_dir, output_dir, config):
    """Run YOLO COCO inference on all images"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_config = config['models']['yolo_coco']
    print(f"Loading YOLO model: {model_config['model_name']}")
    model = YOLO(model_config['model_name'])
    
    # Get classes of interest
    classes_of_interest = model_config.get('classes_of_interest', None)
    if classes_of_interest:
        print(f"Filtering to {len(classes_of_interest)} classes of interest:")
        for cls in classes_of_interest:
            print(f"  - {cls}")
    else:
        print("No filtering - outputting all detected classes")
    
    # Get image paths
    image_dir = Path(image_dir)
    formats = config['processing']['image_formats']
    image_paths = []
    for fmt in formats:
        image_paths.extend(list(image_dir.glob(f'*{fmt}')))
    
    print(f"\nFound {len(image_paths)} images")
    print(f"Running inference on {model_config['device']}...")
    print(f"Confidence threshold: {model_config['conf_threshold']}\n")
    
    # Storage for results
    summary_list = []      # Summary: counts per image
    detailed_list = []     # Detailed: one row per detection
    
    # Process images
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=model_config['conf_threshold'],
            device=model_config['device'],
            verbose=False
        )[0]
        
        # Count objects by class (with filtering)
        class_counts = Counter()
        
        # Process each detection
        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                xywh = box.xywh[0].tolist()  # [x_center, y_center, width, height]
                
                # Filter to classes of interest if specified
                if classes_of_interest is None or class_name in classes_of_interest:
                    class_counts[class_name] += 1
                    
                    # Save detailed detection
                    detailed_list.append({
                        'image_path': img_path.name,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox_x1': xyxy[0],
                        'bbox_y1': xyxy[1],
                        'bbox_x2': xyxy[2],
                        'bbox_y2': xyxy[3],
                        'bbox_center_x': xywh[0],
                        'bbox_center_y': xywh[1],
                        'bbox_width': xywh[2],
                        'bbox_height': xywh[3]
                    })
        
        # Store summary result
        row = {'image_path': img_path.name}
        row.update(dict(class_counts))
        summary_list.append(row)
    
    # Create summary DataFrame (counts per image)
    df_summary = pd.DataFrame(summary_list).fillna(0)
    count_cols = [c for c in df_summary.columns if c != 'image_path']
    df_summary[count_cols] = df_summary[count_cols].astype(int)
    
    # Ensure all classes of interest are present as columns (even if zero)
    if classes_of_interest:
        for cls in classes_of_interest:
            if cls not in df_summary.columns:
                df_summary[cls] = 0
    
    # Create detailed DataFrame (one row per detection)
    df_detailed = pd.DataFrame(detailed_list)
    
    # Save both files
    summary_file = output_dir / 'detections_summary.csv'
    detailed_file = output_dir / 'detections_detailed.csv'
    
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    if len(df_detailed) > 0:
        df_detailed.to_csv(detailed_file, index=False)
        print(f"Detailed detections saved to: {detailed_file}")
    else:
        print("No detections to save in detailed file")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY - YOLO COCO")
    print(f"{'='*60}")
    print(f"Total images processed: {len(df_summary)}")
    print(f"Total detections: {len(df_detailed)}")
    print(f"Classes in output: {len(count_cols)}")
    
    if count_cols:
        print(f"\nDetection statistics (sorted by frequency):")
        totals = df_summary[count_cols].sum().sort_values(ascending=False)
        for class_name, count in totals.items():
            avg_per_image = count / len(df_summary)
            images_with_class = (df_summary[class_name] > 0).sum()
            pct_images = (images_with_class / len(df_summary)) * 100
            
            # Calculate confidence stats from detailed data
            if class_name in df_detailed['class_name'].values:
                class_detections = df_detailed[df_detailed['class_name'] == class_name]
                avg_conf = class_detections['confidence'].mean()
                min_conf = class_detections['confidence'].min()
                max_conf = class_detections['confidence'].max()
                
                print(f"  {class_name}:")
                print(f"    Total: {int(count)} | Avg/image: {avg_per_image:.2f} | In {pct_images:.1f}% of images")
                print(f"    Confidence: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}")
    
    print(f"\n{'='*60}\n")
    
    return df_summary, df_detailed


def main():
    parser = argparse.ArgumentParser(
        description='Run YOLO (COCO) inference on images'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='YOLO model (e.g., yolo11m.pt, yolo8x.pt, yolo11l.pt). Overrides config.'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Use command-line model if specified, otherwise use config
    if args.model:
        model_name = args.model
        print(f"Using model from command line: {model_name}")
        config['models']['yolo_coco']['model_name'] = model_name
    
    # Use config defaults if args not provided
    image_dir = args.image_dir or config['paths']['image_dir']
    output_dir = args.output or Path(config['paths']['results_dir']) / 'yolo_coco'
    
    # Run inference
    run_inference(image_dir, output_dir, config)