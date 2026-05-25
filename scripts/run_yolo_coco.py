#!/usr/bin/env python3
"""
Run YOLO (COCO) inference on street view images
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter
import yaml
import sys

# Add parent directory to path
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
    
    # Get image paths
    image_dir = Path(image_dir)
    formats = config['processing']['image_formats']
    image_paths = []
    for fmt in formats:
        image_paths.extend(list(image_dir.glob(f'*{fmt}')))
    
    print(f"\nFound {len(image_paths)} images")
    print(f"Running inference on {model_config['device']}...")
    print(f"Confidence threshold: {model_config['conf_threshold']}\n")
    
    # Process images
    results_list = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=model_config['conf_threshold'],
            device=model_config['device'],
            verbose=False
        )[0]
        
        # Count objects by class
        class_counts = Counter()
        if results.boxes is not None:
            for box in results.boxes:
                class_name = model.names[int(box.cls[0])]
                class_counts[class_name] += 1
        
        # Store result
        row = {'image_path': img_path.name}
        row.update(dict(class_counts))
        results_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results_list).fillna(0)
    
    # Convert counts to int
    count_cols = [c for c in df.columns if c != 'image_path']
    df[count_cols] = df[count_cols].astype(int)
    
    # Save results
    output_file = output_dir / 'detections.csv'
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(df)}")
    print(f"Classes detected: {len(count_cols)}")
    print(f"\nTop 10 most frequent objects:")
    totals = df[count_cols].sum().sort_values(ascending=False)
    for class_name, count in totals.head(10).items():
        print(f"  {class_name}: {int(count)}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return df


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
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Use config defaults if args not provided
    image_dir = args.image_dir or config['paths']['image_dir']
    output_dir = args.output or Path(config['paths']['results_dir']) / 'yolo_coco'
    
    # Run inference
    run_inference(image_dir, output_dir, config)


if __name__ == '__main__':
    main()