#!/usr/bin/env python3
"""
Run YOLO (Open Images V7) inference
Filters output to classes of interest only
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
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_inference(image_dir, output_dir, config):
    """Run YOLO Open Images V7 inference"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_config = config['models']['yolo_openimages']
    print(f"Loading YOLO Open Images V7 model: {model_config['model_name']}")
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
    print(f"Running inference on {model_config['device']}...\n")
    
    # Process images
    results_list = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        results = model.predict(
            source=str(img_path),
            conf=model_config['conf_threshold'],
            device=model_config['device'],
            verbose=False
        )[0]
        
        # Count objects by class (with filtering)
        class_counts = Counter()
        if results.boxes is not None:
            for box in results.boxes:
                class_name = model.names[int(box.cls[0])]
                
                # Filter to classes of interest if specified
                if classes_of_interest is None or class_name in classes_of_interest:
                    class_counts[class_name] += 1
        
        row = {'image_path': img_path.name}
        row.update(dict(class_counts))
        results_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results_list).fillna(0)
    count_cols = [c for c in df.columns if c != 'image_path']
    df[count_cols] = df[count_cols].astype(int)
    
    # Ensure all classes of interest are present as columns (even if zero)
    if classes_of_interest:
        for cls in classes_of_interest:
            if cls not in df.columns:
                df[cls] = 0
    
    # Save results
    output_file = output_dir / 'detections.csv'
    df.to_csv(output_file, index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - YOLO Open Images V7")
    print(f"{'='*60}")
    print(f"Total images processed: {len(df)}")
    
    # Highlight Tree detection if present
    if 'Tree' in df.columns:
        tree_count = df['Tree'].sum()
        images_with_trees = (df['Tree'] > 0).sum()
        pct_with_trees = (images_with_trees / len(df)) * 100
        avg_trees = tree_count / len(df)
        print(f"\n🌳 Tree Detection:")
        print(f"  Total trees: {int(tree_count)}")
        print(f"  Images with trees: {images_with_trees} ({pct_with_trees:.1f}%)")
        print(f"  Average trees/image: {avg_trees:.2f}")
    
    if count_cols:
        print(f"\nAll detected classes (sorted by frequency):")
        totals = df[count_cols].sum().sort_values(ascending=False)
        for class_name, count in totals.items():
            avg_per_image = count / len(df)
            images_with_class = (df[class_name] > 0).sum()
            pct_images = (images_with_class / len(df)) * 100
            print(f"  {class_name}:")
            print(f"    Total: {int(count)} | Avg/image: {avg_per_image:.2f} | In {pct_images:.1f}% of images")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Run YOLO (Open Images V7) inference'
    )
    parser.add_argument('--image-dir', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--config', type=str, default='config.yaml')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    image_dir = args.image_dir or config['paths']['image_dir']
    output_dir = args.output or Path(config['paths']['results_dir']) / 'yolo_openimages'
    
    run_inference(image_dir, output_dir, config)


if __name__ == '__main__':
    main()