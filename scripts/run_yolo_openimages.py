#!/usr/bin/env python3
"""
Run YOLO (Open Images V7) inference for tree detection
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
    
    # Get image paths
    image_dir = Path(image_dir)
    formats = config['processing']['image_formats']
    image_paths = []
    for fmt in formats:
        image_paths.extend(list(image_dir.glob(f'*{fmt}')))
    
    print(f"\nFound {len(image_paths)} images")
    print(f"Running inference on {model_config['device']}...")
    
    # Classes of interest (if specified)
    classes_of_interest = model_config.get('classes_of_interest', None)
    if classes_of_interest:
        print(f"Filtering for classes: {', '.join(classes_of_interest)}")
    print()
    
    # Process images
    results_list = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
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
                
                # Filter if classes_of_interest specified
                if classes_of_interest is None or class_name in classes_of_interest:
                    class_counts[class_name] += 1
        
        row = {'image_path': img_path.name}
        row.update(dict(class_counts))
        results_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results_list).fillna(0)
    count_cols = [c for c in df.columns if c != 'image_path']
    df[count_cols] = df[count_cols].astype(int)
    
    # Save results
    output_file = output_dir / 'detections.csv'
    df.to_csv(output_file, index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Open Images V7")
    print(f"{'='*60}")
    print(f"Total images processed: {len(df)}")
    
    if 'Tree' in df.columns:
        tree_count = df['Tree'].sum()
        images_with_trees = (df['Tree'] > 0).sum()
        print(f"\nTree Detection:")
        print(f"  Total trees detected: {int(tree_count)}")
        print(f"  Images with trees: {images_with_trees} ({images_with_trees/len(df)*100:.1f}%)")
        print(f"  Average trees per image: {tree_count/len(df):.2f}")
    
    print(f"\nAll detected classes:")
    totals = df[count_cols].sum().sort_values(ascending=False)
    for class_name, count in totals.items():
        print(f"  {class_name}: {int(count)}")
    
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