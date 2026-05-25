#!/usr/bin/env python3
"""
Run OneFormer panoptic segmentation
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import yaml
import sys
import torch
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_inference(image_dir, output_dir, config):
    """Run OneFormer panoptic segmentation"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_config = config['models']['oneformer']
    print(f"Loading OneFormer model: {model_config['model_name']}")
    print(f"Dataset: {model_config['dataset']}")
    
    processor = OneFormerProcessor.from_pretrained(model_config['model_name'])
    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_config['model_name']
    )
    
    # Move to device
    device = 'cuda' if model_config['device'] != 'cpu' and torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Running on: {device}\n")
    
    # Get image paths
    image_dir = Path(image_dir)
    formats = config['processing']['image_formats']
    image_paths = []
    for fmt in formats:
        image_paths.extend(list(image_dir.glob(f'*{fmt}')))
    
    print(f"Found {len(image_paths)} images\n")
    
    # Process images
    results_list = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        image = Image.open(img_path)
        
        # Run panoptic segmentation
        inputs = processor(
            image,
            task_inputs=[model_config['task']],
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        result = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]
        
        # Extract counts and coverage
        panoptic_seg = result["segmentation"].cpu().numpy()
        segments_info = result["segments_info"]
        total_pixels = panoptic_seg.size
        
        thing_counts = Counter()
        stuff_coverage = {}
        
        for segment in segments_info:
            segment_id = segment["id"]
            label_id = segment["label_id"]
            class_name = model.config.id2label[label_id]
            is_thing = segment["isthing"]
            
            mask = panoptic_seg == segment_id
            pixel_count = mask.sum()
            pixel_percentage = (pixel_count / total_pixels) * 100
            
            if is_thing:
                thing_counts[class_name] += 1
            else:
                if class_name not in stuff_coverage:
                    stuff_coverage[class_name] = 0
                stuff_coverage[class_name] += pixel_percentage
        
        # Combine results
        row = {'image_path': img_path.name}
        
        # Add thing counts with prefix
        for class_name, count in thing_counts.items():
            row[f'count_{class_name}'] = count
        
        # Add stuff coverage with prefix
        for class_name, coverage in stuff_coverage.items():
            row[f'coverage_{class_name}'] = coverage
        
        results_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results_list).fillna(0)
    
    # Save results
    output_file = output_dir / 'detections.csv'
    df.to_csv(output_file, index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - OneFormer Panoptic Segmentation")
    print(f"{'='*60}")
    print(f"Total images processed: {len(df)}")
    
    # Thing counts
    count_cols = [c for c in df.columns if c.startswith('count_')]
    if count_cols:
        print(f"\nCountable objects (things):")
        for col in count_cols:
            total = df[col].sum()
            print(f"  {col.replace('count_', '')}: {int(total)}")
    
    # Stuff coverage
    coverage_cols = [c for c in df.columns if c.startswith('coverage_')]
    if coverage_cols:
        print(f"\nAverage pixel coverage (stuff):")
        for col in coverage_cols:
            avg = df[col].mean()
            print(f"  {col.replace('coverage_', '')}: {avg:.2f}%")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Run OneFormer panoptic segmentation'
    )
    parser.add_argument('--image-dir', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--dataset', type=str, choices=['cityscapes', 'coco', 'ade20k'])
    parser.add_argument('--config', type=str, default='config.yaml')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Override dataset if specified
    if args.dataset:
        config['models']['oneformer']['dataset'] = args.dataset
        # Update model name based on dataset
        dataset_models = {
            'cityscapes': 'shi-labs/oneformer_cityscapes_swin_large',
            'coco': 'shi-labs/oneformer_coco_swin_large',
            'ade20k': 'shi-labs/oneformer_ade20k_swin_large'
        }
        config['models']['oneformer']['model_name'] = dataset_models[args.dataset]
    
    image_dir = args.image_dir or config['paths']['image_dir']
    output_dir = args.output or Path(config['paths']['results_dir']) / 'oneformer'
    
    run_inference(image_dir, output_dir, config)


if __name__ == '__main__':
    main()