#!/usr/bin/env python3
"""
Visualize YOLO detections by drawing bounding boxes on images
"""

import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

def draw_detections(image_path, detections_df, output_dir, class_colors=None):
    """
    Draw bounding boxes on an image and save it
    
    Args:
        image_path: Path to image file
        detections_df: DataFrame with detections_detailed.csv data
        output_dir: Directory to save visualized images
        class_colors: Dict mapping class names to BGR colors
    """
    
    img_name = Path(image_path).name
    img_name_norm = img_name.replace('.jpg', '').replace('.jpeg', '')
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    
    height, width = img.shape[:2]
    
    # Get detections for this image
    img_detections = detections_df[
        detections_df['image_path'].str.replace('.jpg', '').str.replace('.jpeg', '') == img_name_norm
    ]
    
    if len(img_detections) == 0:
        # No detections, still save
        output_path = Path(output_dir) / f"viz_{img_name}"
        cv2.imwrite(str(output_path), img)
        return True
    
    # Default colors for classes
    if class_colors is None:
        class_colors = {}
    
    # Draw bounding boxes
    for _, det in img_detections.iterrows():
        x1 = int(det['bbox_x1'])
        y1 = int(det['bbox_y1'])
        x2 = int(det['bbox_x2'])
        y2 = int(det['bbox_y2'])
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Get color for class (or random if not specified)
        if class_name in class_colors:
            color = class_colors[class_name]
        else:
            # Generate color from class name hash
            h = hash(class_name)
            b = (h % 256)
            g = ((h // 256) % 256)
            r = ((h // 65536) % 256)
            color = (b, g, r)
            class_colors[class_name] = color
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
    
    # Save visualized image
    output_path = Path(output_dir) / f"viz_{img_name}"
    cv2.imwrite(str(output_path), img)
    
    return True

def visualize_detections(image_dir, detections_file, output_dir):
    """
    Create visualized images with bounding boxes for all detections
    """
    
    print("="*80)
    print("VISUALIZING YOLO DETECTIONS")
    print("="*80)
    
    # Load detections
    df_detections = pd.read_csv(detections_file)
    print(f"\nLoaded {len(df_detections)} detections from {detections_file}")
    
    # Get unique images
    images_with_detections = df_detections['image_path'].unique()
    print(f"Images with detections: {len(images_with_detections)}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Draw detections
    print(f"\nDrawing bounding boxes...")
    successful = 0
    failed = 0
    
    for img_name in tqdm(images_with_detections):
        img_path = Path(image_dir) / img_name
        
        if not img_path.exists():
            # Try without extension variations
            for fmt in ['.jpg', '.jpeg', '.png']:
                alt_path = Path(image_dir) / (img_name + fmt)
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        if img_path.exists():
            if draw_detections(img_path, df_detections, output_dir):
                successful += 1
            else:
                failed += 1
        else:
            failed += 1
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Successfully visualized: {successful} images")
    print(f"Failed: {failed} images")
    print(f"Saved to: {output_dir}")
    print(f"\nView images:")
    print(f"  open {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO detections')
    parser.add_argument('--image-dir', type=str, default='data/images',
                       help='Directory containing images')
    parser.add_argument('--detections', type=str, default='results/test_yolo26x/detections_detailed.csv',
                       help='Path to detections_detailed.csv file')
    parser.add_argument('--output', type=str, default='results/test_yolo26x/visualizations',
                       help='Output directory for visualized images')
    
    args = parser.parse_args()
    
    visualize_detections(args.image_dir, args.detections, args.output)

if __name__ == '__main__':
    main()
