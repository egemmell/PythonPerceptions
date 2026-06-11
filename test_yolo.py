"""
Quick test script for YOLO COCO inference with class filtering
"""

from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import pandas as pd
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configuration from config file
IMAGE_DIR = config['paths']['image_dir']
NUM_TEST_IMAGES = 5
model_config = config['models']['yolo_coco']
CONF_THRESHOLD = model_config['conf_threshold']
DEVICE = model_config['device']
CLASSES_OF_INTEREST = model_config.get('classes_of_interest', None)

def test_yolo():
    print("="*60)
    print("YOLO COCO Test Script (with class filtering)")
    print("="*60)
    
    # Load model
    print("\n1. Loading YOLO model...")
    model = YOLO(model_config['model_name'])
    print("   ✓ Model loaded successfully")
    
    # Show filtering info
    if CLASSES_OF_INTEREST:
        print(f"\n   Filtering to {len(CLASSES_OF_INTEREST)} classes:")
        for cls in CLASSES_OF_INTEREST:
            print(f"     - {cls}")
    
    # Get test images
    print(f"\n2. Finding test images in {IMAGE_DIR}...")
    image_dir = Path(IMAGE_DIR)
    
    if not image_dir.exists():
        print(f"   ✗ ERROR: Directory '{IMAGE_DIR}' not found!")
        return
    
    image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg'))
    
    if len(image_paths) == 0:
        print(f"   ✗ ERROR: No images found in {IMAGE_DIR}")
        return
    
    test_images = image_paths[:NUM_TEST_IMAGES]
    print(f"   ✓ Found {len(image_paths)} total images")
    print(f"   ✓ Testing on first {len(test_images)} images")
    
    # Run inference
    print(f"\n3. Running inference (device={DEVICE})...")
    results_list = []
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n   Image {i}/{len(test_images)}: {img_path.name}")
        
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False
        )[0]
        
        # Count detections (with filtering)
        class_counts = Counter()
        if results.boxes is not None:
            for box in results.boxes:
                class_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                # Filter to classes of interest
                if CLASSES_OF_INTEREST is None or class_name in CLASSES_OF_INTEREST:
                    class_counts[class_name] += 1
                    print(f"      ✓ {class_name}: {conf:.2f}")
                else:
                    print(f"      ✗ {class_name}: {conf:.2f} (filtered out)")
        else:
            print(f"      - No objects detected")
        
        row = {'image': img_path.name, 'total_objects': sum(class_counts.values())}
        row.update(dict(class_counts))
        results_list.append(row)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    df = pd.DataFrame(results_list).fillna(0)
    print("\nDetections per image:")
    print(df.to_string(index=False))
    
    count_cols = [c for c in df.columns if c not in ['image', 'total_objects']]
    if count_cols:
        print("\nTotal detections across all test images:")
        totals = df[count_cols].sum().sort_values(ascending=False)
        for class_name, count in totals.items():
            if count > 0:
                print(f"  {class_name}: {int(count)}")
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)

if __name__ == '__main__':
    test_yolo()