


from scripts.run_yolo_coco import run_inference, load_config

print("Starting test...")
config = load_config()
print("✓ Config loaded")

image_dir = "data/images"
output_dir = "results/test_yolo26x"

print(f"Running inference on 20 images...")
try:
    run_inference(image_dir, output_dir, config, sample=20)
    print("✓ Complete!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
EOF
