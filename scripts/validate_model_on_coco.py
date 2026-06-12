#Validate_model_on_coco

from ultralytics import YOLO

model = YOLO('yolo26x.pt')

# Just show the model info
model.info()

# Quick inference on a few sample images
results = model.predict(
    source=[
        'https://ultralytics.com/images/bus.jpg',
        'https://ultralytics.com/images/zidane.jpg',
    ],
    device=0,  # CPU only
    conf=0.25
)

print(f"Sample inference works: {len(results)} images processed")