from ultralytics import YOLO
import os

# Load your best trained model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Run inference on your test images
# source: path to your test folder or a single image/video URL
results = model.predict(source=f"{dataset.location}/test/images", # Here for source give the path or the url of the image you want to test
                        conf=0.25,      # Minimum confidence to show a detection
                        save=True)     # Saves results to runs/detect/predict/

print(" Inference complete. Results saved in 'runs/detect/predict'")
