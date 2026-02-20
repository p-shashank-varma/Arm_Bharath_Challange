from ultralytics import YOLO

# Load the Nano model for PYNQ-Z2 compatibility
model = YOLO('yolov8n.pt') 

# Train using the fixed cloud path
model.train(
    data=dataset.location, #In place of dataset.location give your path where the dataset is loacted
    epochs=50, 
    imgsz=320,  # Keep at 320 for FPGA memory constraints
    device=0    # Uses the cloud GPU
)
