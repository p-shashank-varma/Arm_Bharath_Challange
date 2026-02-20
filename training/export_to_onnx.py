from ultralytics import YOLO

# Load your newly trained model
model = YOLO('/content/runs/detect/train/weights/best.pt') #Give your best.pt file path here

# Export to ONNX specifically for FPGA/Hardware
# opset=12 is recommended for the best compatibility with Xilinx tools
model.export(format='onnx', imgsz=320, opset=12)
