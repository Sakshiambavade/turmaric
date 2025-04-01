from ultralytics import YOLO

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")

# Train on turmeric dataset
results = model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cuda"  # Use "cpu" if no GPU available
)

# Save trained model
model_path = "best_model.pt"
model.export(format="onnx")  # Export to ONNX for deployment
print(f"Model saved at {model_path}")
