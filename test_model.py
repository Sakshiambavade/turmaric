from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("best_model.pt")

# Read test image
image_path = "test_image.jpg"
img = cv2.imread(image_path)

# Run inference
results = model(img, show=True)

# Display detections
for r in results:
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
