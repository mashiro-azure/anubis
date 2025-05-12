import os
import random
from ultralytics import YOLO

# Configuration variables
DATASET_DIR = "./coco/val2017"  # Folder with your test images
OPENVINO_MODEL_PATH = "./yolo11s_int8_openvino_model/"  # Path to your exported OpenVINO model

# Load the OpenVINO model using the native Ultralytics YOLO class.
# The model automatically detects its type (e.g., OpenVINO) from the file extension.
model = YOLO(OPENVINO_MODEL_PATH, task="detect")

# Get a list of image files from the dataset directory.
image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if not image_files:
    raise Exception(f"No image files found in {DATASET_DIR}.")

# Pick a random image.
random_image_path = os.path.join(DATASET_DIR, random.choice(image_files))
print(f"Randomly selected image: {random_image_path}")

# Run detection using the native predict() method.
# The "show=True" parameter will automatically display the resulting annotated image.
results = model.predict(source=random_image_path, show=True, classes=[0])
print(results[0].boxes)
