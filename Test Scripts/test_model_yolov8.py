from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('YOLO Model\\Model\\Trained Models\\test_0\\train\\weights\\best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('YOLO Model\\Data\\Test\\screenshot_411.png', imgsz=480)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
