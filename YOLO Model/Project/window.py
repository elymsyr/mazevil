import cv2
import numpy as np
import mss
import torch

# Function to capture a part of the screen
def capture_screen(bbox):
    with mss.mss() as sct:
        img = np.array(sct.grab(bbox))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

# Define the bounding box (bbox) for the part of the screen to capture
bbox = (0, 0, 800, 600)

# Load YOLOv7 model
model = torch.load('YOLO Model\\Model\\Trained Models\\testmodel02_yolov7-8-100.pt')
model.eval()  # Set the model to evaluation mode

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to preprocess the image for YOLOv7
def preprocess(image, img_size=640):
    img = cv2.resize(image, (img_size, img_size))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float32)
    return img

# Function to post-process the detections
def postprocess(detections, original_image, img_size=640, conf_threshold=0.25):
    h, w, _ = original_image.shape
    ratio = min(img_size / w, img_size / h)
    padding_w = (img_size - w * ratio) / 2
    padding_h = (img_size - h * ratio) / 2

    detections = detections[0]
    detections = detections[detections[:, 4] > conf_threshold]

    bboxes = []
    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        conf = detection[4]
        cls = detection[5]

        x1 = (x1 - padding_w) / ratio
        x2 = (x2 - padding_w) / ratio
        y1 = (y1 - padding_h) / ratio
        y2 = (y2 - padding_h) / ratio

        bboxes.append((x1, y1, x2, y2, conf, cls))

    return bboxes

# Real-time object detection loop
while True:
    # Capture the screen
    frame = capture_screen(bbox)

    # Preprocess the image for YOLOv7
    img = preprocess(frame).to(device)

    # Run the model
    with torch.no_grad():
        detections = model(img)

    # Post-process the detections
    bboxes = postprocess(detections, frame)

    # Draw bounding boxes and labels on the frame
    for bbox in bboxes:
        x1, y1, x2, y2, conf, cls = bbox
        label = f'{int(cls)} {conf:.2f}'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the result
    cv2.imshow('YOLOv7 Object Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
