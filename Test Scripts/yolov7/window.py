import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
import torch
from detect import *


def capture_window(window: gw.Win32Window, top_crop=0, bottom_crop=1):
    avg = 8
    extra_top_avg = 22
    bbox = (window.left + avg, window.top + avg + extra_top_avg, window.right - avg, window.bottom - avg)
    img = ImageGrab.grab(bbox)
    matlike_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return matlike_image[top_crop:-bottom_crop, :]

# Load YOLOv7 model
model, device = model_load(path_or_model='YOLO Model/Model/Trained Models/testmodel02_yolov7-8-100.pt')

# Set model parameters
model.conf = 0.3  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold

# Select the window to capture
window_title = 'Mazevil'
window = gw.getWindowsWithTitle(window_title)[0]

while True:
    # Load and preprocess image
    img = capture_window(window)

    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        prediction = model(img)


    img = display(prediction.imgs, prediction.pred, prediction.names)

    cv2.imshow('YOLOv7 Object Detection', img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()