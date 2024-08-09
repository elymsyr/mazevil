from time import perf_counter
import time
import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
import torch
from detect import *
from path import path_detection


def capture_window(window: gw.Win32Window, top_crop=0, bottom_crop=1):
    avg = 8
    extra_top_avg = 22
    bbox = (window.left + avg, window.top + avg + extra_top_avg, window.right - avg, window.bottom - avg)
    img = ImageGrab.grab(bbox)
    matlike_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return matlike_image[top_crop:-bottom_crop, :]

# Load YOLOv7 model
model, device = model_load(path_or_model='YOLO Model/Model/Trained Models/testmodel02_yolov7-8-100.pt')
# model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='YOLO Model\\Model\\Trained Models\\testmodel02_yolov7-8-100.pt')
# model = model.autoshape()  # for PIL/cv2/np inputs and NMS


# Set model parameters
model.conf = 0.3  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold

lower_bound = np.array([100, 50, 50])
upper_bound = np.array([255, 150, 150])

# Select the window to capture
window_title = 'Mazevil'
window = gw.getWindowsWithTitle(window_title)[0]
capture_time = []
detection_time = []
path_time = []
show_time = []
fps_list = []
prevTime = 0
fps = 0
fpsText = 0
counter = 0
while True:
    currTime = perf_counter()
    fps = 1 / (currTime - prevTime)
    counter += 1
    fps_list.append(fps)
    prevTime = currTime    

    t1 = time.perf_counter()
    img = capture_window(window)
    t2 = time.perf_counter()
    capture_time.append(t2-t1)
    t2 = time.perf_counter()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        prediction = model(img)    
    t3 = time.perf_counter()
    detection_time.append(t3-t2)
    t3 = time.perf_counter()
    binary_array = path_detection(rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), downscale_order=[9, 3], lower_bound=lower_bound, upper_bound=upper_bound)
    t4 = time.perf_counter()
    path_time.append(t4-t3)
    t4 = time.perf_counter()
    
    # img = display(prediction.imgs, prediction.pred, prediction.names)

    t5 = time.perf_counter()
    show_time.append(t5-t4)
    if len(show_time) > 20 : break

    cv2.imshow('YOLOv7 Object Detection', img)
    if cv2.waitKey(1) == 27:
        break

print(f"Avg Fps: {sum(fps_list)/len(fps_list)}\nAvg counter times:\n  Capture time: {(sum(capture_time)/len(capture_time)):.4f}, {max(capture_time)}, {min(capture_time)}\n  Detection time: {(sum(detection_time)/len(detection_time)):.4f}, {max(detection_time)}, {min(detection_time)}\n  Path time: {(sum(path_time)/len(path_time)):.4f}, {max(path_time)}, {min(path_time)}\n  Show time: {(sum(show_time)/len(show_time)):.4f}")

# Avg FPS: 0.7097454746069215

cv2.destroyAllWindows()