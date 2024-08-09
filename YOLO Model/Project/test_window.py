from time import perf_counter
import time
import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
import torch
from detect import *
from path import path_detection
from torch.hub import load


def capture_window(window: gw.Win32Window, top_crop=0, bottom_crop=1):
    avg = 8
    extra_top_avg = 22
    bbox = (window.left + avg, window.top + avg + extra_top_avg, window.right - avg, window.bottom - avg)
    img = ImageGrab.grab(bbox)
    matlike_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return matlike_image[top_crop:-bottom_crop, :]

# Load YOLOv7 model
model, device = model_load(path_or_model='YOLO Model/Model/Trained Models/testmodel02_yolov7-8-100.pt')
# model = load('WongKinYiu/yolov7', 'custom', path_or_model='YOLO Model\\Model\\Trained Models\\testmodel02_yolov7-8-100.pt')
# model = model.autoshape()  # for PIL/cv2/np inputs and NMS



# Set model parameters
model.conf = 0.3  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold

lower_bound = np.array([100, 50, 50])
upper_bound = np.array([255, 150, 150])

# Select the window to capture
window_title = 'Mazevil'
window = gw.getWindowsWithTitle(window_title)[0]
fps_list = []
prevTime = 0
while True:
    if len(fps_list) > 100: break
    
    currTime = perf_counter()
    fps = 1 / (currTime - prevTime)
    fps_list.append(fps)
    prevTime = currTime
    
    img = capture_window(window)
    
    prediction = model(img)  # includes NMS
    prediction.print()    

    if cv2.waitKey(1) == 27:
        break

print(f"Avg Fps: {sum(fps_list)/len(fps_list)}")

# Avg FPS: 0.7097454746069215

cv2.destroyAllWindows()