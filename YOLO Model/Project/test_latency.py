import cv2
import numpy as np
import tensorflow as tf
import pygetwindow as gw
from PIL import ImageGrab
from time import perf_counter
from model import model, model_detection
from path import path_detection

def upscale_binary_array(array, alpha):
    # Check if alpha is an integer
    if not isinstance(alpha, int) or alpha <= 0:
        raise ValueError("Alpha must be a positive integer.")
    
    # Get the shape of the original array
    old_shape = array.shape
    
    # Compute the new shape
    new_shape = (old_shape[0] * alpha, old_shape[1] * alpha)
    
    # Create an empty array with the new shape
    upscaled_array = np.zeros(new_shape, dtype=array.dtype)
    
    # Fill the upscaled array
    for i in range(old_shape[0]):
        for j in range(old_shape[1]):
            upscaled_array[i*alpha:(i+1)*alpha, j*alpha:(j+1)*alpha] = array[i, j]
    
    return upscaled_array

def capture_window(window: gw.Win32Window, top_crop=0, bottom_crop=1):
    avg = 8
    extra_top_avg = 22
    bbox = (window.left+avg, window.top+avg+extra_top_avg, window.right-avg, window.bottom-avg)
    img = ImageGrab.grab(bbox)
    matlike_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return matlike_image[top_crop:-bottom_crop, :]

def window(model_path: str, lblpath: str, show: bool = True, scale_order: list = [9,3], min_conf: float = 0.4, window_title = 'Mazevil', full_title = 'Mazevil', lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    
    
    prevTime = 0
    fps = 0
    fpsText = 0
    counter = 0
    
    
    lower_bound = np.array([100, 50, 50])
    upper_bound = np.array([255, 150, 150])
    inter_values = model(model_path = model_path, lblpath = lblpath)
    
    windows = gw.getWindowsWithTitle(window_title)
    window = windows[0]
    window_image = capture_window(window, top_crop=20, bottom_crop=50)
    imH, imW, _ = window_image.shape

    # for item in windows:
    #     if item.title == full_title: window = item

    # np.set_printoptions(precision=6, suppress=True)

    fps_list = []

    while True:
        window_image = capture_window(window, top_crop=20, bottom_crop=50)

        # *_, environment = model_detection(image=window_image, inter_values=inter_values, min_conf=min_conf)

        # binary_array = path_detection(rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), downscale_order=scale_order, lower_bound=lower_bound, upper_bound=upper_bound)

        currTime = perf_counter()
        fps = 1 / (currTime - prevTime)
        counter += 1
        fps_list.append(fps)
        prevTime = currTime
        if len(fps_list) > 100: break
        
        cv2.imshow('CV2', window_image)
        cv2.waitKey(1)
        
    print(f"Avg FPS: {sum(fps_list) / len(fps_list)}")

    # (both) Avg FPS: 3.888680504014728
    # (only binary_array) Avg FPS: 7.937991522340554
    # (only environment) Avg FPS: 5.4663052496469176
    # (nothing) Avg FPS: 19.854902853796332
    # (only imshow) Avg FPS: 17.83412855280484

window(model_path = 'Model\\test_model_001\\detect.tflite', lblpath = 'Model\\test_model_001\\labelmap.txt')
