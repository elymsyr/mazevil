import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
import numpy as np
import cv2
import tensorflow as tf
from tflite_detect_image import tflite_detection
from array_downscale import downscale_binary_array

window_title = 'Mazevil'

    # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def path_detection(rgb, downscale_order = [3,3,3], lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    binary_array = (mask > 0).astype(np.uint8)
    for downscale in downscale_order:
        binary_array = downscale_binary_array(binary_array, downscale)
    return binary_array

def capture_window(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    bbox = (window.left, window.top, window.right, window.bottom)
    img = ImageGrab.grab(bbox)
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

while True:
    window_image = capture_window(window_title)
    binary_array = path_detection(rgb=window_image, downscale_order=[])
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()