import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
import numpy as np
from time import sleep
import cv2
from array_scale import downscale_binary_array, upscale_binary_array

window_title = 'Mazevil'
scale_order = [3, 3, 3]

def path_detection(rgb, downscale_order = [3,3,3], lower_bound = np.array([89,51,53]), upper_bound = np.array([228,166,114])):
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    binary_array = (mask > 0).astype(np.uint8)
    for downscale in downscale_order:
        binary_array = downscale_binary_array(binary_array, downscale)
    return binary_array

def capture_window(window):
    bbox = (window.left, window.top, window.right, window.bottom)
    img = ImageGrab.grab(bbox)
    matlike_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(matlike_image, cv2.COLOR_BGR2RGB)
    # return matlike_image

windows = gw.getWindowsWithTitle(window_title)

# for id, window in enumerate(windows):
#     print(f"{id} - {window}")
# id = input("Choose window... ")
# window = windows[id]

window = windows[0]

for item in windows:
    if item.title == 'Mazevil': window = item

while True:
    window_image = capture_window(window)
    binary_array = path_detection(rgb=window_image, downscale_order=scale_order)

    for scale in scale_order:
        binary_array = upscale_binary_array(binary_array, scale)

    cv2.imshow('paths', (binary_array * 255).astype(np.uint8))
    # cv2.imshow('paths', window_image)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()