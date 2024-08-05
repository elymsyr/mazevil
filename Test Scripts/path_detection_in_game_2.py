import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
from array_scale import downscale_binary_array, upscale_binary_array

window_title = 'Mazevil'
scale_order = [9, 3]

lower_bound = np.array([100, 50, 50])
upper_bound = np.array([255, 150, 150])

def path_detection(rgb, downscale_order = [3,3,3], lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    binary_array = (mask > 0).astype(np.uint8)
    for downscale in downscale_order:
        binary_array = downscale_binary_array(binary_array, downscale)
    return binary_array

def capture_window(window, top_crop=20, bottom_crop=50):
    avg = 8
    bbox = (window.left+avg, window.top+avg+22, window.right-avg, window.bottom-avg)
    img = ImageGrab.grab(bbox)
    matlike_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return matlike_image[top_crop:-bottom_crop, :]

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
    binary_array = path_detection(rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), downscale_order=scale_order, lower_bound=lower_bound, upper_bound=upper_bound)
    # print(binary_array.shape) # 14,29
    for scale in scale_order:
        binary_array = upscale_binary_array(binary_array, scale)

    cv2.imshow('paths', (binary_array * 255).astype(np.uint8))
    # cv2.imshow('paths', window_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()