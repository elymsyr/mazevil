import cv2
import numpy as np

def path_detection(rgb, lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    mask = (mask > 0).astype(np.uint8)
    return np.stack([mask, mask, mask], axis=-1) * 255