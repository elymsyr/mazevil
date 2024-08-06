import cv2
import numpy as np
import tensorflow as tf
import pygetwindow as gw
from PIL import ImageGrab
from model import model, model_detection
from path import path_detection

def capture_window(window: gw.Win32Window, top_crop=0, bottom_crop=1):
    return cv2.cvtColor(np.array(ImageGrab.grab(bbox=(window.left+8, window.top+8+22, window.right-8, window.bottom-8))), cv2.COLOR_BGR2RGB)[top_crop:-bottom_crop, :]

def window(model_path: str, lblpath: str, show: bool = True, scale_order: list = [9,3], min_conf: float = 0.4, window_title = 'Mazevil', full_title = 'Mazevil', lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
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

    while True:
        window_image = capture_window(window, top_crop=20, bottom_crop=50)

        boxes, classes, scores, environment = model_detection(image=window_image, inter_values=inter_values, min_conf=min_conf)

        binary_array = path_detection(rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), downscale_order=scale_order, lower_bound=lower_bound, upper_bound=upper_bound)

        # for row in environment:
        #     print(f"\nclass={inter_values['labels'][int(row[2])]}   x={float(row[0]):3f}  y={float(row[1]):3f}")

        detections = []
        if show:
            for i in range(len(scores)):
                if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    cv2.rectangle(window_image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    object_name = inter_values['labels'][int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(window_image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(window_image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                    detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])                

            cv2.imshow(f'{full_title} Proccessed', window_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

window(model_path = 'Model\\test_model_001\\detect.tflite', lblpath = 'Model\\test_model_001\\labelmap.txt')
