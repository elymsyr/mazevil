window_title = 'Mazevil'
model_path = 'Model\\custom_model_lite\\detect.tflite'
lblpath = 'Model\\custom_model_lite\\labelmap.txt'
min_conf = 0.3

import pygetwindow as gw
from PIL import ImageGrab
import numpy as np
import cv2
import tensorflow as tf
from tflite_detect_image import tflite_detection

with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

def preprocess_image(image):
    input_shape = input_details[0]['shape']
    resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))
    normalized_image = resized_image / 255.0
    input_data = np.expand_dims(normalized_image.astype(np.float32), axis=0)
    return input_data, resized_image

def capture_window(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    bbox = (window.left, window.top, window.right, window.bottom)
    img = ImageGrab.grab(bbox)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
while True:
    window_image = capture_window(window_title)
    # input_data, processed_image = preprocess_image(window_image)
    imH, imW, _ = window_image.shape
    boxes, classes, scores = tflite_detection(window_image, labels, interpreter, input_details, output_details, float_input, width, height)
    
    detections = []
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(window_image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(window_image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(window_image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
    
    cv2.imshow("Processed Input Data", window_image)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
