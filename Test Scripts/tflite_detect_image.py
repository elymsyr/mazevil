# Script to run custom TFLite model on test images to detect objects
# Source: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_image.py

# Import packages
import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt


# # Set up variables for running user's model
# PATH_TO_IMAGES='/content/images/test'   # Path to test images folder
# PATH_TO_MODEL='/content/custom_model_lite/detect.tflite'   # Path to .tflite model file
# PATH_TO_LABELS='/content/labelmap.txt'   # Path to labelmap.txt file
# min_conf_threshold=0.5   # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
# images_to_test = 10   # Number of images to run detection on

# # Run inferencing function!
# tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)

def list_png_files_with_full_path(folder_path):
    """Finds all PNG files in the specified folder and returns a list of their full paths."""
    png_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    return png_files

### Define function for inferencing with TFLite model and displaying results

def tflite_detection(image, interpreter: Interpreter, inter_values: dict, input_mean = 127.5, input_std = 127.5):
    # Get input and output tensors
    output_details = inter_values['output_details']
   
    # imH, imW, _ = image.shape
    image_resized = cv2.resize(image, (inter_values['width'], inter_values['height']))
    input_data = np.expand_dims(image_resized, axis=0)    

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if inter_values['float_input']:
        input_data = (np.float32(input_data) - input_mean) / input_std    

    interpreter.set_tensor(inter_values['input_details'][0]['index'],input_data)
    interpreter.invoke()
        
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    
    return boxes, classes, scores


def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.5, num_test_images=10, savepath='/content/results', txt_only=False):

    # Grab filenames of all images in test folder
    images = list_png_files_with_full_path(imgpath)

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Randomly select test images
    images_to_test = random.sample(images, num_test_images)

    # Loop over every image and perform detection
    for image_path in images_to_test:

        # Load image and resize to expected shape [1xHxWx3]
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

        detections = []

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])


        # All the results have been drawn on the image, now display the image
        if txt_only == False: # "text_only" controls whether we want to display the image results or just save them in .txt files
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12,16))
            plt.imshow(image)
            plt.show()

        # Save detection results in .txt files (for calculating mAP)
        elif txt_only == True:

            # Get filenames and paths
            image_fn = os.path.basename(image_path)
            base_fn, ext = os.path.splitext(image_fn)
            txt_result_fn = base_fn +'.txt'
            txt_savepath = os.path.join(savepath, txt_result_fn)

            # Write results to text file
            # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
            with open(txt_savepath,'w') as f:
                for detection in detections:
                    f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

    return