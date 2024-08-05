from tensorflow.lite.python.interpreter import Interpreter
import cv2
import tensorflow as tf
import numpy as np


def model(model_path: str, lblpath: str):
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]    
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    float_input = (input_details[0]['dtype'] == np.float32)

    return {'labels': labels, 'interpreter': interpreter, 'input_details': input_details, 'output_details': output_details, 'height': height, 'width': width, 'float_input': float_input}


def model_detection(image, inter_values: dict, input_mean = 127.5, input_std = 127.5):
    interpreter = inter_values['interpreter']
    
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