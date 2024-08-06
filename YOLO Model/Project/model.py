import numpy as np


# def model(model_path: str, lblpath: str):
#     with open(lblpath, 'r') as f:
#         labels = [line.strip() for line in f.readlines()]    
    
#     interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.dll')])
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     height = input_details[0]['shape'][1]
#     width = input_details[0]['shape'][2]
    
#     float_input = (input_details[0]['dtype'] == np.float32)

#     return {'labels': labels, 'interpreter': interpreter, 'input_details': input_details, 'output_details': output_details, 'height': height, 'width': width, 'float_input': float_input}

# def find_center_coordinates(boxes, scores, classes, min_conf):
#     centers = []
#     for index, box in enumerate(boxes):
#         ymin, xmin, ymax, xmax = box
#         center_x = (xmin + xmax) / 2
#         center_y = (ymin + ymax) / 2
#         if scores[index] > min_conf: centers.append([center_x, center_y, classes[index ]])
#     centers_array = np.array(centers)  # First, create a numpy array
#     return np.ndarray(centers_array.shape, buffer=centers_array.data)

def model_detection(image, inter_values: dict, min_conf: float):
    environment: np.ndarray
    
    return environment
