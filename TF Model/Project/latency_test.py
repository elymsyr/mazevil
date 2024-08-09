import time
import dxcam, cv2, win32gui
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
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
    
    upscaled_array = np.where(upscaled_array[..., None] == 0, [0, 0, 0], [1, 1, 1])
    
    return upscaled_array

def capture_window(window: gw.Win32Window, top_crop=0, bottom_crop=1):
    avg = 8
    extra_top_avg = 22
    bbox = (window.left+avg, window.top+avg+extra_top_avg, window.right-avg, window.bottom-avg)
    img = ImageGrab.grab(bbox)
    matlike_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return matlike_image[top_crop:-bottom_crop, :]

def window(model_name : str, model_path: str, lblpath: str, map_color: dict, time_passed:int = 200, show_result:bool = True, path: bool = True, model_detect: bool = True, scale_order: list = [8,3], min_conf: float = 0.4, window_title = 'Mazevil', full_title = 'Mazevil', lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    lower_bound = np.array([100, 50, 50])
    upper_bound = np.array([255, 150, 150])
    inter_values = model(model_path = model_path, lblpath = lblpath)
    
    windows = gw.getWindowsWithTitle(window_title)
    window = windows[0]
    window_image = capture_window(window, top_crop=20, bottom_crop=50)
    imH, imW, _ = window_image.shape
    
    print(imH, imW)
    
    capture_time = []
    detection_time = []
    path_time = []
    show_time = []
    
    fps_list = []
    prevTime = 0
    fps = 0
    while True:
        currTime = time.perf_counter()
        fps = 1 / (currTime - prevTime)
        fps_list.append(fps)
        prevTime = currTime        
        
        t1 = time.perf_counter()
        window_image = capture_window(window, top_crop=20, bottom_crop=50)
        t2 = time.perf_counter()
        capture_time.append(t2-t1)
        t2 = time.perf_counter()
        if model_detect: boxes, classes, scores, _ = model_detection(image=window_image, inter_values=inter_values, min_conf=min_conf)
        t3 = time.perf_counter()
        detection_time.append(t3-t2)
        t3 = time.perf_counter()
        
        if model_detect:
            for i in range(len(scores)):
                if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    object_name = inter_values['labels'][int(classes[i])]
                    if object_name in map_color.keys():
                        expand_by = 20 if object_name == 'trap_off' else 5
                        cv2.rectangle(window_image, (xmin - expand_by, ymin - expand_by), (xmax + expand_by, ymax + expand_by), map_color[object_name], -1)

        if path: binary_array = path_detection(rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), downscale_order=scale_order, lower_bound=lower_bound, upper_bound=upper_bound)
        t4 = time.perf_counter()
        path_time.append(t4-t3)
        t4 = time.perf_counter()

        if show_result:
            if path:
                for scale in scale_order:
                    binary_array = upscale_binary_array(binary_array, scale)
                map = (binary_array * 255).astype(np.uint8)
            else: map = window_image
            cv2.imshow(f'{full_title} Proccessed', map)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        t5 = time.perf_counter()
        show_time.append(t5-t4)
        if len(show_time) > time_passed : break
    cv2.destroyAllWindows()
    
    return f"GW:\n  Times Passed: {time_passed}\n  Model: {model_name if model_detect else '__no_detection__'}\n  Avg Fps: {sum(fps_list)/len(fps_list)}\n  Avg counter times:\n    Capture time: {(sum(capture_time)/len(capture_time)):.4f}, {max(capture_time)}, {min(capture_time)}\n    Detection time: {(sum(detection_time)/len(detection_time)) if model_detect else '__no_detection__'}, {max(detection_time)}, {min(detection_time)}\n    Path time: {(sum(path_time)/len(path_time)) if path else '__no_path__'}, {max(path_time)}, {min(path_time)}\n    Show time: {(sum(show_time)/len(show_time)) if show_result else '__no_show__'}, , {max(show_time)}, {min(show_time)}\n\n"

def window_dxcam(model_name: str, model_path: str, lblpath: str, map_color: dict, time_passed: int = 200, show_result:bool = True, path: bool = True, model_detect: bool = True, scale_order: list = [9, 3], min_conf: float = 0.4, window_title = 'Mazevil', full_title = 'Mazevil', lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    lower_bound = np.array([100, 50, 50])
    upper_bound = np.array([255, 150, 150])
    inter_values = model(model_path = model_path, lblpath = lblpath)

    window_title = 'Mazevil'
    hwnd = win32gui.FindWindow(None, window_title)
    cam = dxcam.create(output_color="BGR")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    window_image = cam.grab(region=(left+8, top+30+20, right-8, bottom-8-50))
    imH, imW, _ = window_image.shape

    print(imH, imW)

    capture_time = []
    detection_time = []
    path_time = []
    show_time = []
    fps_list = []
    prevTime = 0
    fps = 0
    while True:
        
        t1 = time.perf_counter()
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        window_image = cam.grab(region=(left+8, top+30+20, right-8, bottom-8-50))
        t2 = time.perf_counter()

        if window_image is not None:
            capture_time.append(t2-t1)

            currTime = time.perf_counter()
            fps = 1 / (currTime - prevTime)
            fps_list.append(fps)
            prevTime = currTime

            t2 = time.perf_counter()
            if model_detect: boxes, classes, scores, _ = model_detection(image=window_image, inter_values=inter_values, min_conf=min_conf)
            t3 = time.perf_counter()
            detection_time.append(t3-t2)
            t3 = time.perf_counter()

            if model_detect:
                for i in range(len(scores)):
                    if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))
                        object_name = inter_values['labels'][int(classes[i])]
                        if object_name in map_color.keys():
                            expand_by = 20 if object_name == 'trap_off' else 5
                            cv2.rectangle(window_image, (xmin - expand_by, ymin - expand_by), (xmax + expand_by, ymax + expand_by), map_color[object_name], -1)

            if path: binary_array = path_detection(rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), downscale_order=scale_order, lower_bound=lower_bound, upper_bound=upper_bound)

            t4 = time.perf_counter()
            path_time.append(t4-t3)
            
            t4 = time.perf_counter()

            if show_result:
                if path:
                    for scale in scale_order:
                        binary_array = upscale_binary_array(binary_array, scale)
                    map = (binary_array * 255).astype(np.uint8)
                else: map = window_image
                cv2.imshow(f'{full_title} Proccessed', map)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            t5 = time.perf_counter()
            show_time.append(t5-t4)
            if len(show_time) > time_passed : break
    cv2.destroyAllWindows()
    del cam

    return f"DXCAM:\n  Times Passed: {time_passed}\n  Model: {model_name if model_detect else '__no_detection__'}\n  Avg Fps: {sum(fps_list)/len(fps_list)}\n  Avg counter times:\n    Capture time: {(sum(capture_time)/len(capture_time)):.4f}, {max(capture_time)}, {min(capture_time)}\n    Detection time: {(sum(detection_time)/len(detection_time)) if model_detect else '__no_detection__'}, {max(detection_time)}, {min(detection_time)}\n    Path time: {(sum(path_time)/len(path_time)) if path else '__no_path__'}, {max(path_time)}, {min(path_time)}\n    Show time: {(sum(show_time)/len(show_time)) if show_result else '__no_show__'}, , {max(show_time)}, {min(show_time)}\n\n"

model_name = 'test_1'

model_path = f'TF Model\\Model\\{model_name}\\detect.tflite'
lblpath = f'TF Model\\Model\\{model_name}\\labelmap.txt'

map_color = {
    'trap_off': (184,111,80)
}

conf = {
    'time_passed': 300,
    'show_result': True,
    'path': False,
    'model_detect': True,
    'model_name': model_name,
    'model_path': model_path,
    'lblpath': lblpath,
    'scale_order': [],
    'map_color': map_color
}

print('GW window')
text1 = window(**conf)
print('DXCAM window_dxcam')
text2 = window_dxcam(**conf)
note = 'scale_order: []'

# with open('TF Model\\Project\\latency_test_results.txt', 'a') as file:
#     if len(note)>3 : file.write(f"Notes: {note}")
#     file.write(text1)
#     file.write(text2)