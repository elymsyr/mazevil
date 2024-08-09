import time
import dxcam, cv2, win32gui
import numpy as np
from model import model, model_detection
from path import path_detection


def window_dxcam(model_name: str, model_path: str, lblpath: str, map_color: dict, time_passed: int = 200, show_result:bool = True, path: bool = True, model_detect: bool = True, scale_order: list = [9, 3], min_conf: float = 0.4, window_title = 'Mazevil', full_title = 'Mazevil', lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    lower_bound = np.array([63, 40, 50]) # 100, 50, 50
    upper_bound = np.array([228, 166, 114]) # 255, 150, 150
    inter_values = model(model_path = model_path, lblpath = lblpath)

    window_title = 'Mazevil'
    hwnd = win32gui.FindWindow(None, window_title)
    cam = dxcam.create(output_color="BGR")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    window_image = cam.grab(region=(left+8, top+30+20, right-8, bottom-8-50))
    imH, imW, _ = window_image.shape

    capture_time = []
    detection_time = []
    path_time = []
    show_time = []
    fps_list = []
    prevTime = 0
    fps = 0
    
    while True:
        
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        window_image = cam.grab(region=(left+8, top+30+20, right-8, bottom-8-50))

        if window_image is not None:

            currTime = time.perf_counter()
            fps = 1 / (currTime - prevTime)
            fps_list.append(fps)
            prevTime = currTime

            if model_detect: 
                boxes, classes, scores, _ = model_detection(image=window_image, inter_values=inter_values, min_conf=min_conf)
                if path: 
                    binary_array = path_detection(rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), downscale_order=scale_order, lower_bound=lower_bound, upper_bound=upper_bound)
                    window_image = (binary_array * 255).astype(np.uint8)
                for i in range(len(scores)):
                    if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))
                        object_name = inter_values['labels'][int(classes[i])]

                        if object_name in map_color.keys():
                            expand_by = 2
                            cv2.rectangle(window_image, (xmin - (int(expand_by * 1.5) if object_name == 'trap_off' else expand_by), ymin - expand_by), (xmax + (int(expand_by * 1.5) if object_name == 'trap_off' else expand_by), ymax + expand_by), map_color[object_name], -1)

            if show_result:
                cv2.imshow(f'{full_title} Proccessed', window_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cv2.destroyAllWindows()
    del cam
    return f"DXCAM:\n  Times Passed: {time_passed}\n  Model: {model_name if model_detect else '__no_detection__'}\n  Avg Fps: {sum(fps_list)/len(fps_list)}\n  Avg counter times:\n    Capture time: {(sum(capture_time)/len(capture_time)):.4f}, {max(capture_time)}, {min(capture_time)}\n    Detection time: {(sum(detection_time)/len(detection_time)) if model_detect else '__no_detection__'}, {max(detection_time)}, {min(detection_time)}\n    Path time: {(sum(path_time)/len(path_time)) if path else '__no_path__'}, {max(path_time)}, {min(path_time)}\n    Show time: {(sum(show_time)/len(show_time)) if show_result else '__no_show__'}, , {max(show_time)}, {min(show_time)}\n\n"

model_name = 'test_1'

model_path = f'TF Model\\Model\\{model_name}\\detect.tflite'
lblpath = f'TF Model\\Model\\{model_name}\\labelmap.txt'

map_color = {
    'trap_off': (80,111,184)
}

conf = {
    'time_passed': 300,
    'show_result': True,
    'path': True,
    'model_detect': True,
    'model_name': model_name,
    'model_path': model_path,
    'lblpath': lblpath,
    'scale_order': [],
    'map_color': map_color
}

text = window_dxcam(**conf)
print(text)