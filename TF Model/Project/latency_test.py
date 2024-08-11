import time
import dxcam, cv2, win32gui
import numpy as np
from model import model, model_detection
from scipy.ndimage import convolve
from path import path_detection


def window_dxcam(model_name: str, model_path: str, lblpath: str, map_color: dict, draw: bool = True, time_passed: int = 200, show_result:bool = True, path: bool = True, model_detect: bool = True, scale_order: list = [4], min_conf: float = 0.4, window_title = 'Mazevil', full_title = 'Mazevil', lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    lower_bound = np.array([24,20,37]) # 100, 50, 50
    upper_bound = np.array([24,20,37]) # 255, 150, 150
    inter_values = model(model_path = model_path, lblpath = lblpath)

    window_title = 'Mazevil'
    hwnd = win32gui.FindWindow(None, window_title)
    cam = dxcam.create(output_color="BGR")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    window_image = cam.grab(region=(left+8, top+30+20, right-8, bottom-8-50))
    imH, imW, _ = window_image.shape

    kernel = np.array([[1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]])

    fps_list = []
    prevTime = 0
    fps = 0
    
    t1, t2, t3 = 0,0,0
    path_list = []
    draw_list = []
    
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
                    t1 = time.perf_counter()
                    window_image, masked = path_detection(rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), lower_bound=lower_bound, upper_bound=upper_bound)

                    # Count the number of neighbors that are 1 for each pixel
                    neighbor_count = convolve(masked, kernel, mode='constant', cval=0)
                    # Apply the condition: Set pixel to 0 if it has less than 3 neighbors that are 1
                    masked_cleared = np.where(neighbor_count > 4, masked, 0)
                    
                    for scale in scale_order:
                        # Count the number of neighbors that are 1 for each pixel
                        neighbor_count = convolve(masked_cleared, kernel, mode='constant', cval=0)
                        # Apply the condition: Set pixel to 0 if it has less than 3 neighbors that are 1
                        masked_cleared = np.where(neighbor_count > scale, masked_cleared, 0)                    
                        
                    window_image = np.stack([masked_cleared, masked_cleared, masked_cleared], axis=-1) * 255   
                    t2 = time.perf_counter()

                if draw:
                    t2 = time.perf_counter()
                    for i in range(len(scores)):
                        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                            ymin = int(max(1,(boxes[i][0] * imH)))
                            xmin = int(max(1,(boxes[i][1] * imW)))
                            ymax = int(min(imH,(boxes[i][2] * imH)))
                            xmax = int(min(imW,(boxes[i][3] * imW)))
                            object_name = inter_values['labels'][int(classes[i])]

                            if object_name in map_color.keys():
                                expand_by = 2
                                cv2.rectangle(window_image, (xmin - (int(expand_by * 1.5) if object_name == 'trap_off' or object_name == 'trap_on' else expand_by), ymin - expand_by), (xmax + (int(expand_by * 1.5) if object_name == 'trap_off' or object_name == 'trap_on' else expand_by), ymax + expand_by), map_color[object_name], -1 if path else 2)
                                if not path:
                                    text_position = (xmin - expand_by, ymax + int(1.5 * expand_by))

                                    # Write the object's name below the rectangle
                                    cv2.putText(
                                        window_image, 
                                        object_name, 
                                        text_position, 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5,  # Font scale (adjust size as needed)
                                        map_color[object_name], 
                                        1,  # Thickness
                                        cv2.LINE_AA  # Anti-aliased text for better quality
                                    )
                    t3 = time.perf_counter()
            
            if path: path_list.append(t2-t1)
            if draw :draw_list.append(t3-t2)
            
            if show_result:            
                cv2.putText(window_image, f"FPS: {int(fps) if abs(fps_list[-1] - fps) > 1 else int(fps_list[-1])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f'{full_title} Proccessed', window_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if len(fps_list) > time_passed: break
    
    cv2.destroyAllWindows()
    del cam
    return path_list if path else None, draw_list if draw else None, fps_list

model_name = 'test_1'

model_path = f'TF Model\\Model\\{model_name}\\detect.tflite'
lblpath = f'TF Model\\Model\\{model_name}\\labelmap.txt'

map_color = {
    'enemy_slug': (255, 0, 0),         # Red
    'enemy_slug_big': (0, 255, 0),     # Green
    'trap_off': (255, 165, 0),         # Orange
    'trap_on': (0, 0, 255),            # Blue
    'enemy_skeleton': (255, 0, 255),   # Magenta
    'enemy_skeleton_shoot': (0, 255, 255), # Cyan
    'door': (128, 128, 0),             # Olive
    'trap_door': (0, 128, 128),        # Teal
    'gold': (255, 255, 0),             # Yellow
    'treasury_open': (75, 0, 130),     # Indigo
    'treasury_close': (255, 20, 147),  # Deep Pink
    'treasury_monster': (64, 224, 208),# Turquoise
    'key': (0, 128, 0),                # Dark Green
    'enemy_slug_boss': (128, 0, 128)   # Purple
}


conf = {
    'time_passed': 500,
    'show_result': False,
    'path': False,
    'draw': False,
    'model_detect': False,
    'model_name': model_name,
    'model_path': model_path,
    'lblpath': lblpath,
    'scale_order': [],
    'map_color': map_color
}

path, draw,fps = window_dxcam(**conf)
print(f"fps: {(sum(fps)/len(fps) if len(fps) > 0 else 0.0):.3f} max({(max(fps)):.4f}) min({(min(fps)):.3f})")
if path: print(f"path: {(sum(path)/len(path) if len(path) > 0 else 0.0):.3f} max({(max(path)):.4f}) min({(min(path)):.3f})")
if draw: print(f"draw: {(sum(draw)/len(draw) if len(draw) > 0 else 0.0):.3f} max({(max(draw)):.4f}) min({(min(draw)):.3f})")