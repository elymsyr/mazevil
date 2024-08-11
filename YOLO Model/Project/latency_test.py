import time
import dxcam, cv2, win32gui
from ultralytics import YOLO
import numpy as np
from scipy.ndimage import convolve

def path_detection(rgb, lower_bound = np.array([24,20,37]), upper_bound = np.array([24,20,37])):
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    mask_white = (mask > 0).astype(np.uint8)
    return np.stack([mask_white, mask_white, mask_white], axis=-1) * 255 , (mask > 0).astype(np.uint8)

def window_dxcam(model_path: str, draw: bool = True, imgsz: int = 480, show_result:bool = True, path: bool = True, model_detect: bool = True, scale_order: list = [4], min_conf: float = 0.4, window_title = 'Mazevil', lower_bound = np.array([24,20,37]), upper_bound = np.array([24,20,37])):
    lower_bound = np.array([24,20,37]) # 100, 50, 50
    upper_bound = np.array([24,20,37]) # 255, 150, 150
    
    model = YOLO(model_path)

    hwnd = win32gui.FindWindow(None, window_title)
    cam = dxcam.create(output_color="BGR")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)

    kernel = np.array([[1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]])

    fps_list = []
    prevTime = 0
    fps = 0
    path_list = []
    draw_list = []
    detection_list = []
    
    while True:
        
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        window_image = cam.grab(region=(left+8, top+30+20, right-8, bottom-8-50))

        if window_image is not None:

            currTime = time.perf_counter()
            fps = 1 / (currTime - prevTime)
            fps_list.append(fps)
            prevTime = currTime
        

            if model_detect:
                t1 = time.perf_counter()
                results = model(source=window_image, conf=min_conf, imgsz=imgsz, stream=True, verbose=False)
                t2 = time.perf_counter()
                detection_list.append(t2-t1)   

            if path:
                t1 = time.perf_counter()
                window_image, masked = path_detection(rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), lower_bound=lower_bound, upper_bound=upper_bound)

                neighbor_count = convolve(masked, kernel, mode='constant', cval=0)
                masked_cleared = np.where(neighbor_count > 4, masked, 0)
                
                for scale in scale_order:
                    neighbor_count = convolve(masked_cleared, kernel, mode='constant', cval=0)
                    masked_cleared = np.where(neighbor_count > scale, masked_cleared, 0)                    
                    
                window_image = np.stack([masked_cleared, masked_cleared, masked_cleared], axis=-1) * 255   
                t2 = time.perf_counter()
                path_list.append(t2-t1)

                if draw:
                    t1 = time.perf_counter()
                    for result in results:
                        window_image = result.plot(img = window_image)
                    t2 = time.perf_counter()
                    draw_list.append(t2-t1)
                        
            
            if show_result:
                cv2.putText(window_image, f"FPS: {int(fps) if abs(fps_list[-1] - fps) > 1 else int(fps_list[-1])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f'{window_title} YOLOV8', window_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
            if len(fps_list) > 300 : break
    
    cv2.destroyAllWindows()
    del cam
    return path_list if path_list else None , draw_list if draw_list else None , detection_list if detection_list else None , fps_list if fps_list else None 

model_name = 'test_0'

model_path = f'YOLO Model\\Model\\Trained Models\\{model_name}\\train\\weights\\best.pt'

conf1 = {
    'model_path': model_path,
    'imgsz': 480,
    'draw' : False,
    'show_result' : False,
    'path' : False,
    'model_detect': False,
    'scale_order' : [4,4]
}

conf2 = {
    'model_path': model_path,
    'imgsz': 480,
    'draw' : False,
    'show_result' : True,
    'path' : False,
    'model_detect': False,
    'scale_order' : [4,4]
}

conf3 = {
    'model_path': model_path,
    'imgsz': 480,
    'draw' : False,
    'show_result' : False,
    'path' : False,
    'model_detect': True,
    'scale_order' : [4,4]
}

conf4 = {
    'model_path': model_path,
    'imgsz': 480,
    'draw' : True,
    'show_result' : False,
    'path' : False,
    'model_detect': True,
    'scale_order' : [4,4]
}

conf5 = {
    'model_path': model_path,
    'imgsz': 480,
    'draw' : True,
    'show_result' : False,
    'path' : True,
    'model_detect': True,
    'scale_order' : [4,4]
}

conf6 = {
    'model_path': model_path,
    'imgsz': 480,
    'draw' : True,
    'show_result' : False,
    'path' : True,
    'model_detect': True,
    'scale_order' : []
}

conf7 = {
    'model_path': model_path,
    'imgsz': 480,
    'draw' : True,
    'show_result' : True,
    'path' : True,
    'model_detect': True,
    'scale_order' : []
}

conf8 = {
    'model_path': model_path,
    'imgsz': 480,
    'draw' : True,
    'show_result' : True,
    'path' : True,
    'model_detect': True,
    'scale_order' : [4,4]
}

def try_conf(conf):

    path, draw, detect, fps = window_dxcam(**conf)

    t0 = f"fps: {(sum(fps)/len(fps) if len(fps) > 0 else 0.0):.3f} max({(max(fps)):.4f}) min({(min(fps)):.3f})"
    if path: t1 = f"path: {(sum(path)/len(path) if len(path) > 0 else 0.0):.3f} max({(max(path)):.4f}) min({(min(path)):.3f})"
    if detect: t2 = f"detect: {(sum(detect)/len(detect) if len(detect) > 0 else 0.0):.3f} max({(max(detect)):.4f}) min({(min(detect)):.3f})"
    if draw: t3 = f"draw: {(sum(draw)/len(draw) if len(draw) > 0 else 0.0):.3f} max({(max(draw)):.4f}) min({(min(draw)):.3f})"

    with open('YOLO Model\\Project\\latency_test.txt', 'a') as file:
        file.writelines(f"  -- {model_name.upper()} --  ")
        file.writelines('\n')
        for key, value in conf.items():
            if key != 'model_path': file.writelines(f"{key}: {value}\n")
        file.writelines(f"  {t0}")
        if path: file.writelines(f"  {t1}")
        if detect: file.writelines(f"  {t2}")
        if draw: file.writelines(f"  {t3}")
        file.writelines('\n\n')
        
    print('DONE')
    
for conf in [conf1, conf2, conf3, conf4, conf5, conf6, conf7, conf8]:
    try_conf(conf=conf)