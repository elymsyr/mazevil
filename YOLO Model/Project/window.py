import time
import dxcam, cv2, win32gui
from ultralytics import YOLO
import numpy as np
from scipy.ndimage import convolve

def path_detection(rgb, lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    mask_white = (mask > 0).astype(np.uint8)
    return np.stack([mask_white, mask_white, mask_white], axis=-1) * 255 , (mask > 0).astype(np.uint8)

def window_dxcam(model_path: str, draw: bool = True, imgsz: int = 480, show_result:bool = True, path: bool = True, model_detect: bool = True, scale_order: list = [4], min_conf: float = 0.4, window_title = 'Mazevil', lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
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
    
    while True:
        
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        window_image = cam.grab(region=(left+8, top+30+20, right-8, bottom-8-50))

        if window_image is not None:

            currTime = time.perf_counter()
            fps = 1 / (currTime - prevTime)
            fps_list.append(fps)
            prevTime = currTime

            if model_detect: 
                results = model(source=window_image, conf=min_conf, imgsz=imgsz, stream=True, verbose=False)
                
                if path: 
                    window_image, masked = path_detection(rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), lower_bound=lower_bound, upper_bound=upper_bound)

                    neighbor_count = convolve(masked, kernel, mode='constant', cval=0)
                    masked_cleared = np.where(neighbor_count > 4, masked, 0)
                    
                    for scale in scale_order:
                        neighbor_count = convolve(masked_cleared, kernel, mode='constant', cval=0)
                        masked_cleared = np.where(neighbor_count > scale, masked_cleared, 0)                    
                        
                    window_image = np.stack([masked_cleared, masked_cleared, masked_cleared], axis=-1) * 255   


                if draw:
                    for result in results:
                        window_image = result.plot(img = window_image)
            
            if show_result:            
                cv2.putText(window_image, f"FPS: {int(fps) if abs(fps_list[-1] - fps) > 1 else int(fps_list[-1])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f'{window_title} YOLOV8', window_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cv2.destroyAllWindows()
    del cam

model_name = 'test_0'

model_path = f'YOLO Model\\Model\\Trained Models\\{model_name}\\train\\weights\\best.pt'

conf = {
    'model_path': model_path,
    'draw' : True,
    'imgsz': 480,
    'show_result' : True,
    'path' : True,
    'model_detect': True,
    'scale_order' : [4,4],
    'min_conf' : 0.4,
    'window_title' : 'Mazevil',
    'lower_bound' : np.array([24,20,37]),
    'upper_bound' : np.array([24,20,37])
}

window_dxcam(**conf)
