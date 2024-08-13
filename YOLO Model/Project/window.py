import time
import dxcam, cv2, win32gui
from ultralytics import YOLO
import numpy as np

def path_detection(boxes, rgb, lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    masked_cleared = (mask > 0).astype(np.uint8)
    window_image = np.stack([masked_cleared, masked_cleared, masked_cleared], axis=-1) * 255   
    for box in boxes:
        cv2.rectangle(img=window_image, **box)
    height, width = window_image.shape[:2]
    window_image = cv2.floodFill(window_image, None, (int(width/2), int(height/2+30)), (256, 0, 0))[1]
    window_image = cv2.circle(window_image, (int(width/2), int(height/2+30)), 1, (0, 0, 256))
    return window_image

def window_dxcam(model_path: str, draw: bool = True, imgsz: int = 480, show_result:bool = True, path: bool = True, model_detect: bool = True, scale_order: list = [4], min_conf: float = 0.4, window_title = 'Mazevil', lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    lower_bound = np.array([24,20,37]) # 100, 50, 50
    upper_bound = np.array([24,20,37]) # 255, 150, 150
    
    model = YOLO(model_path)

    hwnd = win32gui.FindWindow(None, window_title)
    cam = dxcam.create(output_color="BGR")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)

    fps_list = []
    prevTime = 0
    fps = 0
    
    while True:
        
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        top_offset, bottom_offset = 50, 58
        window_image = cam.grab(region=(left+8, top+top_offset, right-8, bottom-bottom_offset))

        if window_image is not None:

            currTime = time.perf_counter()
            fps = 1 / (currTime - prevTime)
            fps_list.append(fps)
            prevTime = currTime

            if model_detect: 
                results = model(classes=[0,1,2,3,4,5,7,8,9,10,11,12,13,14],device=0,source=window_image, conf=min_conf, imgsz=imgsz, stream=True, verbose=False)
                if path: 
                    boxes = []
                    for result in results:
                        # Calculate the top-left corner of the bounding box
                        for box in result.boxes:
                            # Extract bounding box coordinates and other information
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = box.conf
                            if confidence > min_conf:
                                class_id = box.cls
                                if int(class_id) == 2: # trap off
                                    color = (0, 0, 0)
                                elif int(class_id) == 3: # trap on
                                    color = (0, 0, 256)
                                else: continue
                                boxes.append({"pt1": (x1-2, y1),"pt2": (x2+2, y2+2),"color": color,"thickness": -1})
                    window_image = path_detection(boxes = boxes, rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), lower_bound=lower_bound, upper_bound=upper_bound)
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
    print(sum(fps_list)/sum(fps_list))

model_name = 'test_1'

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
