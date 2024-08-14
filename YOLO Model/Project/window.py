import time
import dxcam, cv2, win32gui, win32api, win32con
from ultralytics import YOLO
import numpy as np

CLICKED = False

def find_optimal_direction(player_pos, object_positions, directions):
    # Ensure that object_weights has the same length as object_positions
    object_weights = [x[0] for x in object_positions]
    object_positions = [x[1] for x in object_positions]
    
    # Compute the weighted optimal movement vector
    optimal_vector = np.zeros(2, dtype=np.float64)
    
    for obj, weight in zip(object_positions, object_weights):
        obj_vector = np.array(obj) - np.array(player_pos)
        norm = np.linalg.norm(obj_vector)
        if norm > 0:
            obj_vector = (obj_vector/norm)
        optimal_vector = optimal_vector - weight * obj_vector
        
    # Normalize the optimal vector
    if np.linalg.norm(optimal_vector) > 0:
        optimal_vector = (optimal_vector / np.linalg.norm(optimal_vector))
    
    # Find the closest direction to the optimal vector
    best_direction = directions[np.argmin(np.linalg.norm(directions - optimal_vector, axis=1))]

    return best_direction

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def path_detection(center, boxes, rgb, lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    masked_cleared = (mask > 0).astype(np.uint8)
    window_image = np.stack([masked_cleared, masked_cleared, masked_cleared], axis=-1) * 255   
    for box in boxes:
        cv2.rectangle(img=window_image, **box)
    window_image = cv2.floodFill(window_image, None, center, (250, 0, 0))[1]
    window_image = cv2.circle(window_image, center, 1, (0, 0, 256))
    window_image = cv2.inRange(window_image, np.array([250, 0, 0]), np.array([250, 0, 0]))
    return window_image

def enemy_found(enemies, window_x, window_y):
    global CLICKED
    choosen_enemy = enemies[0][1]
    screen_x = choosen_enemy[0]+window_x+8
    screen_y = choosen_enemy[1]+window_y+54
    win32api.SetCursorPos((screen_x, screen_y))
    if not CLICKED: win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
    CLICKED = True

def find_available_directions(center, map, directions):
    for direction in directions:
        check_point = (int(center[0] + direction[0] * 15), int(center[1] + direction[1] * 15))
        if map(check_point) != (256,256,256):
            print(direction)
    

def window_dxcam(model_path: str, draw: bool = True, imgsz: int = 480, show_result:bool = True, path: bool = True, model_detect: bool = True, scale_order: list = [4], min_conf: float = 0.4, window_title = 'Mazevil', lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    lower_bound = np.array([24,20,37]) # 100, 50, 50
    upper_bound = np.array([24,20,37]) # 255, 150, 150
    
    model = YOLO(model_path)

    hwnd = win32gui.FindWindow(None, window_title)
    window_rect = win32gui.GetWindowRect(hwnd)
    window_x = window_rect[0]
    window_y = window_rect[1]
    max_distance = 500
    cam = dxcam.create(output_color="BGR")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    top_offset, bottom_offset = 50, 58
    window_image = cam.grab(region=(left+8, top+top_offset, right-8, bottom-bottom_offset))
    print(window_image)
    height, width = window_image.shape[:2]
    center = (int(width/2), int(height/2+30))
    fps_list = []
    prevTime = 0
    fps = 0
    
    directions = np.array([
        [0, 1],   # Up
        [1, 1],   # Up-Right
        [1, 0],   # Right
        [1, -1],  # Down-Right
        [0, -1],  # Down
        [-1, -1], # Down-Left
        [-1, 0],  # Left
        [-1, 1]   # Up-Left
    ], dtype=np.float64)    
    
    while True:
        global CLICKED
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        top_offset, bottom_offset = 50, 58
        window_image = cam.grab(region=(left+8, top+top_offset, right-8, bottom-bottom_offset))

        if window_image is not None:

            currTime = time.perf_counter()
            fps = 1 / (currTime - prevTime)
            fps_list.append(fps)
            prevTime = currTime

            if model_detect: 
                results = model(classes=[0,1,2,3,4,5,7,9,10,11,12,13,14],device=0,source=window_image, conf=min_conf, imgsz=imgsz, stream=True, verbose=False)
                if path:
                    enemy = [] 
                    boxes = []
                    rewards = []
                    for result in results:
                        # Calculate the top-left corner of the bounding box
                        for box in result.boxes:
                            # Extract bounding box coordinates and other information
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            x, y, *_ = map(int, box.xywh[0])
                            confidence = box.conf
                            if confidence > min_conf:
                                class_id = box.cls
                                if int(class_id) == 2: # trap off
                                    color = (0, 0, 0)
                                    boxes.append({"pt1": (x1-2, y1),"pt2": (x2+2, y2+2),"color": color,"thickness": -1})
                                elif int(class_id) == 3: # trap on
                                    color = (0, 0, 256)
                                    boxes.append({"pt1": (x1-2, y1),"pt2": (x2+2, y2+2),"color": color,"thickness": -1})
                                elif int(class_id) in [0,1,4,5,12,14]:
                                    enemy.append([1.5 if int(class_id) == 5 else 1,(x,y)])
                                elif int(class_id) in [7,9,10,11,13]:
                                    rewards.append((x,y, int(class_id)))
                                else: continue
                    window_image = path_detection(center=center, boxes = boxes, rgb=cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), lower_bound=lower_bound, upper_bound=upper_bound)
                    enemies = sorted(
                        [point for point in enemy if euclidean_distance(point[1], center) <= max_distance],
                        key=lambda point: euclidean_distance(point[1], center)
                    )
                    direction = find_optimal_direction(center, directions=directions)
                    if enemies:
                        for enemy in enemies:
                            window_image = cv2.circle(window_image, enemy[1], 3, (0, 0, 256))
                        
                        enemy_found(enemies=enemies, window_x=window_x, window_y=window_y)
                        
                        
                        pt1 = (int(center[0]), int(center[1]))  
                        pt2 = (int(center[0] + direction[0] * 20), int(center[1] + direction[1] * 20))  # End point
                    
                        window_image = cv2.line(window_image, pt1, pt2, [128,128,256],2)
                    elif CLICKED:
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, window_x, window_y, 0, 0)
                        CLICKED = False
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
    print(sum(fps_list)/len(fps_list))

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
