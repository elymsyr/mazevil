import time
import dxcam, cv2, win32gui, win32api, win32con
from ultralytics import YOLO
import numpy as np


class Mazevil(object):
    def __init__(self, model_path, window_title = 'Mazevil'):
        self.window_title = window_title
        self.CLICKED = False
        self.lower_bound, self.upper_bound = np.array([24,20,37]), np.array([24,20,37])
        
        self.model: YOLO = YOLO(model_path)
        
        self.hwnd = win32gui.FindWindow(None, self.window_title)
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        self.window_x = self.window_rect[0]
        self.window_y = self.window_rect[1]
        
        self.max_distance = 200
        
        self.cam: dxcam.DXCamera = dxcam.create(output_color="BGR")
        self.left, self.top, self.right, self.bottom = win32gui.GetWindowRect(self.hwnd)
        window_image = self.cam.grab(region=(self.left+8, self.top+50, self.right-8, self.bottom-58))
        self.height, self.width = window_image.shape[:2]
        self.center = (int(self.width/2), int(self.height/2+30))
        
        self.fps_list = []
        
        self.directions = np.array([
            [0, 1],   # Up
            [1, 1],   # Up-Right
            [1, 0],   # Right
            [1, -1],  # Down-Right
            [0, -1],  # Down
            [-1, -1], # Down-Left
            [-1, 0],  # Left
            [-1, 1]   # Up-Left
        ], dtype=np.float64)
        
    def find_optimal_direction(self, object_positions, directions):
        # Ensure that object_weights has the same length as object_positions
        object_weights = [x[0] for x in object_positions]
        object_positions = [x[1] for x in object_positions]
        # Compute the weighted optimal movement vector
        optimal_vector = np.zeros(2, dtype=np.float64)
        for obj, weight in zip(object_positions, object_weights):
            obj_vector = np.array(obj) - np.array(self.center)
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
    
    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)    
    
    def path_detection(self, boxes, rgb):
        mask = cv2.inRange(rgb, self.lower_bound, self.upper_bound)
        masked_cleared = (mask > 0).astype(np.uint8)
        window_image = np.stack([masked_cleared, masked_cleared, masked_cleared], axis=-1) * 255   
        for box in boxes:
            cv2.rectangle(img=window_image, **box)
        window_image = cv2.floodFill(window_image, None, self.center, (250, 0, 0))[1]
        # window_image = cv2.circle(window_image, center, 1, (0, 0, 256))
        mask = cv2.inRange(window_image, np.array([250, 0, 0]), np.array([250, 0, 0]))
        masked_cleared = (mask > 0).astype(np.uint8)
        self.window_image = np.stack([masked_cleared, masked_cleared, masked_cleared], axis=-1) * 255   
    
    def enemy_found(self, enemies):
        choosen_enemy = enemies[0][1]
        screen_x = choosen_enemy[0]+self.window_x+8
        screen_y = choosen_enemy[1]+self.window_y+54
        win32api.SetCursorPos((screen_x, screen_y))
        if not self.CLICKED: win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
        self.CLICKED = True
    
    def find_directions(self):
        open_directions = []
        for direction in self.directions:
            x = int(self.center[0] + direction[0] * 21)
            y = int(self.center[1] + direction[1] * 21)
            pt2 = (x, y)
            if np.array_equal(self.window_image[y, x], [255,255,255]):
                self.window_image = cv2.line(self.window_image, self.center, pt2, [10,20,128],1)
                open_directions.append(direction)
        return open_directions

    def window_dxcam(self, draw: bool = True, imgsz: int = 480, show_result:bool = True, path: bool = True, model_detect: bool = True, min_conf: float = 0.4):
        self.hwnd = win32gui.FindWindow(None, self.window_title)
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        self.window_x = self.window_rect[0]
        self.window_y = self.window_rect[1]
        
        prevTime = 0
        fps = 0        
        self.fps_list = []
        
        while True:
            self.left, self.top, self.right, self.bottom = win32gui.GetWindowRect(self.hwnd)
            top_offset, bottom_offset = 50, 58
            self.window_image = self.cam.grab(region=(self.left+8, self.top+top_offset, self.right-8, self.bottom-bottom_offset))
            if self.window_image is not None:

                currTime = time.perf_counter()
                fps = 1 / (currTime - prevTime)
                self.fps_list.append(fps)
                prevTime = currTime

                if model_detect: 
                    results = self.model(classes=[0,1,2,3,4,5,7,9,10,11,12,13,14],device=0,source=self.window_image, conf=min_conf, imgsz=imgsz, stream=True, verbose=False)
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
                        self.path_detection(boxes = boxes, rgb=cv2.cvtColor(self.window_image, cv2.COLOR_BGR2RGB))
                        enemies = sorted(
                            [point for point in enemy if self.euclidean_distance(point[1], self.center) <= self.max_distance],
                            key=lambda point: self.euclidean_distance(point[1], self.center)
                        )
                        open_directions = self.find_directions()
                        if enemies:
                            for enemy in enemies:
                                self.window_image = cv2.circle(self.window_image, enemy[1], 3, (0, 0, 256))
                            
                            self.enemy_found(enemies=enemies)
                            direction = self.find_optimal_direction(enemies, directions=np.array(open_directions) if len(open_directions)>0 else self.directions)
                            pt2 = (int(self.center[0] + direction[0] * 20), int(self.center[1] + direction[1] * 20))  # End point
                            self.window_image = cv2.line(self.window_image, self.center, pt2, [128,128,256],2)
                        elif self.CLICKED:
                            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, self.window_x, self.window_y, 0, 0)
                            self.CLICKED = False
                    if draw:
                        for result in results:
                            self.window_image = result.plot(img = self.window_image)

                if show_result:
                    cv2.putText(self.window_image, f"FPS: {int(fps) if abs(self.fps_list[-1] - fps) > 1 else int(self.fps_list[-1])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(f'{self.window_title} YOLOV8', self.window_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        cv2.destroyAllWindows()
        del self.cam
        print(sum(self.fps_list)/len(self.fps_list))


if '__main__'== __name__:
    model_name = 'test_1'
    model_path = f'YOLO Model\\Model\\Trained Models\\{model_name}\\train\\weights\\best.pt'
    conf = {
        'imgsz': 480,
        'show_result' : True,
        'draw' : True,
        'path' : True,
        'model_detect': True,
        'min_conf' : 0.4
    }    
    agent = Mazevil(model_path=model_path)
    while True:
        # input('start...')
        agent.window_dxcam(**conf)