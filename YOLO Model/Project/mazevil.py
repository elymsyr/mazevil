import dxcam, cv2, win32gui, win32api, win32con, time, heapq
from ultralytics import YOLO
import numpy as np


class Mazevil():
    def __init__(self, model_path, window_title = 'Mazevil'):
        self.window_title = window_title
        self.CLICKED = False
        self.lower_bound, self.upper_bound = np.array([24,20,37]), np.array([24,20,37])

        self.path_finding = GreedyBFS()
        self.model: YOLO = YOLO(model_path)

        self.hwnd = win32gui.FindWindow(None, self.window_title)
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        self.window_x = self.window_rect[0]
        self.window_y = self.window_rect[1]

        self.max_distance = 200
        self.path_found = []

        self.cam: dxcam.DXCamera = dxcam.create(output_color="BGR")
        self.left, self.top, self.right, self.bottom = win32gui.GetWindowRect(self.hwnd)

        self.path_window_image = None
        self.window_image = self.cam.grab(region=(self.left+8, self.top+50, self.right-8, self.bottom-58))
        self.masked_cleared = (cv2.inRange(self.window_image, self.lower_bound, self.upper_bound) > 0).astype(np.uint8)
        
        self.multip = 2
        
        # self.width, self.height = int(self.window_image.shape[1]//self.multip), int(self.window_image.shape[0]//self.multip)
        self.width, self.height = int(self.window_image.shape[1]), int(self.window_image.shape[0])
        self.center = (int(self.width/2), int(self.height/2+11))

        self.path_width, self.path_height = int(self.window_image.shape[1]//self.multip), int(self.window_image.shape[0]//self.multip)
        self.path_center = (int(self.path_width/2), int(self.path_height/2+11))
        

        self.fps_list = []

        self.directions = np.array([
            [0, 1],   # Up
            [1, 1],   # Up-Right
            [1, 0],   # Right
            [1, -1],  # Down-Right
            [0, -1],  # Down
            [-1, -1], # Down-Left
            [-1, 0],  # Left
            [-1, 1],  # Up-Left
        ], dtype=np.float64)

    def find_optimal_direction(self, object_positions, directions):
        pass

    def path_detection(self, boxes):
        self.path_window_image = np.stack([self.masked_cleared, self.masked_cleared, self.masked_cleared], axis=-1) * 255   
        for box in boxes:
            self.path_window_image = cv2.rectangle(img=self.path_window_image, **box)
        self.path_window_image = cv2.resize(self.path_window_image, (self.path_width, self.path_height), interpolation=cv2.INTER_NEAREST)

        self.path_window_image = cv2.floodFill(self.path_window_image, None, self.path_center, (250, 0, 0))[1]
        mask = cv2.inRange(self.path_window_image, np.array([250, 0, 0]), np.array([250, 0, 0]))
        masked_cleared = (mask > 0).astype(np.uint8)
        self.path_window_image = np.stack([masked_cleared, masked_cleared, masked_cleared], axis=-1) * 255

    def shoot_closest(self, enemies, shoot):
        target = None
        for enemy in enemies:
            self.path_window_image = cv2.circle(self.path_window_image, (enemy[1][0]//self.multip, enemy[1][1]//self.multip), 3, (0, 0, 256))
            if int(enemy[0]) != 5 and not target:
                target = enemy[1]
        if shoot and target:
            screen_x = target[0]+self.window_x+9
            screen_y = target[1]+self.window_y+53
            win32api.SetCursorPos((screen_x, screen_y))
            if not self.CLICKED: win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
            self.CLICKED = True

    def find_directions(self):
        open_directions = []
        check_up = 1
        for direction in self.directions:
            x = int(self.path_center[0] + direction[0] * 12)
            x1 = int(self.path_center[0] + direction[0] * 9)            
            y = int(self.path_center[1] + direction[1] * 12)
            y1 = int(self.path_center[1] + direction[1] * 9)           
            if np.array_equal(self.path_window_image[y, x], [255,255,255])\
                and np.array_equal(self.path_window_image[y1, x1], [255,255,255]):
                if np.array_equal(direction, [0,-1]): continue
                else:
                    open_directions.append(direction)
            elif np.array_equal(direction, [-1, -1]) or np.array_equal(direction, [1, -1]):
                check_up -= 1
        if check_up > 0:
            open_directions.append([0, -1])

        return open_directions

    def window_dxcam(self, shoot: bool = False, draw: bool = True, imgsz: int = 480, show_result:bool = True, path: bool = True, model_detect: bool = True, min_conf: float = 0.4):
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
                    
                    self.mask = cv2.inRange(cv2.cvtColor(self.window_image, cv2.COLOR_BGR2RGB), self.lower_bound, self.upper_bound)
                    self.masked_cleared = (self.mask > 0).astype(np.uint8)
                    
                    if path:
                        enemies = []
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
                                    if int(class_id) == 2 or int(class_id) == 3: # trap off
                                        color = (0, 0, 0) if int(class_id) == 2 else (255, 255, 255)
                                        boxes.append({"pt1": (x1-2, y1),"pt2": (x2+2, y2+2),"color": color,"thickness": -1})
                                    elif int(class_id) in [0,1,4,5,12,14]:
                                        dist = self.path_finding.euclidean_distance((x,y), self.center)
                                        weight = 100 / dist * (4 if int(class_id) == 5 else 2)
                                        enemies.append([int(class_id), (x,y), weight, dist])
                                    elif int(class_id) in [7,9,10,11,13]:
                                        rewards.append((x,y, int(class_id)))
                                    else: continue

                        enemies = sorted(enemies, key=lambda x: x[-1])

                        self.path_detection(boxes = boxes)

                        open_directions = self.find_directions()

                        self.window_image = cv2.circle(self.window_image, self.center, self.max_distance, (10,20,128), 1)

                        if enemies:
                            self.shoot_closest(enemies=enemies, shoot=shoot)
                            # optimal_direction = self.find_optimal_direction(enemies, directions=np.array(open_directions) if len(open_directions)>0 else self.directions)
                        elif self.CLICKED:
                            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, self.window_x, self.window_y, 0, 0)
                            self.CLICKED = False

                        for direction in open_directions:
                            x = int(self.path_center[0] + direction[0] * 12)
                            y = int(self.path_center[1] + direction[1] * 12)
                            self.path_window_image = cv2.line(self.path_window_image, self.path_center, (x,y), [10,20,138],1)

                        # if enemies:
                        #     x = int(self.center[0] + optimal_direction[0] * 12)
                        #     y = int(self.center[1] + optimal_direction[1] * 12)
                        #     self.window_image = cv2.line(self.window_image, self.center, (x,y), [128,128,256],2)

                    if draw:
                        for result in results:
                            self.window_image = result.plot(img = self.window_image)

                if show_result:
                    cv2.putText(self.window_image, f"FPS: {int(fps) if abs(self.fps_list[-1] - fps) > 1 else int(self.fps_list[-1])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(self.path_window_image, f"FPS: {int(fps) if abs(self.fps_list[-1] - fps) > 1 else int(self.fps_list[-1])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.imshow(f'{self.window_title} YOLOV8 IMAGE', self.window_image)
                    cv2.imshow(f'{self.window_title} YOLOV8 PATH', self.path_window_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            # if len(self.fps_list) > 100: break
        
        print(sum(self.fps_list)/len(self.fps_list))
        cv2.destroyAllWindows()

class GreedyBFS():
    def __init__(self):
        pass

    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def greedy_best_first_search(self, grid, start, goal):
        """Perform Greedy Best-First Search on a grid."""
        rows, cols = grid.shape[0], grid.shape[1]
        open_set = []
        heapq.heappush(open_set, (self.manhattan_heuristic(start, goal), start))
        
        came_from = {}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if self.manhattan_heuristic(current, goal) < 5:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < rows) and (0 <= neighbor[1] < cols):
                    if np.array_equal(grid[neighbor[0], neighbor[1]], 1):
                        if neighbor not in came_from:
                            came_from[neighbor] = current
                            heapq.heappush(open_set, (self.manhattan_heuristic(neighbor, goal), neighbor))

        return []

    def manhattan_heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def draw_path_on_image(self, image, path: list, color=(0, 255, 0), thickness=2):
        """
        Draws the path on the given image.
        
        Parameters:
        - image: The image on which to draw the path. It should be a numpy ndarray of shape (381, 800, 3).
        - path: The list of points representing the path.
        - color: The color of the path (default is green).
        - thickness: The thickness of the path lines (default is 2).
        
        Returns:
        - The image with the path drawn on it.
        """
        if path and len(path) > 1:
            for i in range(0, len(path) - 1):
                pt1 = path[i]
                pt2 = path[i+1]
                # Draw line between consecutive points
                cv2.line(image, pt1, pt2, color, thickness)
        return image

if '__main__'== __name__:
    model_name = 'test_1'
    model_path = f'YOLO Model\\Model\\Trained Models\\{model_name}\\train\\weights\\best.pt'
    conf = {
        'imgsz': 480,
        'show_result' : True,
        'shoot' : True,
        'draw' : True,
        'path' : True,
        'model_detect': True,
        'min_conf' : 0.4
    }
    agent = Mazevil(model_path=model_path)
    agent.window_dxcam(**conf)
    
    # path_finding = GreedyBFS()
    # loaded_array = np.load('array_file.npy')
    # center = (int(loaded_array.shape[1]/2), int(loaded_array.shape[0]/2))
    # # Find the indices of all pixels that are [0, 0, 0]
    
    # print(loaded_array)
    
    # black_pixels = np.argwhere((loaded_array == 1).all())
    
    # # Check if there are any black pixels
    # if not len(black_pixels) > 0:
    #     print("No black pixels found.")
    
    # # Randomly choose one of the black pixels
    # random_index = np.random.choice(len(black_pixels))
    # random_pixel = black_pixels[random_index]
    # # Return the (x, y) coordinates
    # y,x = tuple(random_pixel)
    # window_image = np.stack([loaded_array,loaded_array,loaded_array], axis=-1) * 255
    # # window_image = loaded_array
    # window_image = cv2.circle(window_image, center, 2, (0,255,0), -1)
    # print(window_image.shape)
    # cv2.imshow('a', window_image)
    # cv2.waitKey(1000)
    # print(f"{x=}  {y=}  {center=}  {loaded_array.shape=}")
    # path_found = path_finding.greedy_best_first_search(start=(x,y), goal=center, grid=loaded_array)
    # print(path_found)