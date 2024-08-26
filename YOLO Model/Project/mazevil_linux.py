def run_commands():
    import subprocess
    try:
        # Change permissions
        subprocess.run(['sudo', 'chmod', '666', '/dev/uinput'], 
                       check=True, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)
        print("Permissions for /dev/uinput changed successfully.")

        # Load uinput module
        subprocess.run(['sudo', 'modprobe', 'uinput'], 
                       check=True, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)
        print("uinput module loaded successfully.")

    except subprocess.CalledProcessError as e:
        print("Error executing command:", e.stderr)

# run_commands()

import cv2, time, heapq, multiprocessing
from ultralytics import YOLO
import numpy as np
from collections import deque
from Xlib import X, display
import mss
import libclicker
from pyKey import pressKey, releaseKey
import matplotlib.pyplot as plt

def find_farthest_point(binary_array):
    # Get the indices of all points with value 1
    points = np.argwhere(binary_array == 1)
    
    if len(points) <= 1:
        # If there's only one or no point, return the point itself or None
        return points[0] if len(points) == 1 else None
    
    max_distance = 0
    farthest_point = points[0]
    
    # Compare each point to every other point
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Calculate Euclidean distance between points[i] and points[j]
            distance = np.linalg.norm(points[i] - points[j])
            # Update the farthest point if this distance is the largest so far
            if distance > max_distance:
                max_distance = distance
                farthest_point = points[j]
    
    return tuple(farthest_point)

class Mazevil():
    def __init__(self, model_path, window_title = 'Mazevil'):
        self.window_title = window_title
        self.CLICKED = False
        self.lower_bound, self.upper_bound = np.array([24,20,37]), np.array([24,20,37])
        
        self.path_finding = GreedyBFS()
        self.model: YOLO = YOLO(model_path)

        self.path = None

        self.display = display.Display()
        self.root = self.display.screen().root
        self.window, self.monitor = self.find_window()

        self.max_distance = 200
        self.path_found = []
        self.multip = 2

        self.bug_counter = 0

        manager = multiprocessing.Manager()
        self.shared_data = manager.dict()
        self.shared_data['move'] = (2,2)

        self.fps_list = []

        self.directions = np.array([
            [1, 1],   # Down-Right
            [0, 1],   # Down
            [1, 0],   # Right
            [1, -1],  # Up-Right
            [0, -1],  # Up
            [-1, -1], # Up-Left
            [-1, 0],  # Left
            [-1, 1],  # Down-Left
        ], dtype=np.float64)

        self.current_keys = set()
        self.key_map = {
            (1, 1): ('S', 'D'),
            (0, 1): ('S',),
            (1, 0): ('D',),
            (1, -1): ('W', 'D'),
            (0, -1): ('W',),
            (-1, -1): ('W', 'A'),
            (-1, 0): ('A',),
            (-1, 1): ('S', 'A'),
            (2,2): ()
        }

    def capture(self, sct):
        img = sct.grab(self.monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

    def create_monitor(self, window):
        geometry = window.query_tree().parent.get_geometry()
        return {
            "top": int(geometry.y)+50,
            "left": geometry.x+1,
            "width": geometry.width-1,
            "height": int(geometry.height)-108}

    def find_window(self):
        window = None
        window_id = None
        window_ids = self.root.get_full_property(self.display.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType).value
        for window_id in window_ids:
            window = self.display.create_resource_object('window', window_id)
            window_name_str = window.get_wm_name()
            if window_name_str and self.window_title in window_name_str:
                window = window
                window_id = window_id
                break
        monitor = self.create_monitor(window=window) if window else None
        return window, monitor

    def click(self, x, y):
        libclicker.click(x, y)

    def update_keys(self):
        # Get the target keys based on the input tuple
        if (not self.shared_data['move'] == (2,2)) and (not self.shared_data['move'] == (3,3)):
            target_keys = self.key_map.get(self.shared_data['move'], ())
            target_keys = list(target_keys)
            keys_to_press = [key for key in target_keys if key not in self.current_keys]
            keys_to_release = [key for key in self.current_keys if key not in target_keys]

            for key in keys_to_press:
                pressKey(key)

            for key in keys_to_release:
                releaseKey(key)

            # Update the current keys set
            self.current_keys = target_keys
        elif self.shared_data['move'] == (2,2):
            for key in ['W', 'A', 'S', 'D']:releaseKey(key); self.current_keys=[]
            self.shared_data['move'] = (3,3)
        if len(self.current_keys) > 0: print(self.current_keys)

    def move_player(self):
        while True:
            time.sleep(0.5)
            self.update_keys()

    def start_movement(self):
        # Create a process to run the move_player method
        # Ensure `move_player` does not depend on unpickleable attributes
        process = multiprocessing.Process(target=self.move_player)
        process.start()
        return process

    def stop_movement(self, process):
        # Terminate the process
        process.terminate()
        process.join()

    def optimal_direction(self, enemies, open_directions):
        enemy_positions = enemies[:, 1]
        weights =  enemies[:, 2]

        resultant_vector = np.zeros(2)
        for enemy_position, weight in zip(enemy_positions, weights):
            vector_to_player = np.array([self.center[0], self.center[1]]) - enemy_position
            normalized_vector = vector_to_player / np.linalg.norm(vector_to_player)
            weighted_vector = normalized_vector * weight
            resultant_vector += weighted_vector

        # Find the closest direction
        norms = np.linalg.norm(open_directions - resultant_vector, axis=1)
        best_direction_index = np.argmin(norms)
        best_direction = open_directions[best_direction_index]

        return best_direction

    def find_closest_non_zero_neighbor(self, grid, point = None):
        y,x = self.path_center if not point else point
        if np.array_equal(grid[x, y], [255,255,255]):
            # Define the neighbors' relative positions
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            min_distance = float('inf')
            closest_neighbor = None

            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                # Check if the neighbor is within the grid bounds
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    if not np.array_equal(grid[nx, ny], [255,255,255]):
                        distance = np.linalg.norm(np.array([nx, ny]) - np.array([x, y]))
                        if distance < min_distance:
                            min_distance = distance
                            closest_neighbor = (ny,nx)

            # while np.array_equal(grid[closest_neighbor[1], closest_neighbor[0]], [0, 0, 0]):
            #     closest_neighbor = self.find_closest_non_zero_neighbor(grid)
            
            return closest_neighbor
        return self.path_center

    def path_detection(self, boxes):
        test_path_window_image = np.stack([self.masked_cleared, self.masked_cleared, self.masked_cleared], axis=-1) * 255
        for box in boxes:
            test_path_window_image = cv2.rectangle(img=test_path_window_image, **box)
        test_path_window_image = cv2.resize(test_path_window_image, (self.path_width, self.path_height), interpolation=cv2.INTER_NEAREST)

        test_path_window_image = cv2.line(test_path_window_image, (self.path_center[0]-10, self.path_center[1]-5), (self.path_center[0]+10, self.path_center[1]-5), (0,255,0), 1)

        fl_point = self.find_closest_non_zero_neighbor(test_path_window_image)

        test_path_window_image = cv2.floodFill(test_path_window_image, None, fl_point, (250, 0, 0))[1]
        mask = cv2.inRange(test_path_window_image, np.array([250, 0, 0]), np.array([250, 0, 0]))
        masked_cleared = (mask > 0).astype(np.uint8)
        
        far_point = find_farthest_point(masked_cleared)
        
        test_path_window_image = np.stack([masked_cleared, masked_cleared, masked_cleared], axis=-1) * 255

        # far_point = self.bfs(test_path_window_image)
        print(far_point)
        if far_point: cv2.circle(test_path_window_image, far_point, 2, (0,255,20), 4)

        self.path_window_image = test_path_window_image

    # def bfs(self, image):
    #     # bfs_path_image = cv2.resize(image, (self.path_width//4, self.path_height//4), interpolation=cv2.INTER_NEAREST)
    #     # cv2.imshow('path',bfs_path_image)        
    #     mask = cv2.inRange(image, np.array([250, 0, 0]), np.array([250, 0, 0]))
    #     masked_cleared = (mask > 0).astype(np.uint8)
    #     # print(masked_cleared[0:10, 0:10])
        
    #     return far_point
    
    def shoot_closest(self, enemies, shoot):
        target = None
        for enemy in enemies:
            self.path_window_image = cv2.circle(self.path_window_image, (enemy[1][0]//self.multip, enemy[1][1]//self.multip), 3, (0, 0, 256))
            if int(enemy[0]) != 5 and not target:
                target = enemy[1]
        if shoot and target:
            screen_x = target[0]+self.monitor["left"]+63
            screen_y = target[1]+self.monitor["top"]+33
            self.bug_counter += 1
            self.click(screen_x, screen_y)

    def find_directions(self):
        open_directions = []
        check_up = 1
        for direction in self.directions:
            x = int(self.path_center[0] + direction[0] * 16)
            x1 = int(self.path_center[0] + direction[0] * 8)
            y = int(self.path_center[1] + direction[1] * 16)
            y1 = int(self.path_center[1] + direction[1] * 8)
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

    def window_linux(self, shoot: bool = False, draw: bool = True, imgsz: int = 480, show_result:bool = True, path: bool = True, model_detect: bool = True, min_conf: float = 0.4):
        with mss.mss() as sct:
            
            self.path_window_image = None

            # self.window_image = self.cam.grab(region=(self.left+8, self.top+50, self.right-8, self.bottom-58))
            self.window_image = self.capture(sct)
            self.mask = cv2.inRange(cv2.cvtColor(self.window_image, cv2.COLOR_BGR2RGB), self.lower_bound, self.upper_bound)
            self.masked_cleared = (self.mask > 0).astype(np.uint8)

            # self.width, self.height = int(self.window_image.shape[1]//self.multip), int(self.window_image.shape[0]//self.multip)
            self.width, self.height = int(self.window_image.shape[1]), int(self.window_image.shape[0])
            self.center = (int(self.width/2), int(self.height/2+11))

            self.path_width, self.path_height = int(self.window_image.shape[1]//self.multip), int(self.window_image.shape[0]//self.multip)
            self.path_center = (int(self.path_width/2), int(self.path_height/2+11))

            # print(self.width, self.height, self.center, self.path_width, self.path_center)

            prevTime = 0
            fps = 0
            self.fps_list = []

            while True:
                self.window_image = self.capture(sct)
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
                            in_maze_room = False
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
                                            weight = 100 / dist * (40 if int(class_id) == 5 else 2)
                                            enemies.append([int(class_id), [x,y], weight, dist])
                                            if int(class_id) != 0: in_maze_room = True
                                        elif int(class_id) in [7,9,10,11,13]:
                                            rewards.append((x,y, int(class_id)))
                                        else: continue
                            
                            if in_maze_room:
                                enemies = [enemy for enemy in enemies if enemy[0] != 0]
                                self.max_distance = 600
                            else: self.max_distance = 200
                            
                            enemies = [enemy for enemy in enemies if enemy[-1] < self.max_distance]
                            enemies = sorted(enemies, key=lambda x: x[-1])

                            self.path_detection(boxes = boxes)

                            open_directions = self.find_directions()

                            self.window_image = cv2.circle(self.window_image, self.center, self.max_distance, (10,20,128), 1)
                            self.path_window_image = cv2.circle(self.path_window_image, self.path_center, int(self.max_distance//self.multip), (10,20,128), 1)

                            if enemies:
                                self.shoot_closest(enemies=enemies, shoot=shoot)
                                enemies_array = np.array(enemies, dtype="object")
                                optimal_direction = self.optimal_direction(enemies_array, np.array(open_directions) if len(open_directions)>0 else self.directions)
                            elif self.CLICKED:
                                self.CLICKED = False
                            elif self.bug_counter > 0: self.bug_counter -= 1

                            for direction in open_directions:
                                x = int(self.path_center[0] + direction[0] * 16)
                                y = int(self.path_center[1] + direction[1] * 16)
                                self.path_window_image = cv2.line(self.path_window_image, self.path_center, (x,y), [10,20,138],1)

                            if enemies:
                                # x = int(self.path_center[0] + optimal_direction[0] * 12)
                                # y = int(self.path_center[1] + optimal_direction[1] * 12)
                                # self.path_window_image = cv2.line(self.path_window_image, self.path_center, (x,y), [128,128,256],2)

                                x = int(self.center[0] + optimal_direction[0] * 24)
                                y = int(self.center[1] + optimal_direction[1] * 24)
                                self.window_image = cv2.line(self.window_image, self.center, (x,y), [128,128,256],2)

                                self.shared_data['move'] = tuple(optimal_direction) if enemies[0][-1] < self.max_distance//2 else (2,2)
                            elif (not self.shared_data['move'] == (2,2)) and (not self.shared_data['move'] == (3,3)): self.shared_data['move'] = (2,2)
                        if draw:
                            for result in results:
                                self.window_image = result.plot(img = self.window_image)

                    if show_result:
                        cv2.putText(self.window_image, f"FPS: {int(fps) if abs(self.fps_list[-1] - fps) > 1 else int(self.fps_list[-1])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(self.path_window_image, f"FPS: {int(fps) if abs(self.fps_list[-1] - fps) > 1 else int(self.fps_list[-1])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow(f'{self.window_title} YOLOV8 IMAGE', self.window_image)
                        cv2.imshow(f'{self.window_title} YOLOV8 PATH', self.path_window_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                if self.bug_counter > 1000: break

        print(sum(self.fps_list)/len(self.fps_list), len(self.fps_list))
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

class BFSPathfinder():
    def __init__(self, start):
        self.start = start

        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    def find_path(self, goal, grid):
        n = len(grid)
        queue = deque([self.start])
        visited = set()
        visited.add(self.start)
        parent = {self.start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                return self.reconstruct_path(parent, goal)

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if (self.is_valid(neighbor, grid, n) and neighbor not in visited):
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

        return None  # No path found

    def is_valid(self, position, grid, n):
        row, col = position
        return (0 <= row < n and
                0 <= col < n and
                grid[row][col] == 0)

    def reconstruct_path(self, parent, goal):
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        return path[::-1]  # Return reversed path

if '__main__'== __name__:
    model_name = 'test_1'
    model_path = f'YOLO Model/Model/Trained Models/{model_name}/train/weights/best.pt'
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
    process = None
    
    process = agent.start_movement()
    
    agent.window_linux(**conf)
    
    if process: agent.stop_movement(process)
