# import numpy as np

# # Example NumPy arrays
# array1 = np.array([[1, 0], [0, 1], [-1, -1]])
# array2 = np.array([[0, 1], [-1, 0], [1, 0], [0, 0]])

# # Find the intersection of the two arrays
# intersection = np.intersect1d(array1.view([('', array1.dtype)] * array1.shape[1]), 
#                               array2.view([('', array2.dtype)] * array2.shape[1]))

# # Reshape the intersection back to the original 2D shape
# intersection_2d = intersection.view(array1.dtype).reshape(-1, array1.shape[1])

# print(intersection_2d)

# import numpy as np

# # Example NumPy array
# array = np.array([[0, 1], [-1, 0], [1, 0], [0, 0]])

# # The tuple to check
# tuple_to_check = (1, 0)

# # Check if the tuple exists in the array
# exists = np.any(np.all(array == tuple_to_check, axis=1))

# print(exists)  # Output: True

# from collections import deque

# # Create a deque with a maximum length of 3
# queue = deque(maxlen=3)

# # Append elements to the deque
# queue.append((1,2))
# queue.append((2,9))
# queue.append((3,8))
# print(queue)  # Output: deque([1, 2, 3], maxlen=3)

# # Append another element (the oldest element will be removed)
# queue.append((4,4))
# print(queue)  # Output: deque([2, 3, 4], maxlen=3)


from multiprocessing import Manager, Process
import time

def process_queue(q):
    # Create a manager and a temporary queue
    manager = Manager()
    temp_queue = manager.Queue()
    
    # Transfer items from original queue to temporary queue
    while not q.empty():
        item = q.get()
        temp_queue.put(item)
    
    # Read items from the temporary queue
    temp_items = []
    while not temp_queue.empty():
        item = temp_queue.get()
        temp_items.append(item)
    
    # Transfer items back to the original queue
    for item in temp_items:
        q.put(item)
    
    temp_items2 = []
    while not temp_queue.empty():
        item = temp_queue.get()
        temp_items2.append(item)    
    
    # Print the items
    print("Items in the queue:", temp_items)
    print("Items in the queue:", temp_items2)
    print(q)

if __name__ == "__main__":
    manager = Manager()
    q = manager.Queue()
    
    # Fill the queue with some items
    for i in range(5):
        q.put(i)
    
    # Process the queue
    p = Process(target=process_queue, args=(q,))
    p.start()
    p.join()
