import os

def find_yolo_classes(directory):
    class_set = set()
    
    # Traverse the directory to find all .txt files
    for filename in os.listdir(directory):
        if filename == 'classes.txt': continue
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # Each line in YOLO format starts with the class label
                    class_label = line.split()[0]
                    class_set.add(class_label)
    
    return class_set

# Specify the directory containing the YOLO .txt files
directory_path = "YOLO Model\Data\Labeled Images"

class_set = find_yolo_classes(directory_path)
print(f"Number of classes: {len(class_set)}")
print(f"Classes used: {sorted([int(x) for x in class_set])}")

import os

def find_files_with_classes(directory, target_classes):
    files_with_classes = {}
    
    # Traverse the directory to find all .txt files
    for filename in os.listdir(directory):
        if filename == 'classes.txt': continue
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # Each line in YOLO format starts with the class label
                    class_label = line.split()[0]
                    if class_label not in target_classes:
                        files_with_classes[class_label].append(filename)
    
    return files_with_classes

# Specify the directory containing the YOLO .txt files
directory_path = "YOLO Model\Data\Labeled Images"

target_classes = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"]

files_with_classes = find_files_with_classes(directory_path, target_classes)
for class_name, files in files_with_classes.items():
    print(f"Class '{class_name}' is found in files: {files}")
