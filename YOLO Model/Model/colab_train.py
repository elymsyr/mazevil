import yaml, requests
import os
from data_create import move_files


yolo_model = 'yolov7x'
model = 'testmodel01'
val_size = 0.2
# from shutil import rmtree; rmtree(f'YOLO Model\\Model\\{model}')

model_path = f'YOLO Model\\Model\\{model}'

os.makedirs(model_path, exist_ok=True)


# CREATE VAL AND TRAIN FILES

# move_files(model = model, val_size = val_size)



train = f"YOLO Model\\Model\\{model}\\train"
val = f"YOLO Model\\Model\\{model}\\val"

# Read class names from a text file
with open(f'{model_path}\\classes.txt', 'r') as file:
    class_names = [line.strip() for line in file.readlines()]

nc = len(class_names)

# Configuration details
config = {
    'names': class_names,  # list of class names
    'nc': nc,  # number of classes
    'val': f"./data/val",  # path to validation dataset
    'train': f"./data/train",  # path to training dataset
}

# Save the configuration to a YAML file
with open(f'YOLO Model\\Model\\{model}\\{model.lower()}_config.yaml', 'w') as file:
    yaml.safe_dump(config, file)

url = f'https://raw.githubusercontent.com/WongKinYiu/yolov7/main/cfg/training/{yolo_model}.yaml'
response = requests.get(url)
response.raise_for_status()  # Ensure we notice bad responses

# Load the YAML content
config = yaml.safe_load(response.text)

config['nc'] = nc 

with open(f"YOLO Model\\Model\\{model}\\{yolo_model}.yaml", 'w') as file:
    yaml.safe_dump(config, file)