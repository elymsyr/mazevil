import yaml, requests
import shutil, os
from data_create import move_files


yolo_model = 'yolov7x'
model = 'testdata01'
val_size = 0.2

model_path = f'YOLO Model\\Model\\{model}'

os.makedirs(model_path, exist_ok=True)

move_files(model = model, val_size = val_size)

train = f"YOLO Model\\Model\\{model}\\train"
val = f"YOLO Model\\Model\\{model}\\val"

# Read class names from a text file
with open(f'{model_path}\\classes.txt', 'r') as file:
    class_names = [line.strip() for line in file.readlines()]

class_names_str = "["
for name in class_names[:-1]:
    class_names_str += f"{name}, "
class_names_str += f"{class_names[-1]}]"

nc = len(class_names)

# Configuration details
config = {
    'train': f"{model}\\train",  # path to training dataset
    'val': f"{model}\\val",  # path to validation dataset
    'nc': nc,  # number of classes
    'names': class_names_str  # list of class names
}

# Save the configuration to a YAML file
with open(f'YOLO Model\\Model\\{model}\\{model.lower()}_config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)


url = f'https://raw.githubusercontent.com/WongKinYiu/yolov7/main/cfg/training/{yolo_model}.yaml'
response = requests.get(url)
response.raise_for_status()  # Ensure we notice bad responses

# Load the YAML content
config = yaml.safe_load(response.text)

config['nc'] = nc 

with open(f"YOLO Model\\Model\\{model}\\{yolo_model}.yaml", 'w') as file:
    yaml.dump(config, file, default_flow_style=False)