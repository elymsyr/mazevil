import xml.etree.ElementTree as ET
import os, shutil

# Define the classes in the order they appear in the YOLO classes file
classes = [
    "enemy_slug", "enemy_slug_big", "trap_off", "trap_on", "enemy_skeleton", 
    "enemy_skeleton_shoot", "next_screen", "door", "trap_door", "gold", 
    "treasury_open", "treasury_close", "treasury_monster", "key", "enemy_slug_boss"
]

def convert_xml_to_yolo(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get the size of the image
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    with open(output_file, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue
            class_id = classes.index(class_name)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Convert to YOLO format
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

def convert_all(input_path = 'YOLO Model\\Data\\Labeled Images', output_path = 'YOLO Model\\Data\\XML'):
    xml = [f.replace('.xml', '') for f in os.listdir(input_path) if f.endswith('.xml')]
    for file in xml:
        convert_xml_to_yolo(os.path.join(input_path, f"{file}.xml"), os.path.join(input_path, f"{file}.txt"))
        shutil.move(os.path.join(input_path, f"{file}.xml"), output_path)
        print(file)

    
convert_all()
