import xml.etree.ElementTree as ET
import os

def convert_annotation(xml_file, txt_file, image_width, image_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    with open(txt_file, 'w') as f:
        for obj in root.findall('object'):
            class_id = obj.find('name').text  # Assuming class names are used directly
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

for image_file in os.listdir('Data\\test'):
    if image_file.endswith('.png'):
        xml_file = os.path.join('Data\\test', image_file.replace('.png', '.xml'))
        txt_file = os.path.join('yolo_annotations', image_file.replace('.png', '.txt'))
        image_width, image_height = 1024, 768  # Set your image dimensions here
        convert_annotation(xml_file, txt_file, image_width, image_height)
