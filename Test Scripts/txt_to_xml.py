import os
import xml.etree.ElementTree as ET

# Define the folder paths
yolo_annotations_folder = 'TF Model\\Data\\LabeledTXT'
pascal_voc_folder = 'TF Model\\Data\\LabeledXML'
images_folder = 'TF Model\\Data\\Images'

# Define the class names
class_names = ["enemy_slug", "enemy_slug_big", "trap_off", "trap_on", "enemy_skeleton", "enemy_skeleton_shoot", "next_screen", "door", "trap_door", "gold", "treasury_open", "treasury_close", "treasury_monster", "key", "enemy_slug_boss"]

# Create the Pascal VOC folder if it doesn't exist
if not os.path.exists(pascal_voc_folder):
    os.makedirs(pascal_voc_folder)

def create_pascal_voc_xml(filename, width, height, depth, objects):
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder").text = os.path.basename(images_folder)
    filename = ET.SubElement(annotation, "filename").text = filename

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    for obj in objects:
        obj_element = ET.SubElement(annotation, "object")
        ET.SubElement(obj_element, "name").text = obj["name"]
        ET.SubElement(obj_element, "pose").text = "Unspecified"
        ET.SubElement(obj_element, "truncated").text = "0"
        ET.SubElement(obj_element, "difficult").text = "0"

        bndbox = ET.SubElement(obj_element, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["ymax"])

    return ET.ElementTree(annotation)

def convert_yolo_to_pascal_voc(yolo_annotation_file, image_file):
    image_filename = os.path.basename(image_file)
    image_name, _ = os.path.splitext(image_filename)

    # Open the image to get dimensions
    import cv2
    image = cv2.imread(image_file)
    height, width, depth = image.shape

    objects = []
    with open(yolo_annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            bbox_width = float(parts[3]) * width
            bbox_height = float(parts[4]) * height

            xmin = int(x_center - bbox_width / 2)
            ymin = int(y_center - bbox_height / 2)
            xmax = int(x_center + bbox_width / 2)
            ymax = int(y_center + bbox_height / 2)

            objects.append({
                "name": class_names[class_id],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

    pascal_voc_xml = create_pascal_voc_xml(image_filename, width, height, depth, objects)
    pascal_voc_xml.write(os.path.join(pascal_voc_folder, image_name + ".xml"))

# Process each YOLO annotation file
for yolo_file in os.listdir(yolo_annotations_folder):
    if yolo_file.endswith(".txt"):
        yolo_annotation_file = os.path.join(yolo_annotations_folder, yolo_file)
        image_file = os.path.join(images_folder, os.path.splitext(yolo_file)[0] + ".png")  # Assuming .jpg images
        if os.path.exists(image_file):
            convert_yolo_to_pascal_voc(yolo_annotation_file, image_file)
        else:
            print(f"Image file not found for annotation: {yolo_file}")
