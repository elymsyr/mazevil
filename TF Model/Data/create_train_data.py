import os, shutil, math, random

def zip_folder(folder_path, output_path, delete = False):
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f'Folder "{folder_path}" has been zipped as "{output_path}.zip"')
    # Delete the original folder
    if delete : shutil.rmtree(folder_path); print(f'Folder "{folder_path}" has been deleted')

def split_list_fraction(input_list, fraction1, fraction2):
    if not (0 <= fraction1 <= 1 and 0 <= fraction2 <= 1):
        raise ValueError("Fractions must be between 0 and 1.")
    if fraction1 + fraction2 > 1:
        raise ValueError("The sum of fractions must not exceed 1.")
    
    size1 = math.ceil(len(input_list) * fraction1)
    size2 = math.ceil(len(input_list) * fraction2)
    size3 = len(input_list) - size1 - size2

    # Shuffle the list to ensure random sampling
    shuffled_list = random.sample(input_list, len(input_list))

    # Split the shuffled list into three parts
    piece1 = shuffled_list[:size1]
    piece2 = shuffled_list[size1:size1 + size2]
    piece3 = shuffled_list[size1 + size2:]

    return piece1, piece2, piece3

def move_files(model = 'test_n', val_size = 0.1, zip = False, train_size = 0.8):
    labeled = "TF Model\\Data\\Labeled Images"
    images = "TF Model\\Data\\Images"
    processed_images = "TF Model\\Data\\Processed Images"
    test = f"TF Model\\Data\\Model Data\\{model}\\test"
    train = f"TF Model\\Data\\Model Data\\{model}\\train"
    val = f"TF Model\\Data\\Model Data\\{model}\\validation"
    
    folder_system = [test, train, val, processed_images]
    
    for folder in folder_system:
        os.makedirs(folder, exist_ok=True)
    
    # Get a list of all xml files in labeled
    xml_files = [f.replace('.xml', '') for f in os.listdir(labeled) if f.endswith('.xml')]
    folder_images = [f.replace('.png', '') for f in os.listdir(images) if f.endswith('.png')]
    folder_p_images = [f.replace('.png', '') for f in os.listdir(processed_images) if f.endswith('.png')]
    
    
    if 'classes' in xml_files: xml_files.remove('classes')

    xml_train, xml_val, xml_test  = split_list_fraction(xml_files, train_size, val_size)
    
    print(f"\n{len(folder_images)} files in Images.")
    print(f"{len(folder_p_images)} files in Processed Images.")
    
    print(f"\n{len(xml_files)} xml files found.")
    print(f"{len(xml_train)} for train.")
    print(f"{len(xml_test)} for test.")
    print(f"{len(xml_val)} for val.")

    count = 0
    for png_file in xml_files:
        if 'classes' in png_file: continue 
        count += 1
        
        if png_file in folder_images: src_png = os.path.join(images, f"{png_file}.png")
        elif png_file in folder_p_images: src_png = os.path.join(processed_images, f"{png_file}.png")
        else: print(f"Error while getting image {png_file}!"); count-= 1; continue
        
        if png_file in xml_train:
            dest_png = os.path.join(train, f"{png_file}.png")
        elif png_file in xml_test:
            dest_png = os.path.join(test, f"{png_file}.png")
        else:
            dest_png = os.path.join(val, f"{png_file}.png")

        shutil.copy(src_png, dest_png)

    print(f"{count} png files moved from Images.")

    count = 0
    for label in xml_files:
        count += 1
        src_png = os.path.join(labeled, f"{label}.xml")
        if label in xml_train:
            dest_png = os.path.join(train, f"{label}.xml")
        elif label in xml_test:
            dest_png = os.path.join(test, f"{label}.xml")
        else:
            dest_png = os.path.join(val, f"{label}.xml")
        shutil.copy(src_png, dest_png)
        
    print(f"Total {count} xml files moved.\n")
    
    shutil.copy(os.path.join(labeled, "classes.txt"), f"YOLO Model\\Model\\{model}")

    map = {
       'train': {'images': [f for f in os.listdir(train) if f.endswith('.png')], 'labels': [f for f in os.listdir(train) if f.endswith('.xml')]},
       'val': {'images': [f for f in os.listdir(val) if f.endswith('.png')], 'labels': [f for f in os.listdir(val) if f.endswith('.xml')]},
       'test': {'images': [f for f in os.listdir(test) if f.endswith('.png')], 'labels': [f for f in os.listdir(test) if f.endswith('.xml')]}
    }
    
    for key, value in map.items():
        print(f"{key}:")
        for subkey, subvalue in value.items():
            print(f"    {subkey} - {len(subvalue)}")
            
    if zip: zip_folder(f"TF Model\\Data\\Model Data\\{model}", 'TF Model\\Data\\Model Data\\images')

move_files(model='test_2', zip=True)
