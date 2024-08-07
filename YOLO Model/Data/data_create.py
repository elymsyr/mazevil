import os, shutil, math, random

def zip_folder(folder_path, output_path, delete = False):
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f'Folder "{folder_path}" has been zipped as "{output_path}.zip"')
    # Delete the original folder
    if delete : shutil.rmtree(folder_path); print(f'Folder "{folder_path}" has been deleted')

def split_list_fraction(input_list, fraction1):
    if not 0 <= fraction1 <= 1:
        raise ValueError("The fraction must be between 0 and 1.")
    
    size1 = math.ceil(len(input_list) * fraction1)
    size2 = len(input_list) - size1
    
    # Randomly sample elements for the first piece
    piece1 = random.sample(input_list, size1)
    
    # Get the remaining elements for the second piece
    piece2 = [item for item in input_list if item not in piece1]
    
    return piece1, piece2

def move_files(model = 'TestModel001', val_size = 0.2, zip = False):
    labeled = "YOLO Model\\Data\\Labeled Images"
    images = "YOLO Model\\Data\\Images"
    train = f"YOLO Model\\Data\\Model Data\\{model}\\train"
    val = f"YOLO Model\\Data\\Model Data\\{model}\\val"
    
    folder_system = [f"YOLO Model\\Data\\Model Data\\{model}", f"{train}\\images", f"{train}\\labels", f"{val}\\images", f"{val}\\labels"]
    
    for folder in folder_system:
        os.makedirs(folder, exist_ok=True)
    
    # Get a list of all txt files in labeled
    txt_files = [f.replace('.txt', '') for f in os.listdir(labeled) if f.endswith('.txt')]
    if 'classes' in txt_files: txt_files.remove('classes')

    txt_val, txt_train = split_list_fraction(txt_files, val_size)
    
    print(f"\n{len(txt_files)} txt files found.")
    print(f"{len(txt_train)} for train.")
    print(f"{len(txt_val)} for val.\n")
    
    count = 0
    for png_file in txt_files:
        if 'classes' in png_file: continue 
        count += 1
        src_png = os.path.join(images, f"{png_file}.png")
        dest_png = os.path.join(f"{train}\\images", f"{png_file}.png") if png_file in txt_train else os.path.join(f"{val}\\images", f"{png_file}.png")

        shutil.copy(src_png, dest_png)

    print(f"Total {count} png files moved.")

    count = 0
    for label in txt_files:
        count += 1    
        src_png = os.path.join(labeled, f"{label}.txt")
        dest_png = os.path.join(f"{train}\\labels", f"{label}.txt") if label in txt_train else os.path.join(f"{val}\\labels", f"{label}.txt")

        shutil.copy(src_png, dest_png)
        
    print(f"Total {count} txt files moved.\n")
    
    shutil.copy(os.path.join(labeled, "classes.txt"), f"YOLO Model\\Data\\Model Data\\{model}")

    map = {
       'train': {'images': [f for f in os.listdir(f"{train}\\images") if f.endswith('.png')], 'labels': [f for f in os.listdir(f"{train}\\labels") if f.endswith('.png')]},
       'val': {'images': [f for f in os.listdir(f"{val}\\images") if f.endswith('.txt')], 'labels': [f for f in os.listdir(f"{val}\\labels") if f.endswith('.txt')]}
    }
    
    for key, value in map.items():
        print(f"{key}:")
        for subkey, subvalue in value.items():
            print(f"    {subkey} - {len(subvalue)}")
            
    if zip: zip_folder(f"YOLO Model\\Data\\Model Data\\{model}", f'yolo_data_{model.lower()}') 

move_files(model = 'TestModel001', val_size = 0.25)
