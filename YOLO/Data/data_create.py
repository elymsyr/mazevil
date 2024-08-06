import os, shutil, math, random

def zip_folder(folder_path, output_path):
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f'Folder "{folder_path}" has been zipped as "{output_path}.zip"')
    # Delete the original folder
    shutil.rmtree(folder_path)
    print(f'Folder "{folder_path}" has been deleted')

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

def move_matching_files(model = 'TestModel001', val_size = 0.2):
    labeled = f"YOLO/Data/Model Data/{model}"
    images = "YOLO/Data/Images"
    # model_all = f'YOLO/Data/Model Data/{model}/all'
    train = f"YOLO/Data/Model Data/{model}/train"
    val = f"YOLO/Data/Model Data/{model}/val"
    
    # Get a list of all txt files in labeled
    txt_files = [f for f in os.listdir(labeled) if f.endswith('.txt')]
    
    txt_val, txt_train = split_list_fraction(txt_files, 0.2)
    
    # Get a list of all png files in images
    png_files = [f for f in os.listdir(images) if f.endswith('.png')]

    # Create a set of txt file names without the extension
    txt_names_train = set(os.path.splitext(f)[0] for f in txt_train)
    txt_names_val = set(os.path.splitext(f)[0] for f in txt_val)
    
    print(f"\n{len(txt_files)} txt files found.")
    print(f"{len(txt_train)} for train.")
    print(f"{len(txt_val)} for val.\n")
    
    count = 0
    for png_file in txt_files:
        count += 1
        src_png = os.path.join(images, f"{png_file}.png")
        dest_png = os.path.join(f"{train}/images", f"{png_file}.png") if png_file in txt_train else os.path.join(f"{val}/images", f"{png_file}.png")

        shutil.move(src_png, dest_png)

    print(f"\nTotal {count} png files moved.")

    count = 0
    for label in txt_files:
        count += 1    
        src_png = os.path.join(labeled, f"{label}.txt")
        dest_png = os.path.join(f"{train}/labels", f"{label}.txt") if png_file in txt_train else os.path.join(f"{val}/labels", f"{label}.txt")

        shutil.move(src_png, dest_png)
        
    print(f"\nTotal {count} txt files moved.")

    map = {
       'train': {'images': [f for f in os.listdir(f"{train}/images") if f.endswith('.png')], 'labels': [f for f in os.listdir(f"{train}/labels") if f.endswith('.png')]},
       'val': {'images': [f for f in os.listdir(f"{val}/images") if f.endswith('.png')], 'labels': [f for f in os.listdir(f"{val}/labels") if f.endswith('.png')]}
    }
    
    for key, value in map.items():
        print(f"{key}:")
        for subkey, subvalue in value.items():
            print(f"    {subkey} - {len(subvalue)}    {subvalue if len(subvalue) < 10 else ""}")
