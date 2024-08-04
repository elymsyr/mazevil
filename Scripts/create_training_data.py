import os
import shutil
from xml_to_csc import png_to_csv

# Define the folder paths
images_folder = "Images"
labeled_images_folder = "Labeled Images"
data_folder = "Data"

# Create the Data folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Get the list of files in the Labeled Images folder
labeled_files = os.listdir(labeled_images_folder)

for file in labeled_files:
    # Construct the full file paths
    labeled_file_path = os.path.join(labeled_images_folder, file)
    image_file_name = os.path.splitext(file)[0] + ".png"
    image_file_path = os.path.join(images_folder, image_file_name)

    # Copy the labeled file to the Data folder
    shutil.copy(labeled_file_path, data_folder)

    # Check if the corresponding image file exists and copy it
    if os.path.exists(image_file_path):
        shutil.copy(image_file_path, data_folder)

png_to_csv('Data')

print("Files have been copied successfully.")


# python object_detection\builders\model_builder_tf2_test.py