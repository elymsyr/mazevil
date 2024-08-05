import os
from xml_to_csc import png_to_csv
import shutil
import xml.etree.ElementTree as ET

def zip_folder(folder_path, output_path):
    shutil.make_archive(output_path, 'zip', folder_path)
    print(f'Folder "{folder_path}" has been zipped as "{output_path}.zip"')
    # Delete the original folder
    shutil.rmtree(folder_path)
    print(f'Folder "{folder_path}" has been deleted')    

# Define the folder paths


def gather_data(images_folder: str = "Processed Images", labeled_images_folder: str = "Labeled Images", data_folder: str = "Data/all", folder_to_zip: str = 'Data', output_zip_path: str = 'Data', create_csv: bool = False):
    # Create the Data folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Get the list of files in the Labeled Images folder
    labeled_files = os.listdir(labeled_images_folder)
    print(len(labeled_files))
    for file in labeled_files:
        # Construct the full file paths
        labeled_file_path = os.path.join(labeled_images_folder, file)
        image_file_name = os.path.splitext(file)[0] + ".png"
        image_file_path = os.path.join(images_folder, image_file_name)
        data_labeled_file_path = os.path.join(data_folder, file)
        data_image_file_path = os.path.join(data_folder, image_file_name)

        # Copy the labeled file to the Data folder
        shutil.copy(labeled_file_path, data_folder)

        # Check if the corresponding image file exists and copy it
        if os.path.exists(image_file_path):
            shutil.copy(image_file_path, data_folder)

        # Modify the XML file to update the <path> element
        tree = ET.parse(data_labeled_file_path)
        root = tree.getroot()
        
        # Find the <path> element and update its text
        for path_elem in root.iter('path'):
            path_elem.text = data_image_file_path
        
        # Write the updated XML file back to the Data folder
        tree.write(data_labeled_file_path)
        
    print("Files have been copied and XML paths updated successfully.")
        
    if create_csv: png_to_csv('Data')

    # # Ensure the folder exists
    # if os.path.isdir(folder_to_zip):
    #     zip_folder(folder_to_zip, output_zip_path)
    # else:
    #     print(f'The folder "{folder_to_zip}" does not exist.')

gather_data()