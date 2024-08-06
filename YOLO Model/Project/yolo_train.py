# # Install dependencies and clone Darknet repository
# !apt-get install -y libopencv-dev
# !git clone https://github.com/AlexeyAB/darknet.git
# %cd darknet
# !make

# # Upload your dataset (images and labels)
# from google.colab import files
# uploaded = files.upload()

# # Create directories for dataset and configuration
# !mkdir -p data/images
# !mkdir -p data/labels
# !mkdir -p cfg
# !mkdir -p backup

# # Move uploaded files to the appropriate directories
# import shutil

# # Assuming the uploaded files are named 'image1.png', 'image2.png', etc., and 'image1.txt', 'image2.txt', etc.
# import os

# for filename in uploaded.keys():
#     if filename.endswith('.png'):
#         shutil.move(filename, 'data/images/' + filename)
#     elif filename.endswith('.txt'):
#         shutil.move(filename, 'data/labels/' + filename)

# # Create YOLO configuration and data files
# cfg_content = """
# [net]
# # Testing
# batch=1
# subdivisions=1
# width=416
# height=416
# channels=3
# momentum=0.9
# decay=0.0005
# learning_rate=0.001

# # [convolutional]
# # [maxpool]
# # [connected]
# # [yolo]
# # Modify these sections based on your dataset and desired architecture
# """

# with open('cfg/yolov4-custom.cfg', 'w') as f:
#     f.write(cfg_content)

# # Create obj.data file
# data_content = """
# classes = 2
# train  = data/train.txt
# valid  = data/test.txt
# names  = data/obj.names
# backup = backup/
# """

# with open('data/obj.data', 'w') as f:
#     f.write(data_content)

# # Create obj.names file
# names_content = """
# class1
# class2
# """

# with open('data/obj.names', 'w') as f:
#     f.write(names_content)

# # Create train.txt and test.txt
# def create_txt_file(image_dir, output_file):
#     with open(output_file, 'w') as f:
#         for filename in os.listdir(image_dir):
#             if filename.endswith('.png'):
#                 f.write(os.path.join('data/images', filename) + '\n')

# create_txt_file('data/images', 'data/train.txt')
# create_txt_file('data/images', 'data/test.txt')

# # Download pre-trained weights (optional, for faster convergence)
# !wget https://pjreddie.com/media/files/yolov4.weights -O yolov4.weights

# # Train the model
# !./darknet detector train data/obj.data cfg/yolov4-custom.cfg yolov4.weights

