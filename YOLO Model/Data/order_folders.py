import os, shutil
labels = [entry.replace('.txt', '') for entry in os.listdir('YOLO Model\\Data\\Labeled Images') if entry.endswith('.txt')]
images = [entry.replace('.png', '') for entry in os.listdir('YOLO Model\\Data\\Images') if entry.endswith('.png')]
pros_images = [entry.replace('.png', '') for entry in os.listdir('YOLO Model\\Data\\Processed Images') if entry.endswith('.png')]



move_files = [label for label in labels if label in images and label not in pros_images]

print(len(move_files))

print(len(labels)-1)
print(len(pros_images))

# for file in move_files:
#     shutil.move(f'YOLO Model\\Data\\Images\\{file}.png', f'YOLO Model\\Data\\Processed Images\\{file}.png')

print([file for file in pros_images if file.replace('.png', '') not in [label.replace('.txt', '') for label in labels]])
print(len([file for file in images if file.replace('.png', '') not in [label.replace('.txt', '') for label in labels]]))
print(len(images))