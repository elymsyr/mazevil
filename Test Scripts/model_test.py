from tflite_detect_image import tflite_detect_images

# Set up variables for running user's model
PATH_TO_IMAGES='Data\\test'   # Path to test images folder
PATH_TO_MODEL='Model\\test_model_001\\detect.tflite'   # Path to .tflite model file
PATH_TO_LABELS='Model\\test_model_001\\labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold=0.5   # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
images_to_test = 5   # Number of images to run detection on

# Run inferencing function!
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)