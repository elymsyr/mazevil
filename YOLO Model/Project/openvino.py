from ultralytics import YOLO
import cv2

model = YOLO('YOLO Model\\Model\\Trained Models\\test_0\\train\\weights\\best.pt')

model.export(format='openvino')

ov_model = YOLO('yolov8n_openvine_model/')

results_ov = ov_model('YOLO Model\\Data\\Test\\screenshot_411.png')

results_original = model('YOLO Model\\Data\\Test\\screenshot_411.png')
