from ultralytics import YOLO
import cv2

# model = YOLO('YOLO Model\\Model\\Trained Models\\test_0\\best.pt')

model = YOLO('yolov8n.pt')

model.export(format='openvino')

ov_model = YOLO('yolov8n_openvine_model/')

results_ov = ov_model('https://ultralytics.com/images/bus.jpg')

results_original = model('https://ultralytics.com/images/bus.jpg')

cv2.imshow('results_ov', results_ov)
cv2.imshow('results_original', results_original)
cv2.waitKey(10000)