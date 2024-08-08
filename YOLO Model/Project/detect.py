import cv2
import matplotlib
import numpy as np
import torch

# YOLOv7 model utilities
from yolov7.models.yolo import Model
from yolov7.utils.torch_utils import select_device
from yolov7.utils.plots import plot_one_box


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)


def display(imgs, pred, names):
    colors = color_list()
    for i, (img, pred) in enumerate(zip(imgs, pred)):
        str = f"image {i + 1}/{len(pred)}: {img.shape[0]}x{img.shape[1]} "
        if pred is not None:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                str += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *box, conf, cls in pred:  # xyxy, confidence, class
                label = f"{names[int(cls)]} {conf:.2f}"
                plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
        # img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
    return img

def model_load(path_or_model='path/to/model.pt', autoshape=True):
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']
    hub_model = Model(model.yaml).to(next(model.parameters()).device)
    hub_model.load_state_dict(model.float().state_dict())
    hub_model.names = model.names
    if autoshape:
        hub_model = hub_model.autoshape()
    # device = select_device('0' if torch.cuda.is_available() else 'cpu')
    device = select_device('cpu')
    return hub_model.to(device), device