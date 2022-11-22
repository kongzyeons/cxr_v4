from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2
import torch
import numpy as np
import random

# from utils.datasets import LoadStreams, LoadImages, letterbox
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.torch_utils import select_device, time_sync

from routes.set_onnx.detector_utils import scale_coords




def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2




def preprocess_image(cv2_img, in_size=(640, 640)):
    """preprocesses cv2 image and returns a norm np.ndarray
        cv2_img = cv2 image
        in_size: in_width, in_height
    """
    resized = pad_resize_image(cv2_img, in_size)
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in /= 255.0
    return img_in

def preprocess_image_cls(cv2_img, in_size=(640, 640)):
    """preprocesses cv2 image and returns a norm np.ndarray
        cv2_img = cv2 image
        in_size: in_width, in_height
    """
    resized = pad_resize_image(cv2_img, in_size)
    img_in = resized.astype(np.float32)  # HWC -> CHW
    img_in /= 255.0
    return img_in


def pad_resize_image(cv2_img, new_size=(640, 480), color=(125, 125, 125)) -> np.ndarray:
    """
    resize and pad image with color if necessary, maintaining orig scale
    args:
        cv2_img: numpy.ndarray = cv2 image
        new_size: tuple(int, int) = (width, height)
        color: tuple(int, int, int) = (B, G, R)
    """
    in_h, in_w = cv2_img.shape[:2]
    new_w, new_h = new_size
    # rescale down
    scale = min(new_w / in_w, new_h / in_h)
    # get new sacled widths and heights
    scale_new_w, scale_new_h = int(in_w * scale), int(in_h * scale)
    resized_img = cv2.resize(cv2_img, (scale_new_w, scale_new_h))
    # calculate deltas for padding
    d_w = max(new_w - scale_new_w, 0)
    d_h = max(new_h - scale_new_h, 0)
    # center image with padding on top/bottom or left/right
    top, bottom = d_h // 2, d_h - (d_h // 2)
    left, right = d_w // 2, d_w - (d_w // 2)
    pad_resized_img = cv2.copyMakeBorder(resized_img,
                                         top, bottom, left, right,
                                         cv2.BORDER_CONSTANT,
                                         value=color)
    return pad_resized_img


def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    # img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
#     draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
#     draw.rectangle(box, width=line_thickness, outline="red")
    draw.rectangle(box, width=line_thickness, outline=color)
    if label:
        fontsize = max(round(max(img.size) / 40), 12)
        font = ImageFont.truetype("routes/set_onnx/font/arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
#         draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill='red')
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=color)
        draw.text((box[0], box[1] - txt_height + 1), label, font=font)
        draw.text(((box[0], box[1] - txt_height + 1)), label,fill='white', font = font)
    return np.asarray(img)

def draw_image(frame, pred, names, img, colors):
    width, height = frame.size
    newsize = (height, width)

    result = []
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(newsize)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], newsize).round()

            # Write results
            for *xyxy, conf, cls in det:
                score = float('%.8f' % (conf)) *100
                result.append({
                    "class_name": names[int(cls)],
                    "confidence": score,
                    "position": {
                        "xmin": int(xyxy[0]),
                        "ymin": int(xyxy[1]),
                        "xmax": int(xyxy[2]),
                        "ymax": int(xyxy[3])
                    }
                })
                label = f'{names[int(cls)]} {score:.2f}%'
                plot_one_box_PIL(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=5)
    return frame, result

