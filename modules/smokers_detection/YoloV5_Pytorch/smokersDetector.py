import argparse

import torch.backends.cudnn as cudnn

import argparse

from modules.smokers_detection.YoloV5_Pytorch.models.experimental import *
from modules.smokers_detection.YoloV5_Pytorch.utils.datasets import *
from modules.smokers_detection.YoloV5_Pytorch.utils.utils import *
import numpy as np
import cv2
import cv2
import torch
from modules.object_tracking.objectTracker import p1p2Toxywh

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def YoloV5Pytorch_smokersInferenceOnFrame(frame, model, half,  image_size, score_threshold, iou_threshold, device, classes,
           agnostic_nms, augment_inference):

    frame = letterbox(frame, new_shape=image_size)[0]
    # frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    frame = frame.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

    # Run inference
    img = torch.from_numpy(frame).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=augment_inference)[0]

    # Apply NMS
    pred = non_max_suppression(pred, score_threshold, iou_threshold, classes=classes, agnostic=agnostic_nms)

    detections = list()
    confidences = list()
    labels = list()


    if pred[0] is not None:
        pred = pred[0].cpu().detach().numpy()
        print("0000000 ", pred)

        for xmin, ymin, xmax, ymax, confidence, class_id in pred:
            detections.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            confidences.append(confidence)
            labels.append(int(class_id))

    t2 = torch_utils.time_synchronized()

    print("Inference time = {0} msec".format(int((t2-t1)*1000)))

    return detections, confidences, labels


def YoloV5Pytorch_smokersDetectorOnFrame(frame,
                                      tracker,
                                        model,
                                        half,
                                        image_size,
                                        score_threshold,
                                        iou_threshold,
                                        device,
                                        classes,
                                        names_file,
                                        agnostic_nms,
                                        augment_inference,
                                        frame_nb,
                                        skip_frames=1):

    hold_detections = None

    if frame is None:
        print("Frame is None")
        return

    if frame_nb % skip_frames == 0:

        detections_img, confidences_img, labels_img = YoloV5Pytorch_smokersInferenceOnFrame(frame, model,  half, image_size, score_threshold, iou_threshold, device, classes,
                                                               agnostic_nms, augment_inference)

        detections = torch.from_numpy(np.asarray(detections_img, dtype=np.float16)).cuda(0)
        confidences = torch.from_numpy(np.asarray(confidences_img, dtype=np.float16)).cuda(0)
        class_ids = torch.from_numpy(np.asarray(labels_img, dtype=np.float16)).cuda(0)


        if detections is not None and tracker is not None and len(detections) > 0:
            #cv2.imwrite("temp.jpg", frame)
            boxs = p1p2Toxywh(detections)
            print("#####################", frame.shape, boxs)

            detections = tracker.update(boxs.float(), confidences, frame, class_ids)

        hold_detections = detections

    if hold_detections is not None:
        if tracker is None:
            hold_detections = [[x1, y1, x2, y2, score, label] for [x1, y1, x2, y2], score, label
                               in zip(detections_img, confidences_img, labels_img)]

            # hold_detections = torch.from_numpy(np.asarray(hold_detections, dtype=np.float16)).cuda(0)

    if hold_detections is None or len(hold_detections) == 0:
        hold_detections_dict = {}
    else:
        hold_detections_dict = toDict(hold_detections, tracker, names_file)

    #print("hold_detections = ", hold_detections, hold_detections_dict)

    return hold_detections_dict

def loadYoloV5Model(weights, device, image_size):
    # Initialize
    device = torch_utils.select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    image_size = check_img_size(image_size, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    return model, half, image_size, device

def toDict(tracking_list, tracker, names_file):
    tracking_dict = {}
    if tracker is None:
        n = 1
        for x1, y1, x2, y2, score, label in tracking_list:
            subDict = {}
            classname = load_names(names_file)[label]
            subDict["bbox"] = [x1, y1, x2, y2]
            subDict["class_id"] = label
            subDict["class_name"] = classname
            subDict["centroid"] = [int(x1) + int((x2 - x1) / 2), int(y1) + int((y2 - y1) / 2)]
            subDict["score"] = score

            tracking_dict[str(classname + "_" + str(n))] = subDict

            n += 1

    else:
        for x1, y1, x2, y2, id, class_id in tracking_list:
            subDict = {}
            subDict["bbox"] = [x1, y1, x2, y2]
            subDict["class_id"] = class_id
            subDict["class_name"] = load_names(names_file)[class_id]
            subDict["centroid"] = [int(x1) + int((x2-x1)/2), int(y1) + int((y2-y1)/2)]

            tracking_dict[id] = subDict

    return tracking_dict

def load_names(PRED_NAMES):
    names = {}
    with open(PRED_NAMES) as f:
        for id_, name in enumerate(f):
            names[id_] = name.split('\n')[0]
    return names

"""if __name__ == '__main__':

    weights = "/media/DATA/2.Research/02.COVID-19/03_YoloV5Trainer/yolov5/weights/maskdetection.pt"
    source = "/media/DATA/2.Research/02.COVID-19/03_YoloV5Trainer/yolov5/1.mp4"
    output_folder = "/home/sitou/Images/output"
    image_size = 704
    score_threshold = 0.3
    iou_threshold = 0.5
    device_id = '0'
    save_txt = False
    classes = None
    agnostic_nms = False
    augment_inference = False
    names_file = "/media/DATA/1.Projets/APACHE/data/mask_dataset.names"

    cap = cv2.VideoCapture(source)
    model, half, image_size, device = loadYoloV5Model(weights, device_id, image_size)

    while True:
        _, frame = cap.read()
        frame = frame.transpose(2,0,1)
        print("=====", frame.shape)
        detections_dict = YoloV5Pytorch_maskDetectorOnFrame(frame, model, half, image_size, score_threshold, iou_threshold, device, classes, names_file,agnostic_nms, augment_inference)"""

