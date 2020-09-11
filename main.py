import sys
import os
import cv2
from modules.smokers_detection.YoloV5_Pytorch.smokersDetector import loadYoloV5Model

from smokersdetection.inference import *

ARCHITECTURE_L1 ="YoloV5"
FRAMEWORK_L1 ="Pytorch"
tracker = None
YOLOV5_PATH = "./modules/smokers_detection/YoloV5_Pytorch"
MAIN_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
YOLO_CLASSES = os.path.join(MAIN_DIRECTORY,"data",'smokers.names')
sys.path.insert(0, YOLOV5_PATH)
WEIGHTS_PATH = './weights/yolov5x_smokers_detection.pt'

source = "./2.mp4"
cap = cv2.VideoCapture(source)
frame_nb = 0
image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

model, half, image_size, device = loadYoloV5Model(weights= WEIGHTS_PATH, device='0', image_size=image_width)

while True:
    ret, frame = cap.read()
    frame_nb += 1

    if frame is not None:
        detection_dict = smokersDetectorOnFrame(frame, tracker=tracker, model=model, half=half,
                                                image_size=image_size, score_threshold=0.2,
                                                iou_threshold=0.5, device=device, classes=None,
                                                names_file=YOLO_CLASSES, agnostic_nms=True,
                                                augment_inference=False, frame_nb=frame_nb,
                                                skip_frames=1, architecture=ARCHITECTURE_L1,
                                                framework=FRAMEWORK_L1)
        frame_detect = drawBoxesLabels(frame, detection_dict, tracker=tracker)
        cv2.imshow("Test", frame_detect)
        cv2.waitKey(25)
        print("detection_dict = ", detection_dict)

    else:
        print("End of stream !!")
        break
