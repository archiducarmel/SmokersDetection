import cv2
import multiprocessing
from multiprocessing import Process
from threading import Thread
import random
import os
import glob
import pings
import time
import datetime
import platform
import numpy as np
import tensorflow as tf
import configparser
import serial
from flask import Flask, render_template, request, Response
#from HW import *
from modules.smokers_detection.YoloV5_Pytorch.smokersDetector import YoloV5Pytorch_smokersDetectorOnFrame


#===== Below imports are specific to mask detection
import torch
from torch.autograd import Variable
from datetime import datetime

def smokersDetectorOnFrame(frame, tracker, model, half, image_size, score_threshold, iou_threshold, device, classes, names_file,agnostic_nms, augment_inference , frame_nb, skip_frames,architecture, framework):
    if architecture == "YoloV5" and framework == "Pytorch":
        return YoloV5Pytorch_smokersDetectorOnFrame(frame, tracker, model, half, image_size, score_threshold, iou_threshold, device, classes, names_file,agnostic_nms, augment_inference, frame_nb, skip_frames)

def drawBoxesLabels(frame, detections_dict, tracker):
    thickness = int((frame.shape[1])/350)
    fontSize = int((frame.shape[1])/200)
    fontType = cv2.FONT_HERSHEY_TRIPLEX
    #lineType = cv2.LINE_AA

    for key, values in detections_dict.items():
        score = None
        id = None

        bbox = values["bbox"]
        classname = values["class_name"]
        centroid = values["centroid"]
        try:
            direction = values["direction"]
        except KeyError:
            direction = ("?", "?")


        if tracker is None:
            score = values["score"]

            if score > 0 and score < 0.3:
                color = (183, 45, 45) #RED
            elif score >=0.3 and score < 0.6:
                color = (53, 153, 251) #ORANGE
            elif score >= 0.6 and score < 0.8:
                color = (0, 213, 255) #YELLOW
            elif score >=0.8:
                color = (102, 204, 0) #GREEN

            label = str(classname + " | " + str(int(score*100)) + "%")
        else:
            id = key

            color = (76, 0, 153)

            label = str(classname + "_" + str(id))

        font_w, font_h = cv2.getTextSize(label, fontType, fontSize/10, 2)
        cv2.rectangle(frame, (bbox[0]-1, bbox[1] - int(1.6 * font_w[1])), (bbox[0] + font_w[0], bbox[1]), color, -1)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        cv2.circle(frame, tuple(centroid), 4, color, -1)
        cv2.putText(frame,
                    label,
                    (bbox[0], max(bbox[1] - 2, font_h)), fontType, fontSize/10,
                    (255, 255, 255), 1)

    return frame
