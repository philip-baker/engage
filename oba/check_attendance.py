"""Documentation for build_system.py
   This script is part of the OBA (Out-of-the-Box-Attendance) it has been designed to be used on a daily basis for a given course.  As an input it will 
   look for an image (named sample.jpg), which if it cannot find it will prompt the user to take an image using any camera attached to the computer.  Then 
   the Tiny Face Detector 
   
   it looks for an image (named sample.jpg) as an input in the folder sample_images
   if it cannot find a picture, it will attempt to take a picture with whatever camera is connected to the computer. Then the Tiny Face Detector is 
   used to detect faces in the image, before using ArcFace to compare these images against the profile images for students in the class.  The results 
   (Present / Absent for each student) are then written to the engage.db 
   
   To run on CPU specifiy the gpu argument as follows check_attendance.py --gpu -1
"""

import sys
import os
currDir = os.getcwd()
os.chdir('..')
sys.path.append(os.getcwd() + '/helper')
sys.path.append(os.getcwd() + '/helper/tinyfaces')
sys.path.append(os.getcwd() +'/models/model-r100-ii')
os.chdir(currDir)
import argparse
import cv2
import numpy as np
import json
from csv import writer
from datetime import datetime
import sqlite3
from io import BytesIO
import torch
import PIL.Image
from torchvision import transforms


__author__ = "Philip Baker & Keith Spencer-Edgar"
__date__ = "25-10-2020"

import face_model
from engagement_model import EngageModel
from functions import get_detections
from model.utils import get_model

# User arguments
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id (-1 to run on CPU)')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--code', default='my_course', type=str, help='The course code of the lecture')
args = parser.parse_args()


# Check for a user supplied image 
count = 0
for filename in os.listdir('sample_images'):
    if filename.endswith(".jpg"):
        count = count + 1
 # Use open cv to take an image of the audience
if count == 0:
    cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
    ret, frame = cap.read() # return a single frame in variable `frame`
    while(True):
        cv2.imshow('img1',frame) # display the captured image
        if cv2.waitKey(0) & 0xFF == ord('q'): # save on pressing 's'
            cv2.imwrite('sample_images/sample.jpg', frame)
        cv2.destroyAllWindows()
        break
    cap.release()

class_image =  'sample_images/sample.jpg'

# Arguments for the Tiny Face Detector
class args_eval():
    def __init__(self):
        self.nms_thresh = 0.3
        self.prob_thresh = 0.03
        self.checkpoint = "../models/tinyfaces/checkpoint_50.pth"
        self.template_file = "../helper/tinyfaces/data/templates.json"
        self.threshold_score = 0


args_tinyface = args_eval()

# Get templates for Tiny Face Detector
templates = json.load(open(args_tinyface.template_file))
json.dump(templates, open(args_tinyface.template_file, "w"))
templates = np.round_(np.array(templates), decimals=8)
num_templates = templates.shape[0]

# Getting transforms for the Tiny Face Detector
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

# Specify whether to run on GPU or CPU based on user arguments 
if args.gpu > 0:
    device = torch.device('gpu')
else:
    device = torch.device('cpu')

# Load the Tiny Face Detector
model_tinyfaces = get_model(args_tinyface.checkpoint, num_templates=num_templates, gpu= (args.gpu > 0))

# Specifications for the Tiny Face Detector
rf = {
    'size': [859, 859],
    'stride': [8, 8],
    'offset': [-1, -1]
}

# Load the insightface ArcFace Face Recognition model
model_arcface = face_model.FaceModel(args)


# Downscale iamge 
img = PIL.Image.open(class_image)
basewidth = 800
scale_ = (basewidth / float(img.size[0]))
if float(img.size[0]) > basewidth:
    hsize = int((float(img.size[1]) * float(scale_)))
    img_downscaled = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS).convert('RGB')
else:
    scale_ = 1
    img_downscaled = img.convert('RGB')
# User Tiny Face Detector to get detections from downscaled image 
img_tensor = transforms.functional.to_tensor(img_downscaled)
dets = get_detections(model_tinyfaces, img_tensor, templates, rf, val_transforms,
                      prob_thresh=args_tinyface.prob_thresh,
                      nms_thresh=args_tinyface.nms_thresh, device=device)

img = PIL.Image.open(class_image)
class_faces = list()
class_scores = list()

# Resize image according to quality
basewidth = int(float(img.size[0]))
hsize = int(float(img.size[1]))
this_img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS).convert('RGB')

# Load student profile embeddings for the class in progress
data = EngageModel.get_profiles(args)

# Get all detections
for i in range(len(dets)):  
    if dets[i][4] > args_tinyface.threshold_score:  # Check detector confidence is above threshold
        bbox = dets[i][0:4]
        new_bbox = bbox * (1 / scale_)
        face_width = new_bbox[2] - new_bbox[0]
        this_face = np.array(this_img.crop(new_bbox))  # Get cropped face
        class_faces.append(list([this_face[:, :, ::-1].copy(), face_width]))  # Append the cropped face

# Calculate arcface embeddings for each sample picture
detection_features, face_widths = EngageModel.get_embeddings(model_arcface, class_faces)

# Compare samples to profile face embeddings
engage_model = EngageModel(args)
EngageModel.compare_embeddings(engage_model, detection_features, data)

sys.exit(0)


