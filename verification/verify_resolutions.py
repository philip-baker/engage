"""Module documentation for visualisations.py
   This script processes the data from the NASA class, iterating over several face resolutions in order to get an attendance record for each individual 
   in each photo of the class at different resolutions. The results are saved to a csv file.
   
   NOTE: the graph in the report is using 37 values of resolution, in this code we have changed it to only 10 to lessen computational time - it will still take up to an hour.
   Uncomment line 123 and comment line 122 to change this back to 37 values.
"""

import argparse
import glob
import json
import sys
import os
import numpy as np
import cv2
from io import BytesIO
import torch
import csv
import itertools
import PIL.Image
from torchvision import transforms

from model.utils import get_model # tinyface
import face_model # arcface
from engagement_model import EngageModel
from functions import get_detections

os.chdir("..")
sys.path.append(os.getcwd() + '/helper')
sys.path.append(os.getcwd() + '/helper/tinyfaces')

__author__ = "Philip Baker & Keith Spencer-Edgar"
__date__ = "25-10-2020"

# Tiny Face Detector
# threshold for tinyfaces

# where to save results - change as desired
csv_save_path = "verification/verify_resolutions_csv.csv"

# change class if desired - this code was written for the NASA data
this_class = 'NASA'

# set cuda = True to run on the model on GPU
cuda = False

# tinyface user arguments
class args_eval():
    def __init__(self):
        self.nms_thresh = 0.3
        self.prob_thresh = 0.03
        self.checkpoint = "models/tinyfaces/checkpoint_50.pth"
        self.template_file = "helper/tinyfaces/data/templates.json"
        self.threshold_score = 0


args_tinyface = args_eval()
threshold_score = 0  # threshold tinyface score for feeding face into ArcFace - saves computation time. It is advised
# to keep this no higher than 0 to avoid filtering out true positives.

# getting templates
templates = json.load(open(args_tinyface.template_file))
json.dump(templates, open(args_tinyface.template_file, "w"))
templates = np.round_(np.array(templates), decimals=8)
num_templates = templates.shape[0]

# getting transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

# specify whether to run on CPU or GPU
if cuda is True:
    device = torch.device('gpu')
else:
    device = torch.device('cpu')

# get model
model_tinyfaces = get_model(args_tinyface.checkpoint, num_templates=num_templates)

rf = {
    'size': [859, 859],
    'stride': [8, 8],
    'offset': [-1, -1]
}

# ArcFace Model
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id (-1 to run on CPU)')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--code', default='my_course', type=str, help='The course code of the lecture')
args_arcface = parser.parse_args()

# load ArcFace model
model_arcface = face_model.FaceModel(args_arcface)

# Processing
class_images = list(glob.glob('verification/verification_images/class_images/' + this_class + '/*.jpg'))
attendance_sheet = list()
for class_image in class_images:
    # find detections with tinyface on downscaled photo
    img = PIL.Image.open(class_image)
    basewidth = 800
    scale_ = (basewidth / float(img.size[0]))
    if float(img.size[0]) > basewidth:
        hsize = int((float(img.size[1]) * float(scale_)))
        img_downscaled = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS).convert('RGB')
    else:
        scale_ = 1
        img_downscaled = img.convert('RGB')

    img_tensor = transforms.functional.to_tensor(img_downscaled)
    dets = get_detections(model_tinyfaces, img_tensor, templates, rf, val_transforms,
                          prob_thresh=args_tinyface.prob_thresh,
                          nms_thresh=args_tinyface.nms_thresh, device=device)

    # for each quality
    quality = list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # use this for 10 values (much quicker)
    # quality = list(np.arange(start=0.1, stop=1.025, step=0.025)) # use this for 37 intervals from 0.1 to 1
    # downscaling factor
    img = PIL.Image.open(class_image)
    for qual in quality:
        class_faces = list()
        class_scores = list()
        # resize image according to quality
        basewidth = int(qual * float(img.size[0]))
        hsize = int((float(img.size[1]) * float(qual)))
        this_img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS).convert('RGB')
        # get all detections for this image quality
        for i in range(len(dets)):  # for each detection
            if dets[i][4] > args_tinyface.threshold_score:  # if the tinyfaces score is good
                bbox = dets[i][0:4]
                new_bbox = bbox * (qual / scale_)
                face_width = new_bbox[2] - new_bbox[0]
                this_face = np.array(this_img.crop(new_bbox))  # get cropped face
                class_faces.append(list([this_face[:, :, ::-1].copy(), face_width]))  # append the cropped face
                # class_scores.append(dets[i][4])  # append the score - not used at the moment

        # calculate arcface embeddings for each sample picture
        detection_features, face_widths = EngageModel.get_embeddings(model_arcface, class_faces)
        if len(face_widths) > 0:
            average_face_width = sum(face_widths) / len(face_widths)
        else:
            average_face_width = None
            # load student profile embeddings for the class in progress
        # data = EngageModel.get_profiles(args)

        names = list()
        features = list()
        for filename in os.listdir('verification/verification_images/class_profiles/' + this_class + '/'):
            if filename.endswith(".jpg"):
                file_path = os.path.join('verification/verification_images/class_profiles/' + this_class + '/', filename)
                ima = cv2.imread(file_path)
                ima = model_arcface.get_input(ima)
                f1 = model_arcface.get_feature(ima)
                name = os.path.splitext(filename)[0]
                names.append(name)
                features.append(f1)
        data = list([names, features])

        # compare samples to profile face embeddings, produce roll
        attendance_sheet.append(EngageModel.class_list(model_arcface, detection_features, data,
                                                       this_class, class_image, qual, average_face_width))

        # calculate FP, FN, sensitivity etc
        print(class_image + " (" + str(class_images.index(class_image) + 1) + "/" + str(len(class_images)) + ")" +
              " at quality " + str(qual) + " didn't fail!")

results = list(itertools.chain.from_iterable(attendance_sheet))

with open(csv_save_path, 'w', newline='') as my_file:
    wr = csv.writer(my_file)
    wr.writerows(results)
