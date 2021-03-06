"""Documentation for verify_results.py
   This script processes class data in order to calculate accuracy, sensitivity etc. on the overall class. This should match results given in our report.
   The script prints these values out to screen.
"""

import os
import sys
import argparse
import glob
import json
import numpy as np
import cv2
from io import BytesIO
import torch
import csv
import itertools
import PIL.Image
from torchvision import transforms
import json

import face_model
from engagement_model import EngageModel
from functions import get_detections
from model.utils import get_model

currDir = os.getcwd()
os.chdir('..')
sys.path.append(os.getcwd() + '/helper')
sys.path.append(os.getcwd() + '/helper/tinyfaces')
sys.path.append(os.getcwd() + '/models/model-r100-ii')
os.chdir(currDir)

__author__ = "Philip Baker & Keith Spencer-Edgar"
__date__  = "25-10-2020"

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id (-1 to run on CPU)')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--code', default='my_course', type=str, help='The course code of the lecture')
args_arcface = parser.parse_args()

# load ArcFace model
model_arcface = face_model.FaceModel(args_arcface)


# load the Tiny Face Detector
class args_eval():
    def __init__(self):
        self.nms_thresh = 0.3
        self.prob_thresh = 0.03
        self.checkpoint = "../models/tinyfaces/checkpoint_50.pth"
        self.template_file = "../helper/tinyfaces/data/templates.json"
        self.threshold_score = 0


args_tinyface = args_eval()
threshold_score = 0

# get templates
templates = json.load(open(args_tinyface.template_file))
json.dump(templates, open(args_tinyface.template_file, "w"))
templates = np.round_(np.array(templates), decimals=8)
num_templates = templates.shape[0]

# get transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

if args_arcface.gpu > 0:
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

# process test cases
classes = ['NZP']

# loop through class folder
output = list()
for this_class in classes:

    class_images = list(glob.glob('verification_images/class_images/' + this_class + '/*.jpg'))
    attendance_sheet = list()
    for class_image in class_images:
        # find detections with tinyface
        img = PIL.Image.open(class_image)
        basewidth = 750
        qual = 100
        img_tensor = transforms.functional.to_tensor(img)
        dets = get_detections(model_tinyfaces, img_tensor, templates, rf, val_transforms,
                              prob_thresh=args_tinyface.prob_thresh,
                              nms_thresh=args_tinyface.nms_thresh, device=device)

        class_faces = list()
        class_scores = list()
        for i in range(len(dets)):  # for each detection
            if dets[i][4] > args_tinyface.threshold_score:  # if the tinyfaces score is good
                this_face = np.array(img.crop(dets[i][0:4]))  # get cropped face
                class_faces.append([this_face[:, :, ::-1].copy(), 100])  # append the cropped face
                class_scores.append(dets[i][4])  # append the scor
            # calculate arcface embeddings for each sample picture
            detection_features, face_widths = EngageModel.get_embeddings(model_arcface, class_faces)

        # load student profile embeddings for the class in progress
        names = list()
        features = list()
        for filename in os.listdir('verification_images/class_profiles/' + this_class + '/'):
            if filename.endswith(".jpg"):
                file_path = os.path.join('verification_images/class_profiles/' + this_class + '/', filename)
                img = cv2.imread(file_path)
                img = model_arcface.get_input(img)
                f1 = model_arcface.get_feature(img)
                name = os.path.splitext(filename)[0]
                names.append(name)
                features.append(f1)
        data = list([names, features])

        # compare samples to profile face embeddings, produce roll
        attendance_sheet.append(EngageModel.class_list(model_arcface, detection_features, data,
                                                       this_class, class_image, qual, 100))

        # calculate FP, FN, sensitivity etc
        print(class_image + str(qual) + "didn't fail!")

    output.append(list([list(itertools.chain.from_iterable(attendance_sheet))]))

with open("output3.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for class_i in output:
        for class_photo in class_i:
            wr.writerows(class_photo)

with open("roll_call.json", "r") as read_file:
    data = json.load(read_file)

for i in classes:
    print('------------------------')
    print('PERFORMANCE METRICS FOR:')
    print(i)
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for j in range(len(output[0][0])):
        if output[0][0][2][2] == i:
            key = output[0][0][j][3]
            upi = output[0][0][j][0]
            status = output[0][0][j][1]
            roll = data.get(key)
            if upi in roll:
                if status == 1:
                    tp += 1
                else:
                    fn += 1
                    print("FALSE NEGATIVE")
                    print(upi)
                    print(status)
                    print(key)
            else:
                if status == 0:
                    tn += 1
                else:
                    fp += 1
                    print("FALSE POSITIVE")
                    print(upi)
                    print(status)
                    print(key)
    print(tp)
    print(fp)
    print(tn)
    print(fn)
    if (tp + tn + fp + fn) > 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print("Model Accuracy")
        print(accuracy)
    else:
        print("Division by zero - could not calculate accuracy")
    if (tp + fn) > 0:
        sensitivity = tp / (tp + fn)
        print("Model sensitivity / recall")
        print(sensitivity)
    else:
        print("Division by zero could not calculate sensitivity")

    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
        print("Model specificity")
        print(specificity)
    else:
        print("Division by zero could not calculate sensitivity")
