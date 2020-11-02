# user arguments
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

os.chdir('..')
sys.path.append(os.getcwd() + '/models/tinyfaces')

import face_model
from engagement_model import EngageModel
from functions import get_detections

## TINYFACES
# threshold for tinyfaces
from model.utils import get_model


class args_eval():
    def __init__(self):
        self.nms_thresh = 0.3
        self.prob_thresh = 0.03
        self.checkpoint = "helper/tinyfaces/model/checkpoint_50.pth"
        self.template_file = "helper/tinyfaces/data/templates.json"
        self.threshold_score = 0


args_tinyface = args_eval()
threshold_score = 0

## getting templates
templates = json.load(open(args_tinyface.template_file))
json.dump(templates, open(args_tinyface.template_file, "w"))
templates = np.round_(np.array(templates), decimals=8)
num_templates = templates.shape[0]

## getting transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

if torch.cuda.is_available():
    # device = torch.device('cuda:0')
    device = torch.device('cpu')
else:
    device = torch.device('cpu')

## get model
model_tinyfaces = get_model(args_tinyface.checkpoint, num_templates=num_templates)

rf = {
    'size': [859, 859],
    'stride': [8, 8],
    'offset': [-1, -1]
}

## ARCFACE
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

## PROCESSING
# single file to represent our results
classes = ['NASA']
# TODO: replace with os code to find folders in directory

# loop through class folder
output = list()
for this_class in classes:

    class_images = list(glob.glob('verification_images/class_images/' + this_class + '/*.jpg'))
    attendance_sheet = list()
    for class_image in class_images:
        # find detections with tinyface on downscaled photo
        img = PIL.Image.open(class_image)
        basewidth = 750
        # if float(img.size[0]) > basewidth:
        #     wpercent = (basewidth / float(img.size[0]))
        #     hsize = int((float(img.size[1]) * float(wpercent)))
        #     img_downscaled = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS).convert('RGB')
        #     with BytesIO() as f: # make img_downscaled.format JPEG, as it turns to NoneType when resizing
        #         img_downscaled.save(f, format = 'JPEG')
        #         f.seek(0)
        #         img_downscaled = PIL.Image.open(f)
        #         img_tensor = transforms.functional.to_tensor(img_downscaled)
        #         dets = get_detections(model_tinyfaces, img_tensor, templates, rf, val_transforms,
        #                               prob_thresh=args_tinyface.prob_thresh,
        #                               nms_thresh=args_tinyface.nms_thresh, device=device)
        # else:
        #     img_downscaled = img.convert('RGB')
        #     img_tensor = transforms.functional.to_tensor(img_downscaled)
        #     dets = get_detections(model_tinyfaces, img_tensor, templates, rf, val_transforms,
        #                           prob_thresh=args_tinyface.prob_thresh,
        #                           nms_thresh=args_tinyface.nms_thresh, device=device)

        qual = 100
        img_tensor = transforms.functional.to_tensor(img)
        dets = get_detections(model_tinyfaces, img_tensor, templates, rf, val_transforms,
                              prob_thresh=args_tinyface.prob_thresh,
                              nms_thresh=args_tinyface.nms_thresh, device=device)
        #  for each quality
        # quality = list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        # for qual in quality:
        #     class_faces = list()
        #     class_scores = list()
        #     # resize image according to quality
        #     basewidth = int(qual * float(img.size[0]))
        #     hsize = int((float(img.size[1]) * float(qual)))
        #     this_img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS).convert('RGB')
        #
        #     # get all detections for this image quality
        #     for i in range(len(dets)):  # for each detection
        #         if dets[i][4] > args_tinyface.threshold_score:  # if the tinyfaces score is good
        #             this_face = np.array(this_img.crop(dets[i][0:4]))  # get cropped face
        #             class_faces.append(this_face[:, :, ::-1].copy())  # append the cropped face
        #             class_scores.append(dets[i][4])  # append the score
        class_faces = list()
        class_scores = list()
        for i in range(len(dets)):  # for each detection
            if dets[i][4] > args_tinyface.threshold_score:  # if the tinyfaces score is good
                this_face = np.array(img.crop(dets[i][0:4]))  # get cropped face
                class_faces.append(this_face[:, :, ::-1].copy())  # append the cropped face
                class_scores.append(dets[i][4])  # append the score
            # calculate arcface embeddings for each sample picture
            detection_features = EngageModel.get_embeddings(model_arcface, class_faces)

            # load student profile embeddings for the class in progress
            # data = EngageModel.get_profiles(args)

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
                                                       this_class, class_image, qual))

        # calculate FP, FN, sensitivity etc
        print(class_image + str(qual) + "didn't fail!")

    output.append(list([list(itertools.chain.from_iterable(attendance_sheet))]))

with open("output3.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for class_i in output:
        for class_photo in class_i:
            wr.writerows(class_photo)
