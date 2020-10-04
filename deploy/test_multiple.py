# Author: Baker
# Date:   Friday 2 October 2020
# Description - This script reads the feature embeddings from a set of samples taken from a single lecture
#               It will to run on CPU use general python test_multiple.py --gpu -1


import face_model
import engagement_model

import argparse
import cv2
import sys
import numpy as np
import os
import json
from csv import writer
from datetime import datetime

# user arguments
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='../models/gamodel-r50/model,0 ', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id (-1 to run on CPU)')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

# load model
model = face_model.FaceModel(args)
dirListing = os.listdir('lecture_samples') # get the number of sample photos
features = np.zeros((len(dirListing), 512))

# calculate embeddings for each sample
i = 0
for filename in os.listdir('lecture_samples'):
    if filename.endswith(".jpg"): # there should probably be someway to probe the file type
        file_path = os.path.join('lecture_samples', filename)
        img = cv2.imread(file_path)
        img = model.get_input(img)
        f1 = model.get_feature(img)
        features[i, :] = f1
        i += 1

# load profile embeddings
with open('student_database.json') as f:
    data = json.load(f)

# compare samples to profile face embeddings
attendance = np.zeros(len(sorted(data)))
for i in range(len(sorted(data))):
    for j in range(np.shape(features)[0] - 1):
        f1 = features[j,:] # sample features
        f2 = data[sorted(data)[i]] # profile features
        dist = np.sum(np.square(f1-f2))
        if dist < args.threshold:
            attendance[i] = 1

# write attendance data to a csv file s]
attendance = attendance.astype('str')
attendance = np.insert(attendance, 0, str(datetime.date(datetime.now())))
with open('attendance_record.csv', 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(attendance)

sys.exit(0)