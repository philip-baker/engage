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
parser.add_argument('--gpu', default=-1, type=int, help='gpu id (-1 to run on CPU)')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--code', default='polsci100', type=str, help='The course code of the lecture')
args = parser.parse_args()

# load model
model = face_model.FaceModel(args)
dirListing = os.listdir('lecture_samples') # get the number of sample photos
features = np.zeros((len(dirListing), 512))

# calculate embeddings for each sample picture
engage_model = engagement_model.EngageModel(args)
features = engagement_model.get_embeddings(model,'lecture_samples')

# load student profile embeddings
with open('polsci100_student_database.json') as f:
    data = json.load(f)

# compare samples to profile face embeddings
attendance = engagement_model.compare_embeddings(engage_model, features, data)

# write attendance data to a csv file s - this can be complicated to probe the csv file
engagement_model.record_attendance(engage_model, attendance)

sys.exit(0)

