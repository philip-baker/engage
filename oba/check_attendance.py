# Date:   Friday 2 October 2020
# Description - This script reads the feature embeddings from a set of samples taken from a single lecture
#               It will to run on CPUe use general python test_multiple.py --gpu -1
import sys
import os
sys.path.append(os.getcwd() +'/helper')
sys.path.append(os.getcwd() +'/helper/tinyfaces')
import face_model
from engagement_model import EngageModel
from tinyfaces import tiny, args_eval

import argparse
import cv2
import sys
import numpy as np
import os
import json
from csv import writer
from datetime import datetime
import sqlite3
import cv2

# user arguments
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id (-1 to run on CPU)')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--code', default='my_course', type=str, help='The course code of the lecture')
args = parser.parse_args()


# use open csv to take an image of the audience

count = 0
for filename in os.listdir('student_profiles'):
    if filename.endswith(".jpg"):
        count = count + 1
if count == 0:
    cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
    ret,frame = cap.read() # return a single frame in variable `frame`
    while(True):
        cv2.imshow('img1',frame) #display the captured image
        if cv2.waitKey(1) & 0xFF == ord('s'): #save on pressing 's'
            cv2.imwrite('helper/tinyfaces/data/images/sample.jpg', frame)
        cv2.destroyAllWindows()
        break
    cap.release()

# run tiny faces
cwd = os.getcwd()
os.chdir(cwd + "/helper/tinyfaces")
tiny_args = args_eval()
tiny(tiny_args)
os.chdir(cwd)

# load model
model = face_model.FaceModel(args)
dirListing = os.listdir('helper/tinyfaces/data/output_faces') # get the number of sample photos
features = np.zeros((len(dirListing), 512))

# calculate embeddings for each sample picture
engage_model = EngageModel(args)
sample_features = EngageModel.get_embeddings(model,'helper/tinyfaces/data/output_faces')

# load student profile embeddings for the class in progress
data = EngageModel.get_profiles(args)

# compare samples to profile face embeddings
EngageModel.compare_embeddings(engage_model, sample_features, data)

sys.exit(0)

