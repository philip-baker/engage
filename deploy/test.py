# Author: Baker
# Date:   Friday 2 October 2020
# Description - This script reads the feature embeddings from a set of samples taken from a single lecture
#               It will to run on CPUe use general python test_multiple.py --gpu -1


import face_model
from engagement_model import EngageModel

import argparse
import cv2
import sys
import numpy as np
import os
import json
from csv import writer
from datetime import datetime
import sqlite3



# user arguments
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id (-1 to run on CPU)')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--code', default='gened', type=str, help='The course code of the lecture')
args = parser.parse_args()

# load model
model = face_model.FaceModel(args)
dirListing = os.listdir('tinyfaces/data/output_faces') # get the number of sample photos
features = np.zeros((len(dirListing), 512))

# calculate embeddings for each sample picture
engage_model = EngageModel(args)
sample_features = EngageModel.get_embeddings(model,'tinyfaces/data/output_faces')

# load student profile embeddings for the class in progress
data = EngageModel.get_profiles(args)

# compare samples to profile face embeddings
EngageModel.compare_embeddings(engage_model, sample_features, data)

sys.exit(0)

