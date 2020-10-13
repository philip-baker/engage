# File: profile_embeddings Author: Bake Date: Started 11/27/1990 Description: This script reads the feature
# embeddings from a set of student profile images. This should be updatex to create a json wit h user defined name, Use coursd code parameter


import face_model
import argparse
import cv2
import sys
import numpy as np
import os
import json
import csv

# set arguments
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id (-1 to run on CPU)')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--code', default='SPT100', type=str, help='The course code of the lecture')
args = parser.parse_args()

# load model
model = face_model.FaceModel(args)
dirListing = os.listdir('student_profiles')  # get the number of sample photos

studentData = {}
# iterate through the profiles and calculate embeddings for each one
i = 0
for filename in os.listdir('student_profiles'):
    if filename.endswith(".jpg"):  # there should probably be someway to probe the file type
        file_path = os.path.join('student_profiles', filename)
        img = cv2.imread(file_path)
        img = model.get_input(img)
        f1 = model.get_feature(img)
        studentData[os.path.splitext(filename)[0]] = list(map(str,list(f1)))
        i += 1


with open(args.code + "_student_database.json", "w") as write_file:
    json.dump(studentData, write_file)  # default=lambda o: o.__dict__, sort_keys=True, indent=2

sys.exit(0)
