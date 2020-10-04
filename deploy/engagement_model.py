import face_model
import argparse
import cv2
import sys
import numpy as np
import os
import json
import csv
from datetime import datetime
from csv import writer
import csv


class EngageModel:
    def __init__(self, args):
        self.args = args
        self.code = args.code
        self.threshold = args.threshold


# calculate embeddings for each sample
def get_embeddings(model, folder):
    dir_listing = os.listdir(folder)
    features = np.zeros((len(dir_listing), 512))
    i = 0
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            file_path = os.path.join(folder, filename)
            img = cv2.imread(file_path)
            img = model.get_input(img)
            f1 = model.get_feature(img)
            features[i, :] = f1
            i += 1
    return features


# compare samples to profile face embeddings
def compare_embeddings(self, features, data):
    attendance = np.zeros(len(sorted(data)))
    for i in range(len(sorted(data))):
        for j in range(np.shape(features)[0] - 1):
            f1 = features[j, :]  # sample features
            f2 = data[sorted(data)[i]]  # profile features
            dist = np.sum(np.square(f1 - f2))
            if dist < self.args.threshold:
                attendance[i] = 1
    return attendance


# writes a line to a csv file storing attendance for each student
def record_attendance(self, attendance):
    attendance = attendance.astype('str')
    attendance = np.insert(attendance, 0, str(datetime.date(datetime.now())))
    filename = self.code + '_attendance_record.csv'
    with open(filename, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(attendance)
