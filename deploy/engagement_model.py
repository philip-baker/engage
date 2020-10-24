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
import sqlite3
import ast


class EngageModel:
    def __init__(self, args):
        self.args = args
        self.code = args.code
        self.threshold = args.threshold

    # calculate embeddings for each sample
    def get_embeddings(model, folder):
        """ This function iterates through a folder of sample student images created by the Tiny Face Detector.
        Inputs:
        model - InsightFace Recognition model (ArcFace) defined in face_model.py
        folder - The file path to the folder containing the sample images stored as a string
        Outputs:
        features - Feature embeddings for each sample image a 512 entry numpy array of floating point values
        """

        dir_listing = os.listdir(folder)
        features = np.zeros((len(dir_listing), 512))
        i = 0
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                file_path = os.path.join(folder, filename)
                img = cv2.imread(file_path)
                # try:
                #     img = model.get_input(img)
                #     f1 = model.get_feature(img)
                #     features[i, :] = f1
                #     i += 1
                # #TODO fix this
                # except:
                #     print('ArcFace could not detect face')
                #     features = np.delete(features, i, 0)
                img = model.get_input(img)
                f1 = model.get_feature(img)
                features[i, :] = f1
                i += 1
        return features

    def get_profiles(self):
        """Retrieves the feature embeddings for students in a specific class
        """
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute(
            "SELECT * FROM features INNER JOIN %s ON %s.upi = features.upi" % (repr(self.code), repr(self.code)))
        data = c.fetchall()
        conn.commit()
        conn.close()

        return data

    # compare samples to profile face embeddings
    def compare_embeddings(self, sample_features, data):
        """Write attendance data for a single lecture to the database
        Line 2 of comment...
        And so on...
        """
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        date = str(datetime.date(datetime.now()))
        for i in range(len(sorted(data))):
            count = 0
            for j in range(np.shape(sample_features)[0] - 1):
                f1 = sample_features[j, :]  # sample features
                f2 = np.array(ast.literal_eval(data[i][1]))  # profile features
                f2 = f2.astype('float64')
                dist = np.sum(np.square(f1 - f2))
                name = data[i][0]
                if dist < self.args.threshold:
                    count = 1
                    c.execute(
                        """INSERT OR IGNORE INTO attendance VALUES (:date, :upi, :course_code, :attendance)""",
                        {'date': date, 'upi': name, 'course_code': self.code, 'attendance': 1})
            # this section may need correcting if this script is being ran multiple times in one day
            if count == 0:
                c.execute("""INSERT OR IGNORE INTO attendance VALUES (:date, :upi, :course_code, :attendance)""",
                          {'date': date, 'upi': name, 'course_code': self.code, 'attendance': 0})
            conn.commit()
        conn.close()
