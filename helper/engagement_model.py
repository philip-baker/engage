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

    def class_list(self, detection_features, data, this_class_name, class_photo_name, quality):
        """
        Parameters:
        ----------
            data: upis and embeddings for students in a class
            detection_features: features of sample photos from tiny fcaes

        Returns:
        -------
            a list of the attendance for each student, as well as the class they belong to
        """
        roll = list()
        for i in range(len(data[0])):  # for each person supposed to be in the class
            name = data[0][i]
            match = False
            j = 0
            # go through all detections and compare to this person
            while not match and j < np.shape(detection_features)[0]:
                dist = np.sum(np.square(detection_features[j, :] - data[1][i][:]))
                if dist < self.args.threshold:
                    match = True
                j += 1

            if match:
                roll.append(list([name, 1, this_class_name,
                                  os.path.basename(os.path.normpath(class_photo_name)), quality]))
            else:
                roll.append(list([name, 0, this_class_name,
                                  os.path.basename(os.path.normpath(class_photo_name)), quality]))

        return roll

    # calculate embeddings for each sample
    def get_embeddings(model, class_faces):
        """ This function iterates through a folder of sample student images created by the Tiny Face Detector.
        Inputs:
        model - InsightFace Recognition model (ArcFace) defined in face_model.py
        folder - The file path to the folder containing the sample images stored as a string
        Outputs:
        features - Feature embeddings for each sample image a 512 entry numpy array of floating point values
        """

        feat_len = len(class_faces)
        features = np.zeros((feat_len, 512))
        i = 0
        for img in class_faces:

            try:
                img = model.get_input(img)
                f1 = model.get_feature(img)
                features[i, :] = f1
                i += 1
            except Exception:
                print('ArcFace could not detect face')
                features = np.delete(features, i, 0)

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
            j = 0
            while count == 0 and j < np.shape(sample_features)[0]:
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
                j += 1
            # this section may need correcting if this script is being ran multiple times in one day
            if count == 0:
                name = data[i][0]
                c.execute("""INSERT OR IGNORE INTO attendance VALUES (:date, :upi, :course_code, :attendance)""",
                          {'date': date, 'upi': name, 'course_code': self.code, 'attendance': 0})
            conn.commit()
        conn.close()
