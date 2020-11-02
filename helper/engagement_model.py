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

    def class_list(self, detection_features, data, this_class_name, class_photo_name, average_face_width):
        """
            iterate through detected faces, comparing their features to those enrolled in the class

        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ---------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox

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
                                  os.path.basename(os.path.normpath(class_photo_name)), average_face_width]))
            else:
                roll.append(list([name, 0, this_class_name,
                                  os.path.basename(os.path.normpath(class_photo_name)), average_face_width]))

        return roll

    # calculate embeddings for each sample
    def get_embeddings(model, class_faces):
        """
            iterate through detected faces and calculate feature embeddings for each face

        Parameters:
        ----------
            class_faces: numpy array, n x 5
                input bboxes
        Returns :
        ---------
            features face_widths : numpy array, n x 1
                start point of the bbox in target image
            face_widths : numpy array, n x 1
                end point of the bbox in target imag

        """

        feat_len = len(class_faces)
        features = np.zeros((feat_len, 512))
        face_widths = list()
        isface = False
        i = 0
        for img in class_faces:

            try:
                img2 = model.get_input(img[0])
                f1 = model.get_feature(img2)
                features[i, :] = f1
                i += 1
                isface = True
            except:
                print('ArcFace could not detect face')
                features = np.delete(features, i, 0)
                isface = False

            if isface:
                face_widths.append(img[1])

        return features, face_widths

    def get_profiles(self):
        """
            query feature embeddings for students in a specific class from the SQLite database
            
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
        """
            compare feature embeddings 

        Parameters:
        ----------
            sample_features : numpy array, n x 5
                input bboxes
            date : numpy array, n x 5
                input bboxes

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
                
            if count == 0:
                name = data[i][0]
                c.execute("""INSERT OR IGNORE INTO attendance VALUES (:date, :upi, :course_code, :attendance)""",
                          {'date': date, 'upi': name, 'course_code': self.code, 'attendance': 0})
            conn.commit()
        conn.close()
