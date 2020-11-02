import sys
import os
os.chdir('..')
sys.path.append(os.getcwd() +'/helper')
import face_model
import argparse
import cv2
import sys
import numpy as np
import os
import json
import csv
import sqlite3
import csv
from create_database import Student, create_db, add_course

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


# create the database
# if os.path.isfile(db):
create_db()
add_course("my_course")

with open("students.csv", 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        try:
            student = Student(row[0], row[1], row[2], float(row[3]))
            Student.add_student(student)
            Student.add_to_class(student, "my_course")
        except:
            print("Error reading csv file, probably a blank row")


# student profile embeddings
conn = sqlite3.connect('engage.db')
c = conn.cursor()

# iterate through the profiles and calculate embeddings for each one
for filename in os.listdir('student_profiles'):
    if filename.endswith(".jpg"):
        file_path = os.path.join('student_profiles', filename)
        img = cv2.imread(file_path)
        img = model.get_input(img)
        f1 = model.get_feature(img)
        f1 = json.dumps(list(map(str, list(f1))))
        name = os.path.splitext(filename)[0]

        c.execute("""INSERT OR REPLACE INTO features VALUES (:upi, :features)""",
                  {'upi': name, 'features': f1})
        conn.commit()

conn.close()
sys.exit(0)
