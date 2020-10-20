# Primary Author: Spencer-Edgar
# Secondary: Baker
# Date:    October 2020
# Description - This script reads student data from csv files

import os
import glob
import csv
from create_database import Student, create_db, add_course

# create the database
# if os.path.isfile(db):
create_db()
# TODO fix line below should be example/*.csv
class_files = list(glob.glob('**/*.csv', recursive=True)) # needs to point to example_classes

for this_class in class_files:

    class_name = this_class.strip(".csv")
    class_name = class_name[16:]

    # add class to database
    add_course(class_name)
    # TODO ADD SOMETHING to deal with blank lines in CSV
    with open(this_class, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # read new line
            try:
                student_1 = Student(row[0], row[1], row[2], float(row[3]))
                Student.add_student(student_1)
                Student.add_to_class(student_1, class_name)
            except:
                print("Error reading csv file, probably a blank row")





