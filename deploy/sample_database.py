import sqlite3
import argparse
from create_database import Student, create_db, EngageDB

# this file creates a sample data base with one class engsci and two students

parser = argparse.ArgumentParser(description='sample database')
parser.add_argument('--code', default='gened', type=str, help='The course code of the lecture')
args = parser.parse_args()

# create the database
create_db()

# create a class/course
EngageDB.add_course(args)

# create two example students
student_1 = Student('agun001', 'Alex', 'Gunasekera', 20)
student_2 = Student('bbra002', 'Brandon', 'Brandon', 21)
student_3 = Student('arun003', 'Alexander', 'Russell', 22)
student_4 = Student('cric004', 'Caleb', 'Rich', 23)
student_5 = Student('cmcm005', 'Connor', 'McMeel', 19)
student_6 = Student('iwan006', 'Ian', 'Wang', 18)
student_7 = Student('rbrk007', 'Richard', 'Brooks', 19)
student_8 = Student('wste008', 'William', 'Stewart', 20)

# add the students to the database
Student.add_student(student_1, args)
Student.add_student(student_2, args)
Student.add_student(student_3, args)
Student.add_student(student_4, args)
Student.add_student(student_5, args)
Student.add_student(student_6, args)
Student.add_student(student_7, args)
Student.add_student(student_8, args)

# add the students to the class/course
Student.add_to_class(student_1, args)
Student.add_to_class(student_2, args)
Student.add_to_class(student_3, args)
Student.add_to_class(student_4, args)
Student.add_to_class(student_5, args)
Student.add_to_class(student_6, args)
Student.add_to_class(student_7, args)
Student.add_to_class(student_8, args)




