import sqlite3
import argparse
from create_database import Student, create_db, EngageDB

# this file creates a sample data base with one class engsci and two students

parser = argparse.ArgumentParser(description='sample database')
parser.add_argument('--code', default='engsci', type=str, help='The course code of the lecture')
args = parser.parse_args()

# create the database
create_db()

# create a class/course
EngageDB.add_course(args)

# create two example students
student_1 = Student('pbak637', 'Philip', 'Baker', 21, 'New Zealand / British', 10)
student_2 = Student('kspe888', 'Keith', 'Spencer-Edgar', 22, 'New Zealand', 10)

# add the students to the database
Student.add_student(student_1, args)
Student.add_student(student_2, args)

# add the students to the class/course
Student.add_to_class(student_1, args)
Student.add_to_class(student_2, args)


# print one student as an example
conn = sqlite3.connect('engage.db')
c = conn.cursor()
c.execute("SELECT * FROM students WHERE student_last = 'Spencer-Edgar'")
print(c.fetchall())
conn.commit()
conn.close()


