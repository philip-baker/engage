import sqlite3
import argparse
from create_database import Student, create_db, EngageDB

# this file creates a sample data base with one class engsci and two students

parser = argparse.ArgumentParser(description='sample database')
parser.add_argument('--code', default='gened', type=str, help='The course code of the lecture')
args = parser.parse_args()

from csv import reader
# open file in read mode
with open('students.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        print(row)

