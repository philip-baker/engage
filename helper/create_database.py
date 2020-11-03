"""Module documentation for create_database.py
   This file contains several functions used to create an SQLite database to interface with the system. It includes severall functions to
   modify the database. In addition, it contains the Student Class. 
"""

import sqlite3

__author__ = "Philip Baker & Keith Spencer-Edgar"
__date__ = "25-10-2020"

def create_db():
    """
        Create an SQLite database (.db) file with three tables: attendance, students, and features. 
       
    """
    conn = sqlite3.connect('engage.db')
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                date text, 
                upi text,
                course_code text,
                attendance integer,
                PRIMARY KEY (date, upi, course_code)
                )""")

    c.execute("""CREATE TABLE IF NOT EXISTS students (
                 upi text,
                 student_first text,
                 student_last text,
                 age real,
                 PRIMARY KEY (upi)
                 )""")

    c.execute("""CREATE TABLE IF NOT EXISTS features (
                 upi text,
                 features text,
                 PRIMARY KEY (upi)
                 )""")

    conn.commit()
    conn.close()


def add_course(name):
    """
        Creates a class in the database

    Parameters:
    ----------
        name: string
            name of class
    """
    conn = sqlite3.connect('engage.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS %s (upi text, PRIMARY KEY(upi))' % name)
    conn.commit()
    conn.close()


class Student:
    """
        A single student in the SQLite database 

    Attributes:
    ----------
        upi (str): The student's UPI (Unique Personal Identifier) 
        first: string
            The student's first name
        last: string 
            The student's last name 
        age: float
            The student's age 
    """

    def __init__(self, upi, first, last, age):
        self.upi = upi
        self.first = first
        self.last = last
        self.age = age

    def add_student(self):
        """
            Adds a student to the SQLite database 
        """
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute("""INSERT OR IGNORE INTO students VALUES (:upi, :first, :last, :age)""",
                  {'upi': self.upi, 'first': self.first,
                   'last': self.last, 'age': self.age})
        conn.commit()
        conn.close()

    def add_to_class(self, class_name):
        """
            Adds a student to a class (roll) in the SQlite database
        Parameters:
        ----------
            class_name: string 
                The name of the class to add the student to

        """
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute("""INSERT OR IGNORE INTO %s VALUES (:upi)""" % class_name,
                  {'upi': self.upi})
        conn.commit()
        conn.close()
