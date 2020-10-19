import sqlite3


def create_db():
    conn = sqlite3.connect('engage.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE attendance (
                date text, 
                upi text,
                course_code text,
                attendance integer,
                PRIMARY KEY (upi)
                )""")

    c.execute("""CREATE TABLE students (
                 upi text,
                 student_first text,
                 student_last text,
                 nationality text,
                 age real,
                 GPA real,
                 PRIMARY KEY (upi)
                 )""")

    c.execute("""CREATE TABLE features (
                 upi text,
                 features text,
                 PRIMARY KEY (upi)
                 )""")

    # c.execute("""CREATE TABLE lecturers (
    #             lecturer_UPI text,
    #             lecturer_title text,
    #             lecturer_first text,
    #             lecturer_last text
    #             )""")
    conn.commit()
    conn.close()


class EngageDB:
    def __init__(self, args):
        self.args = args

    def add_course(args):
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute('CREATE TABLE %s (upi text)' % args.code)
        conn.commit()
        conn.close()


class Student:
    """A sample student class"""

    def __init__(self, upi, first, last, age, nationality, gpa):
        self.upi = upi
        self.first = first
        self.last = last
        self.age = age
        self.nationality = nationality
        self.gpa = gpa

    def add_student(self, args):
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute("""INSERT OR IGNORE INTO students VALUES (:upi, :first, :last, :nationality, :age, :gpa)""",
                  {'upi': self.upi, 'first': self.first,
                   'last': self.last, 'age': self.age, 'nationality': self.nationality,
                   'gpa': self.gpa})
        conn.commit()
        conn.close()

    def add_to_class(self, args):
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute("""INSERT OR IGNORE INTO %s VALUES (:upi)""" % args.code,
                  {'upi': self.upi})
        conn.commit()
        conn.close()
