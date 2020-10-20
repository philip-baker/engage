import sqlite3

def create_db():
    conn = sqlite3.connect('engage.db')
    c = conn.cursor()

    c.execute("""CREATE TABLE attendance (
                date text, 
                upi text,
                course_code text,
                attendance integer,
                PRIMARY KEY (date, upi, course_code)
                )""")

    c.execute("""CREATE TABLE students (
                 upi text,
                 student_first text,
                 student_last text,
                 age real,
                 PRIMARY KEY (upi)
                 )""")

    c.execute("""CREATE TABLE features (
                 upi text,
                 features text,
                 PRIMARY KEY (upi)
                 )""")

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

    def __init__(self, upi, first, last, age):
        self.upi = upi
        self.first = first
        self.last = last
        self.age = age

    def add_student(self, args):
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute("""INSERT OR IGNORE INTO students VALUES (:upi, :first, :last, :age)""",
                  {'upi': self.upi, 'first': self.first,
                   'last': self.last, 'age': self.age})
        conn.commit()
        conn.close()

    def add_to_class(self, args):
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute("""INSERT OR IGNORE INTO %s VALUES (:upi)""" % args.code,
                  {'upi': self.upi})
        conn.commit()
        conn.close()
