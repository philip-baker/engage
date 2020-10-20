import sqlite3


def create_db():
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


    """

    def __init__(self, upi, first, last, age):
        self.upi = upi
        self.first = first
        self.last = last
        self.age = age

    def add_student(self):
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute("""INSERT OR IGNORE INTO students VALUES (:upi, :first, :last, :age)""",
                  {'upi': self.upi, 'first': self.first,
                   'last': self.last, 'age': self.age})
        conn.commit()
        conn.close()

    def add_to_class(self, class_name):
        conn = sqlite3.connect('engage.db')
        c = conn.cursor()
        c.execute("""INSERT OR IGNORE INTO %s VALUES (:upi)""" % class_name,
                  {'upi': self.upi})
        conn.commit()
        conn.close()
