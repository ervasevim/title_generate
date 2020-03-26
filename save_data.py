import tkinter as tk
from database import database
from tkinter import filedialog
import json
import os
import psycopg2 as psycopg2

db = database()
def db_test():  # todo test me
    print("db_test")
    print("connected...")
    db.select(["pre_title","pre_content"])
    db.table("eng_texts_tab")
    return db.get_result()


class save_data:
    texts = db_test()
    totalRows = len(texts)

    def __init__(self, name):
       self.read_json_file()


    def read_json_file(self):
        conn = None
        try:
            # read connection parameters
            params = db.config()
            # connect to the PostgreSQL server
            conn = psycopg2.connect(**params)
            # create a cursor
            cursor = conn.cursor()
            DIR = 'C:/Users/casper/Desktop/Kaynaklar/Bitirme çalışması/datasets/81000 eng news'
            files = os.listdir(DIR)
            total_file = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            for row in files:
                with open('C:/Users/casper/Desktop/Kaynaklar/Bitirme çalışması/datasets/81000 eng news/' + row,
                          encoding="utf8") as json_file:
                    data = json.load(json_file)
                    print(("INSERT INTO eng_texts_tab (title,content) VALUES(%s, %s)", (data["title"], data["text"])))
                    cursor.execute("INSERT INTO eng_texts_tab (title,content) VALUES(%s, %s)", (data["title"], data["text"]))

                    conn.commit()  # <- We MUST commit to reflect the inserted data
            cursor.close()
            conn.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)



p1 = save_data("")
