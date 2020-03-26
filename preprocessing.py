from database import database
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import psycopg2 as psycopg2
import nltk

db = database()
nltk.download("punkt")
stopWords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class preprocessing:

    def __init__(self):
       self.preprocess()


    def preprocess(self):
        #1- punc 2- toke 3-stop 4- Stemming or Lemmatization
        #stemming genelde kırparak köke ayırır Lemmatization genelde morfolojik kökünü analiz etmeye çalışır ama ikisi de köke ayırır temelde.
        sql = """ UPDATE eng_texts_tab
                                    SET pre_title = %s, pre_content = %s
                                    WHERE id = %s"""
        conn = None
        updated_rows = 0
        db.select(["title", "content"])
        db.table("eng_texts_tab")
        datas = db.get_result()

        #for index, data in enumerate(datas, start=1):
        #    pre_data = self.tokenization(data)
        #    pre_data = self.remove_punctuation(pre_data)
        #    pre_data = self.remove_stopword(pre_data)
        #    pre_data = self.lemmatization(pre_data)
        #    self.update_data(index, json.dumps(pre_data[0]), json.dumps(pre_data[1]))


        try:
            params = db.config()  # read database configuration
            conn = psycopg2.connect(**params)  # connect to the PostgreSQL database
            cur = conn.cursor()  # create a new cursor
            for index, data in enumerate(datas, start=1):
                pre_data = self.tokenization(data)
                pre_data = self.remove_punctuation(pre_data)
                pre_data = self.remove_stopword(pre_data)
                pre_data = self.lemmatization(pre_data)
                # self.update_data(index, json.dumps(pre_data[0]), json.dumps(pre_data[1]))
                cur.execute(sql, (json.dumps(pre_data[0]), json.dumps(pre_data[1]), index))  # execute the UPDATE  statement
                updated_rows = cur.rowcount  # get the number of updated rows
                conn.commit()  # Commit the changes to the database
                print(index)
            cur.close()  # Close communication with the PostgreSQL database
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        print(updated_rows)

    def tokenization(self, text):
        words = [word_tokenize(word.lower()) for word in text]
        return words

    def remove_punctuation(self, data):
        # remowing punctuation (^++&%/(/)(=)
        punctuation_words = []
        for text in data:
            punctuation_words.append([word for word in text if word.isalnum()])
        return punctuation_words

    def remove_stopword(self, data):
        #remowing stopword(“the”, “a”, “on”, “ is ”, “all”.)
        removing_stopwords = []
        for text in data:
            removing_stopwords.append([word for word in text if word not in stopWords])
        return removing_stopwords

    def lemmatization(self, data):
        #kelime köklerine ayırma
        lemmatized_word = []
        for text in data:
            lemmatized_word.append([lemmatizer.lemmatize(word) for word in text])

     #   print(lemmatized_word)
        return lemmatized_word

    def save_data(self, id, pre_title, pre_content):
        conn = None
        db = database()
        updated_rows = 0
        try:
            print(pre_title)
            params = db.config()
            conn = psycopg2.connect(**params)
            cursor = conn.cursor()
            cursor.execute( """ UPDATE eng_texts_tab
                    SET pre_title = %s, pre_content = %s
                    WHERE id = %s""", (pre_title,pre_content, id))

            print(cursor.query)
            conn.commit()  # <- We MUST commit to reflect the inserted data
            cursor.close()
            conn.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def update_data (self, id, pre_title, pre_content):
        sql = """ UPDATE eng_texts_tab
                    SET pre_title = %s, pre_content = %s
                    WHERE id = %s"""
        conn = None
        db = database()
        updated_rows = 0
        try:
            params = db.config()  # read database configuration
            conn = psycopg2.connect(**params)  # connect to the PostgreSQL database
            cur = conn.cursor()  # create a new cursor
            for index, data in enumerate(datas, start=1):
                pre_data = self.tokenization(data)
                pre_data = self.remove_punctuation(pre_data)
                pre_data = self.remove_stopword(pre_data)
                pre_data = self.lemmatization(pre_data)
                #self.update_data(index, json.dumps(pre_data[0]), json.dumps(pre_data[1]))
                cur.execute(sql, (json.dumps(pre_data[0]), json.dumps(pre_data[1]), id))   # execute the UPDATE  statement
                updated_rows = cur.rowcount   # get the number of updated rows
                conn.commit()   # Commit the changes to the database
            cur.close()   # Close communication with the PostgreSQL database
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        print(updated_rows)
        return updated_rows



p1 = preprocessing()