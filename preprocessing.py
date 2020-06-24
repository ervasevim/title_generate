from database.database import database
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from snowballstemmer import stemmer
import json
import psycopg2 as psycopg2
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java


ZEMBEREK_PATH = r'C:\Users\casper\PycharmProjects\title_generate\zemberek-full.jar'
startJVM(
    getDefaultJVMPath(),
    '-ea',
    f'-Djava.class.path={ZEMBEREK_PATH}',
    convertStrings=False
)

db = database()
stopWords = set(stopwords.words('turkish'))
lemmatizer = WordNetLemmatizer()
stemmer = stemmer('turkish')

class preprocessing:

    def __init__(self):
        self.preprocess()

    def preprocess(self):

        sql = """ UPDATE texts_tab
                  SET pre_title = %s, pre_content = %s
                  WHERE id = %s"""
        conn = None
        updated_rows = 0
        db.select(["title", "content"])
        db.table("texts_tab")
        datas = db.get_result()

        try:
            params = db.config()  # read database configuration
            conn = psycopg2.connect(**params)  # connect to the PostgreSQL database
            cur = conn.cursor()  # create a new cursor
            for index, data in enumerate(datas, start=52):
                pre_data = self.tokenization(data)
                pre_data = self.remove_punctuation(pre_data)
                pre_data = self.remove_stopword(pre_data)
                pre_data = self.stemming_snowball(pre_data)
                cur.execute(sql,
                            (json.dumps(pre_data[0]), json.dumps(pre_data[1]), index))  # execute the UPDATE  statement
                updated_rows = cur.rowcount  # get the number of updated rows
                conn.commit()  # Commit the changes to the database
                print(index)
            cur.close()  # Close communication with the PostgreSQL database
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()


    def preprocess_eng(self):
        sql = """ UPDATE eng_texts_tab
                         SET pre_title = %s, pre_content = %s
                         WHERE id = %s"""
        conn = None
        updated_rows = 0
        db.select(["title", "content"])
        db.table("eng_texts_tab")
        datas = db.get_result()

        try:
            params = db.config()  # read database configuration
            conn = psycopg2.connect(**params)  # connect to the PostgreSQL database
            cur = conn.cursor()  # create a new cursor
            for index, data in enumerate(datas, start=52):
                pre_data = self.tokenization(data)
                pre_data = self.remove_punctuation(pre_data)
                pre_data = self.remove_stopword(pre_data)
                pre_data = self.lemmatization(pre_data)
                cur.execute(sql,
                            (json.dumps(pre_data[0]), json.dumps(pre_data[1]), index))  # execute the UPDATE  statement
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
        words = [word_tokenize(word) for word in text]
        return words

    def remove_punctuation(self, data):
        # remowing punctuation (^++&%/(/)(=)
        punctuation_words = []
        for text in data:
            punctuation_words.append([word for word in text if word.isalnum()])
        return punctuation_words

    def remove_stopword(self, data):
        # remowing stopword(“the”, “a”, “on”, “ is ”, “all”.)
        removing_stopwords = []
        for text in data:
            removing_stopwords.append([word for word in text if word not in stopWords])
        return removing_stopwords


    def lemmatization(self, data):
        # kelime köklerine ayırma
        lemmatized_word = []
        for text in data:
            lemmatized_word.append([lemmatizer.lemmatize(word) for word in text])
        return lemmatized_word


    def stemming_snowball(self, datas):
        stemming_word = []
        for data in datas:
            stemming_word.append(stemmer.stemWords(data))
        return stemming_word




p1 = preprocessing()
