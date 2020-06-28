from keras.models import Model
from tensorflow.keras.layers import Bidirectional
import numpy as np
from keras.preprocessing.text import Tokenizer
import os
from keras.preprocessing.sequence import pad_sequences
from database.database import database
from attention import attention as att
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from AttentionDecoder import AttentionDecoder
import sys

BASE_DIR = 'C:/PythonProjects/title_generate/glove'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 256  # max_review_length   maxlen
MAX_SEQUENCE_LENGTH_TITLE = 256  # max_review_length   maxlen
MAX_NUM_WORDS = 256  # the number of possible tokens,  (1 + maximum word index),    ##max_features
EMBEDDING_DIM = 100  # the dimensionality of the embeddings  embed_size =100
VALIDATION_SPLIT = 0.2
train_title = []
train_content = []
test_title = []
test_content = []
#MAX_NB_WORDS = 100000

num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'
all_data = []
contents = []
titles   = []

class tokenizer:
    def __init__(self):
        db = database()
        db.select(["title", "content", "is_test"])
        # db.where("is_test = false")
        db.table("eng_texts_tab")
        all_datas = db.get_result()

        tot_num_words = 0

        for data in all_datas:
            all_data.append(data[0])
            all_data.append(data[1] )
            titles.append(data[0] )
            contents.append(data[1] )
            tot_num_words += len(data[0])
            tot_num_words += len(data[1])




    def tokenizer(self):
        my_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        my_tokenizer.fit_on_texts( all_data ) # tüm datayı int atıyor
        all_tokenized_title = my_tokenizer.texts_to_sequences( titles ) # atanan intleri, wordler ile replace ediyor
        pad_all_tokenized_title = pad_sequences(all_tokenized_title, maxlen=MAX_SEQUENCE_LENGTH_TITLE)

        all_tokenized_content = my_tokenizer.texts_to_sequences( titles ) # atanan intleri, wordler ile replace ediyor
        pad_all_tokenized_content = pad_sequences(all_tokenized_content, maxlen=MAX_SEQUENCE_LENGTH)

        word_index = my_tokenizer.word_index

        return pad_all_tokenized_content, pad_all_tokenized_title , word_index, my_tokenizer

    def make_glovevec(self, word_index, veclen=300):
        embeddings_index = {}
        with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8") as f:
            for line in f:
                values = line.split()
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs
        f.close()


        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        counter = 0
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # kelime yoksa embedding index 0 olur
                embedding_matrix[i] = embedding_vector
        print("make_glovevec END")


        return embedding_matrix


    def BidLstm(self, embedding_matrix):
        print("BidLSTM START")
        inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
        print(inp)
        model = Sequential()
        model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False) )
        model.add( LSTM(100))
        # model.add( AttentionDecoder(150, MAX_SEQUENCE_LENGTH) )
        model.add( Dense(1000, activation="relu") )
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='softmax'))

        return model



    def listToString(self, s):
        str1 = " "
        return (str1.join(s))

    def tokenizer2(self):
        # Tokenizer objesini başlat
        tokenizer_content = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer_title = Tokenizer(num_words=MAX_NUM_WORDS)
        return_tokenizer_t = tokenizer_title
        return_tokenizer_c = tokenizer_content
        # her kelimeye benzersiz bir sayı ile temsili atar
        tokenizer_content.fit_on_texts(train_content)
        tokenizer_title.fit_on_texts(train_title)

        # her cümleyi bir önceki adımda atanan kelime temsillerini kullanarak bir sayı dizisine çevir.
        list_tokenized_train_content = tokenizer_content.texts_to_sequences(train_content)
        list_tokenized_test_content = tokenizer_content.texts_to_sequences(test_content)
        list_tokenized_train_title = tokenizer_content.texts_to_sequences(train_title)
        # tüm cümleleri aynı uzunlukta(MAX_SEQUENCE_LENGTH) yap
        X_train = pad_sequences(list_tokenized_train_content, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = pad_sequences(list_tokenized_test_content, maxlen=MAX_SEQUENCE_LENGTH)
        Y_train = pad_sequences(list_tokenized_train_title, maxlen=MAX_SEQUENCE_LENGTH)

        word_index = tokenizer_content.word_index  # tüm kelimeler

        return X_train, X_test, Y_train, word_index, return_tokenizer_t, return_tokenizer_c