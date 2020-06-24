from keras.models import Model
from keras.layers import Bidirectional
import numpy as np
from keras.preprocessing.text import Tokenizer
import os
from keras.preprocessing.sequence import pad_sequences
from database.database import database
from attention import attention as att
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

BASE_DIR = 'C:/Users/casper/Desktop/klasörler/tez/glove'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000  # max_review_length   maxlen
MAX_NUM_WORDS = 33000  # the number of possible tokens,  (1 + maximum word index),    ##max_features
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


class tokenizer:
    def __init__(self):
        db = database()
        db.select(["pre_title", "pre_content", "is_test"])
        db.where("is_test = false")
        db.table("eng_texts_tab")
        train_datas = db.get_result()

        db = database()
        db.select(["pre_title", "pre_content", "is_test"])
        db.where("is_test = true")
        db.table("eng_texts_tab")
        test_datas = db.get_result()


        index = 0
        for data in train_datas:
            index += 1
            train_title.append(self.listToString(data[0]))
            train_content.append(self.listToString(data[1]))
            if index == 2000: #1800 train 200 validate
                break
        index2 = 0
        for data in test_datas:
            index2 += 1
            test_title.append(self.listToString(data[0]))
            test_content.append(self.listToString(data[1]))
            if index2 == 10:
                break

        print('Processing text dataset')


    def tokenizer(self):
        # Tokenizer objesini başlat
        tokenizer_content = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer_title = Tokenizer(num_words=MAX_NUM_WORDS)
        # her kelimeye benzersiz bir sayı ile temsili atar
        tokenizer_content.fit_on_texts(train_content)
        tokenizer_title.fit_on_texts(train_title)
        #her cümleyi bir önceki adımda atanan kelime temsillerini kullanarak bir sayı dizisine çevir.
        list_tokenized_train_content = tokenizer_content.texts_to_sequences(train_content)
        list_tokenized_test_content = tokenizer_content.texts_to_sequences(test_content)
        list_tokenized_train_title = tokenizer_content.texts_to_sequences(train_title)
        #tüm cümleleri aynı uzunlukta(MAX_SEQUENCE_LENGTH) yap
        X_train = pad_sequences(list_tokenized_train_content, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = pad_sequences(list_tokenized_test_content, maxlen=MAX_SEQUENCE_LENGTH)
        Y_train = pad_sequences(list_tokenized_train_title, maxlen=MAX_SEQUENCE_LENGTH)

        word_index = tokenizer_content.word_index  #tüm kelimeler

        return X_train, X_test, Y_train, word_index


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
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                #kelime yoksa embedding index 0 olur
                embedding_matrix[i] = embedding_vector
        print("make_glovevec END")


        return embedding_matrix


    def BidLstm(self, embedding_matrix):
        print("BidLSTM START")
        inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
        print(inp)
        x = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix],
                      trainable=False)(inp)
        x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                               recurrent_dropout=0.25))(x)  #300 units
        x = att(MAX_SEQUENCE_LENGTH)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(1000, activation="relu")(x)
        model = Model(inputs=inp, outputs=x)

        return model

    def LSTM(self,embedding_matrix):
        print( "LSTM START")
        xtr, xte, y, word_index = self.tokenizer()
        embedding_layer = Embedding(MAX_NUM_WORDS,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)

        comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(comment_input)
        x = lstm_layer(embedded_sequences)
        x = Dropout(rate_drop_dense)(x)
        merged = att(MAX_SEQUENCE_LENGTH)(x)
        merged = Dense(num_dense, activation=act)(merged)
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)
        preds = Dense(6, activation='sigmoid')(merged)

        ## train the model
        model = Model(inputs=[comment_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())

        STAMP = 'simple_lstm_glove_vectors_%.2f_%.2f' % (rate_drop_lstm, rate_drop_dense)
        print(STAMP)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        bst_model_path = STAMP + '.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

        #hist = model.fit(len(xtr), len(y), validation_data=(xtr, y), epochs=50, batch_size=256, shuffle=True, callbacks=[early_stopping, model_checkpoint])

        hist = model.fit(xtr, y, batch_size=32, epochs=4, validation_split=0.1)

        model.load_weights(bst_model_path)
        bst_val_score = min(hist.history['val_loss'])

    def listToString(self, s):
        str1 = " "
        return (str1.join(s))

