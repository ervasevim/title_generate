import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tokenizer import tokenizer
import psycopg2 as psycopg2
import sys
from database.database import database
import json

np.random.seed(7)
tk = tokenizer()

if __name__ == "__main__":

    max_features = 100000
    maxlen = 150
    embed_size = 100
    tokenized_content, tokenized_title, word_index, my_tokenizer = tk.tokenizer()
    print(len(tokenized_content))
    print(len(tokenized_title))
    print("tokenized end")
    embedding_vector = tk.make_glovevec(word_index)
    print("glove end")
    print(len(embedding_vector))

    model = tk.BidLstm(embedding_vector)

    print("model compile started")

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    #file_path = "./model.hdf5"
    #ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
    #                       save_best_only=True, mode='min')
    #early = EarlyStopping(monitor="val_loss", mode="min", patience=1)

    print("model fit started")

    #model.fit(xtr, y, batch_size=64, epochs=1, validation_split=0.1, callbacks=[ckpt, early])
    model.fit(tokenized_content[:30000], tokenized_title[:30000], batch_size=512 , epochs=10)

    # save model and architecture to single file
    model.save("model4_new_tokenizer2.h5")
    print("Saved model to disk")
    sys.exit()
    file = open(r"C:/PythonProjects/title_generate/results/new_tokenizer.txt", "w+")

    for content in tokenized_content[20001:]:
        text = []
        encoded_arr = []
        cumle = model.predict(content)
        for kelime_vektor in cumle:
            kelime_sayi = np.argmax(kelime_vektor, axis=0)
            encoded_arr.append(kelime_sayi)

        text = my_tokenizer.sequences_to_texts([encoded_arr])
        file.write(' '.join(text))
        file.write('\r\n')
    file.close()

    sys.exit()
    #******************************************* END CODE ***********************************************************#
    #model.load_weights(file_path)
    y_test = model.predict(xte[1])

    print('test results')
    print(y_test)
    print(y_test.shape)

    for i in range(len(y_test)):
        print("X=%s, Predicted=%s" % (xte[i], y_test[i]))
    '''
    print("X=%s, Predicted=%s" % (xte[1], y_test[1]))
    print("X=%s, Predicted=%s" % (xte[2], y_test[2]))
    print("X=%s, Predicted=%s" % (xte[3], y_test[3]))
    print("X=%s, Predicted=%s" % (xte[4], y_test[4]))
    '''

def get_word_index(self):
    db = database()
    db.select(["pre_title", "pre_content", "is_test"])
    db.table("eng_texts_tab")
    datas = db.get_result()
    token_data = []
    for data in datas:
        token_data.append(self.listToString(data[0]))
        token_data.append(self.listToString(data[1]))

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(token_data)
    list_tokenized = tokenizer.texts_to_sequences(token_data)
    # tokenized_datas = pad_sequences(list_tokenized, maxlen=MAX_SEQUENCE_LENGTH)
    word_index = tokenizer.word_index
    return word_index


